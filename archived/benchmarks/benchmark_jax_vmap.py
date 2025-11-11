#!/usr/bin/env python3
"""
Benchmark JAX vmap for Parallel Episode Execution

This is where JAX truly shines - vectorizing 100s of episodes to run in parallel.
"""

import sys
from pathlib import Path
import time
import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np

project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.arena.arena_1d import Arena1D
from src.arena.arena_1d_jax_jit import Arena1DJAXJit, FighterStateJAX, ArenaStateJAX, stance_to_int
from src.arena.fighter import FighterState
from src.arena.world_config import WorldConfig


def run_episode_python(seed: int, config: WorldConfig, max_ticks: int = 250) -> int:
    """Run a single episode with Python physics."""
    fighter_a = FighterState.create("Alice", 70.0, 2.0, config)
    fighter_b = FighterState.create("Bob", 75.0, 10.0, config)

    arena = Arena1D(fighter_a, fighter_b, config, seed=seed)

    np.random.seed(seed)
    for tick in range(max_ticks):
        accel_a = np.random.uniform(-1, 1)
        accel_b = np.random.uniform(-1, 1)
        stance_a = np.random.choice(["neutral", "extended", "retracted", "defending"])
        stance_b = np.random.choice(["neutral", "extended", "retracted", "defending"])

        action_a = {"acceleration": accel_a, "stance": stance_a}
        action_b = {"acceleration": accel_b, "stance": stance_b}

        arena.step(action_a, action_b)

        if arena.is_finished():
            break

    return tick + 1


def run_episode_jax_single(seed: int, config: WorldConfig, max_ticks: int = 250) -> int:
    """Run a single episode with JAX JIT physics (no vmap)."""
    fighter_a = FighterState.create("Alice", 70.0, 2.0, config)
    fighter_b = FighterState.create("Bob", 75.0, 10.0, config)

    arena = Arena1DJAXJit(fighter_a, fighter_b, config, seed=seed)

    np.random.seed(seed)
    for tick in range(max_ticks):
        accel_a = np.random.uniform(-1, 1)
        accel_b = np.random.uniform(-1, 1)
        stance_a = np.random.choice(["neutral", "extended", "retracted", "defending"])
        stance_b = np.random.choice(["neutral", "extended", "retracted", "defending"])

        action_a = {"acceleration": accel_a, "stance": stance_a}
        action_b = {"acceleration": accel_b, "stance": stance_b}

        arena.step(action_a, action_b)

        if arena.is_finished():
            break

    return tick + 1


# Create vectorized step function
@jit
def vectorized_step_batch(states, actions_a, actions_b, config_values, stance_arrays):
    """
    Run one step for a batch of episodes in parallel.

    Args:
        states: Array of ArenaStateJAX (shape: [n_episodes])
        actions_a: Array of actions for fighter A (shape: [n_episodes, 2])  # [accel, stance]
        actions_b: Array of actions for fighter B (shape: [n_episodes, 2])
        config_values: Tuple of config scalars
        stance_arrays: Tuple of stance arrays

    Returns:
        new_states: Updated states
    """
    dt, max_accel, max_vel, friction, arena_width, stamina_accel_cost, stamina_base_regen, stamina_neutral_bonus = config_values
    stance_reach, stance_defense, stance_drain = stance_arrays

    def single_step(state, action_a, action_b):
        """Step function for a single episode (to be vmapped)."""
        # Convert actions to dict format
        action_a_dict = {"acceleration": action_a[0], "stance": jnp.int32(action_a[1])}
        action_b_dict = {"acceleration": action_b[0], "stance": jnp.int32(action_b[1])}

        new_state, _ = Arena1DJAXJit._jax_step_jit(
            state, action_a_dict, action_b_dict,
            dt, max_accel, max_vel, friction, arena_width,
            stamina_accel_cost, stamina_base_regen, stamina_neutral_bonus,
            stance_reach, stance_defense, stance_drain
        )
        return new_state

    # vmap across the batch dimension (axis 0)
    return vmap(single_step)(states, actions_a, actions_b)


def run_episodes_vmap(n_episodes: int, config: WorldConfig, max_ticks: int = 250):
    """Run multiple episodes in parallel using vmap."""

    # Prepare config values
    dt = config.dt
    max_accel = config.max_acceleration
    max_vel = config.max_velocity
    friction = config.friction
    arena_width = config.arena_width
    stamina_accel_cost = config.stamina_accel_cost
    stamina_base_regen = config.stamina_base_regen
    stamina_neutral_bonus = config.stamina_neutral_bonus
    config_values = (dt, max_accel, max_vel, friction, arena_width, stamina_accel_cost, stamina_base_regen, stamina_neutral_bonus)

    # Prepare stance arrays
    from src.arena.arena_1d_jax_jit import create_stance_arrays
    stance_arrays = create_stance_arrays(config)

    # Initialize all episodes
    fighter_a = FighterState.create("Alice", 70.0, 2.0, config)
    fighter_b = FighterState.create("Bob", 75.0, 10.0, config)

    # Create initial states for all episodes
    initial_states = []
    for i in range(n_episodes):
        jax_fighter_a = FighterStateJAX.from_fighter_state(fighter_a)
        jax_fighter_b = FighterStateJAX.from_fighter_state(fighter_b)
        state = ArenaStateJAX(jax_fighter_a, jax_fighter_b, 0)
        initial_states.append(state)

    # Stack into batch (this is tricky with pytrees - need to use tree_map)
    from jax import tree_map
    states = jax.tree_map(lambda *xs: jnp.stack(xs), *initial_states)

    # Generate random actions for all episodes (pre-generate for determinism)
    rng = np.random.RandomState(42)
    all_actions_a = rng.uniform(-1, 1, size=(max_ticks, n_episodes, 2))  # [accel, stance_int]
    all_actions_a[:, :, 1] = rng.randint(0, 4, size=(max_ticks, n_episodes))  # stance as int
    all_actions_b = rng.uniform(-1, 1, size=(max_ticks, n_episodes, 2))
    all_actions_b[:, :, 1] = rng.randint(0, 4, size=(max_ticks, n_episodes))

    # Convert to JAX arrays
    all_actions_a = jnp.array(all_actions_a)
    all_actions_b = jnp.array(all_actions_b)

    # Run all ticks in parallel
    total_ticks = 0
    for tick in range(max_ticks):
        actions_a = all_actions_a[tick]
        actions_b = all_actions_b[tick]

        states = vectorized_step_batch(states, actions_a, actions_b, config_values, stance_arrays)
        total_ticks += n_episodes

        # Check if all episodes finished (in real training, we'd handle this properly)
        # For now, just run all max_ticks for simplicity

    return total_ticks


def benchmark_vmap():
    """Benchmark vmap speedup."""
    print("\n" + "="*80)
    print("JAX VMAP PARALLEL EPISODE EXECUTION BENCHMARK")
    print("="*80)

    config = WorldConfig()
    max_ticks = 250

    # Warm up JIT compilation
    print("\n⏳ Warming up JIT compilation...")
    run_episodes_vmap(10, config, max_ticks=10)
    print("✅ JIT compilation complete")

    # Benchmark different batch sizes
    batch_sizes = [1, 10, 50, 100, 250, 500]

    print("\n" + "="*80)
    print("VMAP SCALABILITY TEST")
    print("="*80)
    print(f"\nConfig: {max_ticks} ticks per episode")
    print()

    results = []

    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size:,} episodes...")

        start = time.time()
        total_ticks = run_episodes_vmap(batch_size, config, max_ticks)
        elapsed = time.time() - start

        ticks_per_sec = total_ticks / elapsed
        episodes_per_sec = batch_size / elapsed

        results.append({
            "batch_size": batch_size,
            "elapsed": elapsed,
            "ticks_per_sec": ticks_per_sec,
            "episodes_per_sec": episodes_per_sec
        })

        print(f"  ✅ {elapsed:.3f}s | {ticks_per_sec:,.0f} ticks/sec | {episodes_per_sec:.1f} episodes/sec")

    # Compare with Python baseline
    print("\n" + "="*80)
    print("COMPARISON WITH PYTHON")
    print("="*80)

    python_tps = 57088  # From Phase 0 benchmark
    best_vmap_tps = max(r["ticks_per_sec"] for r in results)
    best_batch = [r for r in results if r["ticks_per_sec"] == best_vmap_tps][0]

    print(f"\nPython (single episode):  {python_tps:,} ticks/sec")
    print(f"JAX vmap (batch={best_batch['batch_size']}):  {best_vmap_tps:,.0f} ticks/sec")
    print(f"\nSpeedup: {best_vmap_tps / python_tps:.2f}x")

    if best_vmap_tps > python_tps:
        print(f"🚀 JAX vmap is {best_vmap_tps / python_tps:.2f}x FASTER!")
    else:
        print(f"⚠️  Still slower than Python (need larger batches or GPU)")

    # Show scaling efficiency
    print("\n" + "="*80)
    print("BATCH SCALING EFFICIENCY")
    print("="*80)
    print("\nBatch Size | Episodes/sec | Scaling")
    print("-" * 50)

    baseline_eps = results[0]["episodes_per_sec"]
    for r in results:
        scaling = r["episodes_per_sec"] / baseline_eps
        print(f"{r['batch_size']:>10} | {r['episodes_per_sec']:>12.1f} | {scaling:>6.2f}x")

    print("\n" + "="*80)


if __name__ == "__main__":
    benchmark_vmap()
