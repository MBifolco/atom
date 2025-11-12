"""
JAX-compatible opponent decision functions for vmap training.

These are JAX/JIT-compatible versions of the test dummy opponents,
allowing different opponents to be used across vmapped environments.
"""

import jax
import jax.numpy as jnp
from jax import lax


def stationary_neutral_jax(state, config):
    """Stationary neutral stance - just stands still."""
    return jnp.array([0.0, 0])  # [acceleration, stance_int]


def stationary_extended_jax(state, config):
    """Stationary extended stance - stands still in extended."""
    return jnp.array([0.0, 1])  # [acceleration, stance_int]


def stationary_defending_jax(state, config):
    """Stationary defending stance - stands still in defending."""
    return jnp.array([0.0, 3])  # [acceleration, stance_int]


def stationary_retracted_jax(state, config):
    """Stationary retracted stance - stands still in retracted."""
    return jnp.array([0.0, 2])  # [acceleration, stance_int]


def approach_slow_jax(state, config):
    """Slow approach - moves toward opponent at speed 1.5."""
    my_pos = state.fighter_b.position
    arena_width = config.arena_width
    my_vel = state.fighter_b.velocity

    # Determine direction based on position
    accel = lax.cond(
        my_pos < arena_width * 0.3,
        lambda _: 1.5,  # Left side -> move right
        lambda _: lax.cond(
            my_pos > arena_width * 0.7,
            lambda _: -1.5,  # Right side -> move left
            lambda _: lax.cond(
                my_vel > 0,
                lambda _: 1.5,  # Moving right -> continue
                lambda _: lax.cond(
                    my_vel < 0,
                    lambda _: -1.5,  # Moving left -> continue
                    lambda _: 1.5  # Stopped -> default right
                )
            )
        ),
        None
    )

    return jnp.array([accel, 0])  # neutral stance


def flee_always_jax(state, config):
    """Always flees from opponent."""
    my_pos = state.fighter_b.position
    arena_width = config.arena_width
    my_vel = state.fighter_b.velocity

    # Flee direction (opposite of approach)
    accel = lax.cond(
        my_pos < arena_width * 0.3,
        lambda _: -1.5,  # Left side -> move left (away)
        lambda _: lax.cond(
            my_pos > arena_width * 0.7,
            lambda _: 1.5,  # Right side -> move right (away)
            lambda _: lax.cond(
                my_vel > 0,
                lambda _: 1.5,  # Moving right -> continue away
                lambda _: lax.cond(
                    my_vel < 0,
                    lambda _: -1.5,  # Moving left -> continue away
                    lambda _: -1.5  # Stopped -> default left
                )
            )
        ),
        None
    )

    return jnp.array([accel, 0])  # neutral stance


def circle_left_jax(state, config):
    """Always circles left (constant -2.0 accel)."""
    my_pos = state.fighter_b.position

    # Bounce off left wall
    accel = lax.cond(
        my_pos < 1.0,
        lambda _: 2.0,  # Bounce right
        lambda _: -2.0,  # Default left
        None
    )

    return jnp.array([accel, 0])  # neutral stance


def circle_right_jax(state, config):
    """Always circles right (constant 2.0 accel)."""
    my_pos = state.fighter_b.position
    arena_width = config.arena_width

    # Bounce off right wall
    accel = lax.cond(
        my_pos > arena_width - 1.0,
        lambda _: -2.0,  # Bounce left
        lambda _: 2.0,  # Default right
        None
    )

    return jnp.array([accel, 0])  # neutral stance


def distance_keeper_1m_jax(state, config):
    """Maintains 1m distance from opponent."""
    my_pos = state.fighter_b.position
    my_vel = state.fighter_b.velocity
    arena_width = config.arena_width

    # Calculate distance between fighters
    opp_pos = state.fighter_a.position
    distance = jnp.abs(opp_pos - my_pos)

    target_distance = 1.0
    tolerance = 0.2

    # Determine opponent direction
    opponent_to_right = lax.cond(
        my_pos < arena_width * 0.3,
        lambda _: True,
        lambda _: lax.cond(
            my_pos > arena_width * 0.7,
            lambda _: False,
            lambda _: my_vel >= 0,
            None
        ),
        None
    )

    # Distance control
    accel = lax.cond(
        distance > target_distance + tolerance,
        lambda _: lax.cond(opponent_to_right, lambda _: 2.0, lambda _: -2.0, None),  # Too far: approach
        lambda _: lax.cond(
            distance < target_distance - tolerance,
            lambda _: lax.cond(opponent_to_right, lambda _: -2.0, lambda _: 2.0, None),  # Too close: back away
            lambda _: 0.0,  # Perfect distance
            None
        ),
        None
    )

    # Use extended stance at optimal range
    stance = lax.cond(
        jnp.abs(distance - target_distance) < tolerance,
        lambda _: 1,  # extended
        lambda _: 0,  # neutral
        None
    )

    return jnp.array([accel, stance])


def distance_keeper_3m_jax(state, config):
    """Maintains 3m distance from opponent."""
    my_pos = state.fighter_b.position
    my_vel = state.fighter_b.velocity
    arena_width = config.arena_width
    opp_pos = state.fighter_a.position
    distance = jnp.abs(opp_pos - my_pos)

    target_distance = 3.0
    tolerance = 0.5

    opponent_to_right = lax.cond(
        my_pos < arena_width * 0.3,
        lambda _: True,
        lambda _: lax.cond(
            my_pos > arena_width * 0.7,
            lambda _: False,
            lambda _: my_vel >= 0,
            None
        ),
        None
    )

    accel = lax.cond(
        distance > target_distance + tolerance,
        lambda _: lax.cond(opponent_to_right, lambda _: 2.0, lambda _: -2.0, None),
        lambda _: lax.cond(
            distance < target_distance - tolerance,
            lambda _: lax.cond(opponent_to_right, lambda _: -2.0, lambda _: 2.0, None),
            lambda _: 0.0,
            None
        ),
        None
    )

    stance = lax.cond(
        jnp.abs(distance - target_distance) < tolerance,
        lambda _: 1,  # extended
        lambda _: 0,  # neutral
        None
    )

    return jnp.array([accel, stance])


def distance_keeper_5m_jax(state, config):
    """Maintains 5m distance from opponent."""
    my_pos = state.fighter_b.position
    my_vel = state.fighter_b.velocity
    arena_width = config.arena_width
    opp_pos = state.fighter_a.position
    distance = jnp.abs(opp_pos - my_pos)

    target_distance = 5.0
    tolerance = 0.8

    opponent_to_right = lax.cond(
        my_pos < arena_width * 0.3,
        lambda _: True,
        lambda _: lax.cond(
            my_pos > arena_width * 0.7,
            lambda _: False,
            lambda _: my_vel >= 0,
            None
        ),
        None
    )

    accel = lax.cond(
        distance > target_distance + tolerance,
        lambda _: lax.cond(opponent_to_right, lambda _: 2.0, lambda _: -2.0, None),
        lambda _: lax.cond(
            distance < target_distance - tolerance,
            lambda _: lax.cond(opponent_to_right, lambda _: -2.0, lambda _: 2.0, None),
            lambda _: 0.0,
            None
        ),
        None
    )

    return jnp.array([accel, 0])  # neutral stance


def stamina_efficient_jax(state, config):
    """Conservative stamina management."""
    my_stamina = state.fighter_b.stamina
    max_stamina = config.max_stamina
    stamina_pct = my_stamina / max_stamina

    # Conservative stamina management
    stance = lax.cond(
        stamina_pct > 0.8,
        lambda _: 1,  # High stamina: extended
        lambda _: lax.cond(
            stamina_pct < 0.3,
            lambda _: 2,  # Low stamina: retracted
            lambda _: 0,  # Normal: neutral
            None
        ),
        None
    )

    return jnp.array([0.0, stance])  # Stationary


def stamina_waster_jax(state, config):
    """Always uses extended stance to waste stamina."""
    return jnp.array([0.0, 1])  # Stationary extended


def stamina_cycler_jax(state, config):
    """Cycles through stances based on stamina level."""
    my_stamina = state.fighter_b.stamina
    max_stamina = config.max_stamina
    stamina_pct = my_stamina / max_stamina

    # Cycle stances based on stamina
    stance = lax.cond(
        stamina_pct > 0.66,
        lambda _: 1,  # extended (drains stamina)
        lambda _: lax.cond(
            stamina_pct > 0.33,
            lambda _: 0,  # neutral
            lambda _: 2,  # retracted (regens stamina)
            None
        ),
        None
    )

    return jnp.array([0.0, stance])


def charge_on_approach_jax(state, config):
    """Charges when opponent is approaching."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)

    # Charge (extended stance) when close
    stance = lax.cond(
        distance < 2.0,
        lambda _: 1,  # extended when close
        lambda _: 0,  # neutral when far
        None
    )

    return jnp.array([0.0, stance])


def wall_hugger_left_jax(state, config):
    """Stays near left wall."""
    my_pos = state.fighter_b.position

    # Move toward left wall (position 0)
    accel = lax.cond(
        my_pos < 1.5,
        lambda _: 0.0,  # At wall, stop
        lambda _: -1.5,  # Move to wall
        None
    )

    return jnp.array([accel, 0])  # neutral stance


def wall_hugger_right_jax(state, config):
    """Stays near right wall."""
    my_pos = state.fighter_b.position
    arena_width = config.arena_width

    # Move toward right wall
    accel = lax.cond(
        my_pos > arena_width - 1.5,
        lambda _: 0.0,  # At wall, stop
        lambda _: 1.5,  # Move to wall
        None
    )

    return jnp.array([accel, 0])  # neutral stance


# Shuttle patterns (back and forth movement)
def shuttle_slow_jax(state, config):
    """Shuttles back and forth slowly."""
    my_pos = state.fighter_b.position
    arena_width = config.arena_width

    # Bounce between walls
    accel = lax.cond(
        my_pos < 2.0,
        lambda _: 1.0,  # Near left wall -> go right
        lambda _: lax.cond(
            my_pos > arena_width - 2.0,
            lambda _: -1.0,  # Near right wall -> go left
            lambda _: lax.cond(
                state.fighter_b.velocity > 0,
                lambda _: 1.0,  # Moving right -> continue
                lambda _: -1.0,  # Moving left -> continue
                None
            ),
            None
        ),
        None
    )

    return jnp.array([accel, 0])  # neutral stance


def shuttle_medium_jax(state, config):
    """Shuttles back and forth at medium speed."""
    my_pos = state.fighter_b.position
    arena_width = config.arena_width

    accel = lax.cond(
        my_pos < 2.0,
        lambda _: 1.8,
        lambda _: lax.cond(
            my_pos > arena_width - 2.0,
            lambda _: -1.8,
            lambda _: lax.cond(
                state.fighter_b.velocity > 0,
                lambda _: 1.8,
                lambda _: -1.8,
                None
            ),
            None
        ),
        None
    )

    return jnp.array([accel, 0])


# Opponent registry with integer IDs
JAX_OPPONENT_REGISTRY = {
    # Level 1: Fundamentals (stationary)
    "stationary_neutral": (0, stationary_neutral_jax),
    "stationary_extended": (1, stationary_extended_jax),
    "stationary_defending": (2, stationary_defending_jax),
    "stationary_retracted": (3, stationary_retracted_jax),

    # Level 2: Basic Skills (simple movement)
    "approach_slow": (4, approach_slow_jax),
    "flee_always": (5, flee_always_jax),
    "shuttle_slow": (6, shuttle_slow_jax),
    "shuttle_medium": (7, shuttle_medium_jax),
    "circle_left": (8, circle_left_jax),
    "circle_right": (9, circle_right_jax),

    # Level 3: Intermediate (distance/stamina)
    "distance_keeper_1m": (10, distance_keeper_1m_jax),
    "distance_keeper_3m": (11, distance_keeper_3m_jax),
    "distance_keeper_5m": (12, distance_keeper_5m_jax),
    "stamina_waster": (13, stamina_waster_jax),
    "stamina_cycler": (14, stamina_cycler_jax),
    "stamina_efficient": (15, stamina_efficient_jax),
    "charge_on_approach": (16, charge_on_approach_jax),
    "wall_hugger_left": (17, wall_hugger_left_jax),
    "wall_hugger_right": (18, wall_hugger_right_jax),
}


def create_multi_opponent_func(opponent_paths, config):
    """
    Create a JAX function that selects different opponents based on environment index.

    Args:
        opponent_paths: List of opponent file paths
        config: WorldConfig

    Returns:
        A JIT-compiled function that takes (batched_states) and returns batched opponent actions
    """
    from pathlib import Path

    # Map opponent paths to JAX functions
    opponent_funcs = []
    for path in opponent_paths:
        name = Path(path).stem
        if name in JAX_OPPONENT_REGISTRY:
            opponent_funcs.append(JAX_OPPONENT_REGISTRY[name][1])
        else:
            # Fallback to stationary neutral
            opponent_funcs.append(stationary_neutral_jax)

    n_opponents = len(opponent_funcs)

    # Create wrapper functions that capture config in closure
    wrapped_funcs = [
        lambda s, cfg=config, func=f: func(s, cfg)
        for f in opponent_funcs
    ]

    # Create vmapped selector for batch processing
    def single_opponent_decide(state, env_idx):
        """Select and execute opponent logic for a single environment."""
        # Distribute environments evenly across opponents
        n_envs = 250  # Fixed for now (could be dynamic)
        envs_per_opponent = n_envs // n_opponents
        opponent_idx = env_idx // envs_per_opponent
        # Clamp to avoid index out of bounds
        opponent_idx = jnp.minimum(opponent_idx, n_opponents - 1)

        # Use switch to select opponent function (config captured in closure)
        return lax.switch(
            opponent_idx,
            wrapped_funcs,
            state
        )

    # Return vmapped version that handles batches
    return jax.jit(jax.vmap(single_opponent_decide))
