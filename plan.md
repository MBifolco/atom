# World Model (Dreamer) Training Pipeline — Implementation Plan

## Overview

Add a Dreamer-based training pipeline that produces fighters with the same `decide(snapshot)` interface, then pit them against existing model-free (PPO population) fighters. The key difference: **the policy is trained entirely in imagination** rather than from real environment rewards.

Since the observation space is 13D (not pixels) and physics are deterministic, DreamerV3's heavy machinery (conv encoder/decoder, complex RSSM) simplifies dramatically into an MLP-based world model.

---

## Architecture

### World Model (learns environment dynamics)
```
Transition Model:  MLP (obs_t[13] + action_t[2]) → obs_t+1[13]
Reward Model:      MLP (obs_t[13] + action_t[2]) → reward[1]
Continue Model:    MLP (obs_t[13] + action_t[2]) → P(not_done)[1]
```

No encoder/decoder needed — 13D observations are already compact.
No RSSM recurrence needed — the environment is nearly fully observable (opponent stance_hint provides partial info, but stacking 2-3 frames handles this).

### Actor-Critic (trained in imagination)
```
Actor:   MLP obs[13] → action[2]  (same shape as current PPO policy)
Critic:  MLP obs[13] → value[1]
```

### Training Loop (Dreamer-style)
```
repeat:
  1. COLLECT: Run policy in real env → store transitions in replay buffer
  2. LEARN WORLD MODEL: Sample batches from replay → train dynamics/reward/continue models
  3. IMAGINE: Starting from real states, roll out 15-step trajectories through learned dynamics
  4. LEARN POLICY: Train actor-critic on imagined trajectories (lambda-returns)
```

---

## File Plan

### New Files

| File | Purpose |
|------|---------|
| `src/atom/training/trainers/dreamer/__init__.py` | Package init |
| `src/atom/training/trainers/dreamer/world_model.py` | Transition, reward, continue models (PyTorch) |
| `src/atom/training/trainers/dreamer/actor_critic.py` | Actor and critic networks |
| `src/atom/training/trainers/dreamer/replay_buffer.py` | Experience replay buffer for real transitions |
| `src/atom/training/trainers/dreamer/imagination.py` | Imagination rollout engine (unroll dynamics model) |
| `src/atom/training/trainers/dreamer/dreamer_trainer.py` | Main training loop orchestrator |
| `src/atom/training/trainers/dreamer/dreamer_config.py` | Hyperparameters dataclass |
| `src/atom/training/trainers/dreamer/dreamer_export.py` | Export actor to ONNX + decide() wrapper |
| `src/atom/training/pipelines/dreamer_pipeline.py` | Pipeline: curriculum data collection → dreamer training → export |
| `apps/training/train_dreamer.py` | CLI entry point |
| `tests/test_world_model.py` | World model learns simple dynamics |
| `tests/test_dreamer_imagination.py` | Imagination rollouts produce valid trajectories |
| `tests/test_dreamer_trainer.py` | End-to-end training loop runs |
| `tests/test_dreamer_export.py` | Export produces valid decide() fighter |
| `tests/test_dreamer_vs_population.py` | Dreamer fighter can fight PPO fighter via atom_fight |

### Modified Files

| File | Change |
|------|--------|
| `apps/training/train_progressive.py` | Add `--mode dreamer` option |
| `src/atom/training/pipelines/progressive_trainer.py` | Add `run_dreamer_training()` method |

---

## Implementation Steps

### Step 1: Dreamer Config (`dreamer_config.py`)
Dataclass with all hyperparameters:
- `imagination_horizon`: 15 (steps to unroll in imagination)
- `world_model_lr`: 3e-4
- `actor_lr`: 3e-5
- `critic_lr`: 3e-5
- `replay_buffer_size`: 1_000_000 transitions
- `batch_size`: 256
- `world_model_hidden`: [256, 256] (larger than policy — it needs to learn dynamics)
- `actor_hidden`: [64, 64] (same as current PPO for fair comparison)
- `critic_hidden`: [64, 64]
- `collection_steps_per_epoch`: 1000 (real env steps between world model updates)
- `world_model_train_steps`: 100 (gradient steps per epoch)
- `imagination_train_steps`: 100
- `gamma`: 0.99
- `lambda_`: 0.95 (for lambda-returns in imagination)
- `symlog_rewards`: True (DreamerV3's reward scaling)
- `obs_dim`: 13
- `action_dim`: 2
- `frame_stack`: 1 (increase to 2-3 if partial observability matters)

### Step 2: World Model (`world_model.py`)
**Test first:** `test_world_model.py`
- Test that transition model can overfit on a small batch of real transitions
- Test that reward model predicts correct sign for win/loss
- Test that continue model predicts ~1.0 for mid-episode, ~0.0 for terminal

**Implementation:**
```python
class TransitionModel(nn.Module):
    """(obs_t, action_t) → predicted obs_t+1"""
    # Input: 13 + 2 = 15, Output: 13
    # Loss: MSE on predicted vs actual next obs
    # Optional: predict delta (obs_t+1 - obs_t) for stability

class RewardModel(nn.Module):
    """(obs_t, action_t) → predicted reward"""
    # Input: 15, Output: 1
    # Loss: symlog MSE (DreamerV3 style) or just MSE

class ContinueModel(nn.Module):
    """(obs_t, action_t) → P(episode continues)"""
    # Input: 15, Output: 1 (sigmoid)
    # Loss: binary cross-entropy

class WorldModel:
    """Bundles all three models, provides train_step() and imagine()"""
```

Key design decision: **predict deltas** (`obs_t+1 - obs_t`) rather than absolute next state. This is easier to learn since most obs dimensions change slowly, and it prevents drift during long imagination rollouts.

### Step 3: Replay Buffer (`replay_buffer.py`)
**Test first:** Basic insert/sample, capacity overflow, correct shapes.

```python
class ReplayBuffer:
    """Stores (obs, action, reward, next_obs, done) transitions."""
    def add(self, obs, action, reward, next_obs, done): ...
    def add_batch(self, obs_batch, action_batch, ...): ...  # For vmap envs
    def sample(self, batch_size) → dict[str, Tensor]: ...
    def __len__(self) → int: ...
```

Simple numpy-backed circular buffer. No prioritization needed initially.

### Step 4: Actor-Critic (`actor_critic.py`)
**Test first:** Forward pass shapes, action clamping.

```python
class DreamerActor(nn.Module):
    """obs → action mean + std (continuous, same as PPO policy head)"""
    # Architecture: 64 → 64 → 2 (matching PPO for fair comparison)
    # Uses tanh-squashed Gaussian for bounded actions

class DreamerCritic(nn.Module):
    """obs → value"""
    # Architecture: 64 → 64 → 1
    # Uses symlog targets (DreamerV3)
```

### Step 5: Imagination Engine (`imagination.py`)
**Test first:**
- Rollout produces correct shapes `(horizon, batch, obs_dim)`
- Values propagate gradients back through dynamics

```python
class ImaginationEngine:
    """Generates imagined trajectories through the world model."""

    def imagine(self, start_obs: Tensor, actor: DreamerActor,
                world_model: WorldModel, horizon: int) → ImaginedTrajectory:
        """
        Starting from real observations, unroll actor + dynamics for `horizon` steps.
        Returns imagined obs, actions, rewards, continues — all differentiable.
        """
        # For each step:
        #   action = actor(obs)
        #   next_obs = world_model.transition(obs, action)
        #   reward = world_model.reward(obs, action)
        #   cont = world_model.continue_(obs, action)
        #   obs = next_obs
```

The actor is trained by backpropagating through the imagined trajectory — this is the key advantage over model-free RL. The actor directly optimizes for outcomes in the learned dynamics.

### Step 6: Dreamer Trainer (`dreamer_trainer.py`)
**Test first:** One full epoch runs without error, metrics decrease.

```python
class DreamerTrainer:
    """Main training loop orchestrating real collection + imagination training."""

    def __init__(self, config, env, opponent_fn=None):
        self.world_model = WorldModel(config)
        self.actor = DreamerActor(config)
        self.critic = DreamerCritic(config)
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        self.imagination = ImaginationEngine()
        self.env = env  # VmapEnvWrapper or AtomCombatEnv

    def train(self, total_epochs: int):
        for epoch in range(total_epochs):
            # 1. Collect real experience
            self._collect_experience(self.config.collection_steps_per_epoch)

            # 2. Train world model on replay buffer
            wm_metrics = self._train_world_model(self.config.world_model_train_steps)

            # 3. Imagine + train actor-critic
            ac_metrics = self._train_actor_critic(self.config.imagination_train_steps)

            # 4. Log & checkpoint
            self._log(epoch, wm_metrics, ac_metrics)

    def _collect_experience(self, n_steps):
        """Run actor in real env, store transitions in replay buffer."""
        obs = self.env.reset()
        for _ in range(n_steps):
            action = self.actor.act(obs)  # No grad, add exploration noise
            next_obs, reward, done, truncated, info = self.env.step(action)
            self.replay_buffer.add_batch(obs, action, reward, next_obs, done | truncated)
            obs = next_obs

    def _train_world_model(self, n_steps):
        """Sample from replay, gradient descent on dynamics/reward/continue models."""
        for _ in range(n_steps):
            batch = self.replay_buffer.sample(self.config.batch_size)
            loss = self.world_model.train_step(batch)

    def _train_actor_critic(self, n_steps):
        """Imagine trajectories, compute lambda-returns, update actor+critic."""
        for _ in range(n_steps):
            # Sample starting states from replay buffer
            batch = self.replay_buffer.sample(self.config.batch_size)
            start_obs = batch['obs']

            # Imagine forward
            trajectory = self.imagination.imagine(
                start_obs, self.actor, self.world_model,
                self.config.imagination_horizon
            )

            # Compute lambda-returns
            returns = compute_lambda_returns(
                trajectory.rewards, trajectory.values, trajectory.continues,
                self.config.gamma, self.config.lambda_
            )

            # Update critic (regression on lambda-returns)
            critic_loss = self.critic.train_step(trajectory.obs, returns)

            # Update actor (maximize returns through differentiable dynamics)
            actor_loss = self.actor.train_step(trajectory, returns)
```

### Step 7: Export (`dreamer_export.py`)
**Test first:** Exported fighter loads and returns valid actions.

Reuse the ONNX export pattern from `population_persistence.py`, but export only the actor network. Generate the same `decide()` wrapper template.

### Step 8: Pipeline Integration (`dreamer_pipeline.py`)
```python
class DreamerPipeline:
    """
    Two-phase pipeline:
    1. Curriculum collection: Run a random/simple policy against curriculum
       opponents to seed the replay buffer with diverse experience
    2. Dreamer training: Train world model + actor-critic with imagination
    """
```

**Opponent strategy during real data collection:**
- Phase A: Collect against curriculum dummies (levels 1-5) for diverse data
- Phase B: Collect against self (the actor's current policy) for self-play refinement
- This mirrors how the existing pipeline does curriculum → population, but the policy improvement happens in imagination rather than real interaction

### Step 9: CLI Entry Point (`train_dreamer.py`)
```bash
python apps/training/train_dreamer.py \
  --epochs 500 \
  --collection-steps 1000 \
  --imagination-horizon 15 \
  --use-vmap \
  --device auto \
  --output-dir outputs/dreamer
```

### Step 10: Cross-Training Evaluation
After both pipelines produce fighters:
```bash
# Dreamer fighter vs Population fighter
python apps/cli/atom_fight.py \
  fighters/AIs/dreamer_champion/dreamer_champion.py \
  fighters/AIs/population_champion/population_champion.py \
  --best-of 100 --html dreamer_vs_population.html
```

---

## Testing Strategy (TDD per PERMANENT_CONTEXT.md)

Each step writes tests first:

1. **test_world_model.py** — Transition model overfits small batch, reward model predicts correct sign, continue model predicts terminal states
2. **test_dreamer_imagination.py** — Rollout shapes correct, gradients flow through imagination, trajectories stay in valid observation bounds
3. **test_dreamer_trainer.py** — Full epoch runs, world model loss decreases, actor loss changes
4. **test_dreamer_export.py** — ONNX export produces loadable model, decide() returns valid format, action bounds respected
5. **test_dreamer_vs_population.py** — Both fighters load, match completes without error, results are recorded

---

## Key Design Decisions

1. **Predict deltas not absolutes** — The transition model predicts `obs_t+1 - obs_t` to prevent drift in long imagination rollouts
2. **Same actor architecture (64→64)** — Fair comparison with PPO population fighters; any performance difference is due to training method, not capacity
3. **Symlog reward scaling** — DreamerV3's key insight for handling the wide reward range (-200 to +150)
4. **No RSSM/recurrence** — Environment is nearly fully observable at 13D; recurrence adds complexity without benefit
5. **Exploration noise during collection** — Add Gaussian noise to actor actions during real data collection (not during imagination)
6. **Curriculum opponents for data diversity** — Use the existing 29 test dummies to seed diverse experience before self-play

---

## Implementation Order

Build bottom-up, test each layer before moving to the next:

1. `dreamer_config.py` — Pure dataclass, no deps
2. `replay_buffer.py` + tests — Pure data structure
3. `world_model.py` + tests — PyTorch models, train on synthetic data
4. `actor_critic.py` + tests — PyTorch models, forward pass validation
5. `imagination.py` + tests — Combines world model + actor
6. `dreamer_trainer.py` + tests — Full loop with real env
7. `dreamer_export.py` + tests — ONNX export + decide() wrapper
8. `dreamer_pipeline.py` — Pipeline orchestration
9. `train_dreamer.py` — CLI entry point
10. Integration into `progressive_trainer.py` — Add dreamer mode
11. Cross-training fight evaluation
