# Training Backend Abstraction + SBX SAC Migration

## Goal

Make the training algorithm a pluggable component so we can:
1. Keep PPO (SB3) working as-is
2. Add SAC via SBX (all-JAX, ~10x faster)
3. Support future algorithms without touching orchestration code

## Current State

The `algorithm` parameter exists but the abstraction is leaky:
- `from stable_baselines3 import PPO, SAC` hardcoded in 10+ files
- PPO-specific assumptions in callback timing, NaN recovery, checkpoint format
- `model.predict()` / `model.learn()` called directly with PPO semantics
- Device management assumes PyTorch ("cpu"/"cuda")
- Model mutation in population assumes PyTorch `state_dict()`

---

## Phase 1A: Backend Protocol + PPO Wrapper (PPO-only, no SAC)

### Protocol design

**New file**: `src/atom/training/backends/protocol.py`

```python
class BackendCapabilities:
    name: str                    # "sb3_ppo", "sbx_sac", etc.
    framework: str               # "pytorch", "jax"
    on_policy: bool              # True=PPO (rollouts), False=SAC (replay buffer)

class TrainingBackend(Protocol):

    @property
    def capabilities(self) -> BackendCapabilities: ...

    def create_model(self, envs, seed: int) -> Any:
        """Create a new model for the given environment."""

    def load_model(self, path: str, envs=None) -> Any:
        """Load a model from disk."""

    def save_model(self, model, path: str) -> None:
        """Save model to disk."""

    def predict(self, model, obs, deterministic: bool = False) -> np.ndarray:
        """Get action from observation."""

    def learn(self, model, total_timesteps: int, callback=None,
              reset_num_timesteps: bool = True, progress_bar: bool = False) -> Any:
        """Train the model."""

    def replace_env(self, model, envs) -> None:
        """Replace the model's environment (level transitions, pool refresh).
        Backend handles any internal fixups (e.g., resetting cached obs)."""

    def get_policy_arch(self) -> dict:
        """Return shared network architecture config (net_arch, activation).
        Separate from training hyperparams (LR, batch size, etc.)."""

    def reduce_learning_rate(self, model, factor: float) -> None:
        """Reduce learning rate (for NaN recovery)."""
```

Key design decisions (from Codex feedback):
- `predict()` takes `deterministic` parameter — holdout checks and sanity checks need both modes
- `replace_env()` instead of `set_env()` + `reset_after_set_env()` — one method, backend handles internals
- `get_policy_arch()` not `get_policy_kwargs()` — separates architecture from training hyperparams
- No replay buffer, checkpoint, or mutation methods yet — those come in Phase 1B and Phase 2

### SB3PPOBackend implementation

**New file**: `src/atom/training/backends/sb3_ppo.py`

Wraps existing logic:
- `create_model()` → current `get_stable_ppo_config()` + `PPO(...)`
- `load_model()` → `PPO.load(path, device="cpu")`
- `save_model()` → `model.save(path)`
- `predict()` → `model.predict(obs, deterministic=deterministic)`
- `learn()` → `model.learn(...)`
- `replace_env()` → `model.set_env(envs)` + `model._last_obs = envs.reset()` + `_last_episode_starts`
- `get_policy_arch()` → `{"net_arch": [256, 256], "activation_fn": nn.ReLU, ...}`
- `reduce_learning_rate()` → `model.learning_rate *= factor`

Training hyperparams (LR, n_steps, batch_size, etc.) live inside `SB3PPOBackend`, not shared.

### Update consumers

**`curriculum_components.py`** — `ModelFactory`:
- Accept `backend: TrainingBackend`
- `create_model()` delegates to `backend.create_model()`
- Remove if/else PPO/SAC branches

**`curriculum_trainer.py`**:
- Accept `backend: TrainingBackend` in constructor
- All `PPO.load()` → `backend.load_model()`
- All `model.save()` → `backend.save_model()`
- All `model.predict()` → `backend.predict()`
- All set_env + reset pattern → `backend.replace_env()`
- NaN recovery LR reduction → `backend.reduce_learning_rate()`

**`train_progressive.py`**:
- Add `--backend` flag: `sb3_ppo` (default)
- Construct backend, pass to trainers

### What this does NOT change
- Population trainer (Phase 1B)
- Model mutation/cloning (Phase 1B)
- Checkpoint format (stays SB3 .zip)
- No new algorithms

### Tests
- All existing tests pass with SB3PPOBackend as drop-in
- New: `test_backend_protocol.py` — verify SB3PPOBackend satisfies protocol
- Smoke test: curriculum L1 trains and graduates

**Estimated effort**: 2-3 days. Pure refactor, no behavioral changes.

---

## Phase 1B: Population Backend Seam

### Add mutation/cloning to backend

```python
class TrainingBackend(Protocol):
    # ... existing methods ...

    def clone_and_mutate(self, model, envs, mutation_rate: float) -> Any:
        """Clone a model's weights and apply random mutation.
        Returns a new model with mutated weights."""

    def checkpoint_training_state(self, model) -> dict:
        """Capture full training state for resume.
        PPO: model params + optimizer.
        SAC: model params + optimizer + replay buffer + entropy coef.
        Returns serializable dict."""

    def restore_training_state(self, model, state: dict) -> None:
        """Restore full training state from checkpoint."""
```

### SB3PPOBackend additions
- `clone_and_mutate()` → current `_clone_and_mutate_model()` from `population_evolution.py`
  - `model.policy.state_dict()` → mutate tensors → `load_state_dict()`
- `checkpoint_training_state()` → model save + VecNormalize stats
- `restore_training_state()` → model load + restore stats

### Update population trainer
- `population_trainer.py`: Use `backend.create_model()`, `backend.clone_and_mutate()`
- `population_evolution.py`: Delegate mutation to backend instead of inline PyTorch ops
- Remove direct `from stable_baselines3 import PPO, SAC` imports

### Tests
- Existing population tests pass
- Test: clone_and_mutate produces different weights
- Test: checkpoint/restore round-trip

**Estimated effort**: 2 days.

---

## Phase 2: SBX SAC Backend (separate project)

### Prerequisites
- Phase 1A and 1B complete
- Run 8 results evaluated

### Hard problems to solve BEFORE implementation

#### 1. Replay buffer policy on distribution shifts

SAC stores past experience in a replay buffer. When the environment changes (level transition, opponent pool refresh), old experience becomes partially invalid.

**Decision**: Add `handle_distribution_shift()` to the backend protocol:
```python
def handle_distribution_shift(self, model, kind: str) -> None:
    """Called on level transition or pool refresh.
    kind: 'level_transition' | 'pool_refresh'
    """
```

**Policy**:
- `level_transition`: Clear replay buffer entirely. Old experience has different opponents and reward distributions. SAC needs `learning_starts` steps before training resumes. Safe but wasteful.
- `pool_refresh`: Keep buffer. Same level, similar experience, just fewer opponents. Old transitions are still mostly valid.

PPO backend: no-op (on-policy, no buffer to clear).

#### 2. SAC checkpoint/resume

Faithful SAC resume requires more than PPO:
- Model params (actor + twin critics + target networks)
- Optimizer state (Adam moments for all networks)
- Replay buffer contents (100K transitions × ~18 floats each = ~7MB)
- Entropy coefficient (if auto-tuned)
- VecNormalize running stats (already handled)

`checkpoint_training_state()` for SAC:
```python
def checkpoint_training_state(self, model) -> dict:
    return {
        "model_path": ...,
        "replay_buffer_path": ...,   # Serialized separately (large)
        "ent_coef": model.ent_coef,
    }
```

#### 3. Callback timing (no rollout boundaries)

PPO has natural rollout boundaries (`_on_rollout_start`, `_on_rollout_end`).
SAC collects steps continuously — no rollouts.

**Solution**: Backend provides a `flush_interval` property:
```python
@property
def flush_interval(self) -> int:
    """Steps between periodic flushes (holdout eval, pool refresh).
    PPO: n_steps * n_envs (natural rollout boundary)
    SAC: configurable (e.g., 2048)
    """
```

CurriculumCallback uses this instead of hooking rollout events. The existing `_on_rollout_start` logic (flush holdouts, apply pool refresh) moves to a step-count check in `_on_step`.

#### 4. Model mutation with JAX/Flax

```python
def clone_and_mutate(self, model, envs, mutation_rate):
    import jax
    params = model.policy.params  # Flax pytree
    key = jax.random.PRNGKey(...)
    def add_noise(param, key):
        noise = jax.random.normal(key, param.shape) * mutation_rate * 0.1
        return param + param * noise
    mutated = jax.tree.map(add_noise, params, split_keys)
    new_model = SAC("MlpPolicy", envs, ...)
    new_model.policy.params = mutated
    return new_model
```

### SBXSACBackend implementation

**New file**: `src/atom/training/backends/sbx_sac.py`

```python
from sbx import SAC

class SBXSACBackend:
    def __init__(self):
        self.training_config = {
            "learning_rate": 3e-4,
            "buffer_size": 100_000,
            "learning_starts": 1000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "ent_coef": "auto",
        }

    @property
    def capabilities(self):
        return BackendCapabilities(
            name="sbx_sac", framework="jax", on_policy=False
        )

    @property
    def flush_interval(self):
        return 2048

    def create_model(self, envs, seed):
        return SAC("MlpPolicy", envs, seed=seed,
                    policy_kwargs=self.get_policy_arch(),
                    **self.training_config)

    def predict(self, model, obs, deterministic=False):
        action, _ = model.predict(obs, deterministic=deterministic)
        return action

    def replace_env(self, model, envs):
        model.set_env(envs)
        # SBX may handle _last_obs differently — verify

    def handle_distribution_shift(self, model, kind):
        if kind == "level_transition":
            model.replay_buffer.reset()
```

### Dependencies
- `sbx-rl` package
- JAX already installed
- Flax (transitive via SBX)

### Testing strategy
1. Unit: SBXSACBackend satisfies protocol
2. Integration: Curriculum L1 trains and graduates with SBX SAC
3. A/B: Same curriculum, `--backend sb3_ppo` vs `--backend sbx_sac`, compare wall-clock time, Expert WR, training stability, anchor retention

### Export
- Keep Python-based export (`export_fighters.py`)
- Detect backend from model format
- Both produce identical `decide()` function interface
- ONNX export deferred

**Estimated effort**: 3-5 days after Phase 1A+1B.

---

## Sequencing

```
Phase 1A: Backend protocol + SB3PPO wrapper     (2-3 days)
    ↓ all tests pass, PPO behavior unchanged
Phase 1B: Population backend seam                (2 days)
    ↓ all tests pass, population behavior unchanged
    ↓ evaluate Run 8 results — decide if SAC needed
Phase 2:  SBX SAC backend                        (3-5 days)
    ↓ A/B test PPO vs SAC on same curriculum
Phase 3:  Future backends (TQC, model-based)     (as needed)
```

Total: ~2 weeks from start to SAC A/B comparison.
Phase 1 is zero-risk refactoring. Phase 2 is the algorithm change.

---

## Design Principles

1. **Each backend owns its training hyperparams.** Only network architecture is shared via `get_policy_arch()`. PPO's LR/batch/n_steps and SAC's buffer_size/tau/ent_coef are backend-internal.

2. **One backend per run.** All fighters in a population use the same backend. Curriculum graduate and population must use the same backend. The `--backend` flag applies to both phases.

3. **VecNormalize is framework-agnostic.** Running stats are numpy. Works with both SB3 and SBX.

4. **Export is backend-aware but interface-identical.** Both backends produce fighters with `decide(snapshot) -> action_dict`. Internal prediction differs (PyTorch vs JAX) but the external contract is the same.

---

## Revision Notes (final review feedback)

### 1. Keep PPO rollout-boundary callbacks, don't flatten to step-count flushes

The `flush_interval` approach was too aggressive. PPO's `_on_rollout_start()` provides safe env-replacement boundaries that we fought hard to get right (the stale-env bug). For PPO, keep rollout hooks as-is. For SAC, the backend provides a periodic step-based flush mechanism. The callback adapter handles the difference — not by replacing rollout hooks with a generic interval, but by the backend declaring its `flush_mode`:
- PPO: `flush_mode = "rollout_boundary"` → callback uses `_on_rollout_start()`
- SAC: `flush_mode = "step_interval"` → callback uses step-count check in `_on_step()`

### 2. Backend state vs trainer state — clear ownership

Backend owns (via `checkpoint_training_state` / `restore_training_state`):
- Model params + optimizer state
- Replay buffer + entropy coef (SAC only)

Trainer/recovery manager owns (NOT in backend):
- Curriculum progress (level, episodes, mastery)
- Callback state (episode rewards, wins)
- VecNormalize running stats (framework-agnostic numpy)

VecNormalize stays at the trainer layer. It's already handled there and is framework-agnostic.

### 3. Device lives in backend constructor

```python
backend = SB3PPOBackend(device="auto")  # resolves to "cpu" for PPO
backend = SBXSACBackend()                # JAX handles device automatically
```

`create_model()` and `load_model()` don't take device — the backend owns that decision at construction time. CLI `--device` flag maps to backend construction.

### 4. Phase 1B must cover population base-model loading

Population init loads a base model from curriculum graduate in `population_trainer.py` (~line 1021). This is a `PPO.load()` call that must go through `backend.load_model()`. Explicitly included in Phase 1B scope alongside mutation/cloning.
