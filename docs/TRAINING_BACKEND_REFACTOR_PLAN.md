# Training Backend Abstraction + SBX SAC Migration

## Goal

Make the training algorithm a pluggable component so we can:
1. Keep PPO (SB3) working as-is
2. Add SAC via SBX (all-JAX, ~10x faster)
3. Support future algorithms (TQC, model-based) without touching orchestration code

## Current State

The `algorithm` parameter exists but the abstraction is leaky:
- `from stable_baselines3 import PPO, SAC` hardcoded in 10+ files
- PPO-specific assumptions in callback timing (rollout start/end), NaN recovery
- `model.predict()` / `model.learn()` called directly with PPO-specific semantics
- Checkpoint save/load assumes SB3 format
- `get_stable_ppo_config()` only returns PPO config
- Device management assumes PyTorch ("cpu"/"cuda")

## Phase 1: Extract TrainingBackend Protocol (refactor, no new functionality)

### 1a. Define the protocol

**New file**: `src/atom/training/backends/__init__.py`

```python
from .protocol import TrainingBackend, BackendCapabilities
from .sb3_ppo import SB3PPOBackend
```

**New file**: `src/atom/training/backends/protocol.py`

```python
from typing import Protocol, Any, Callable
import numpy as np

class BackendCapabilities:
    """What a backend supports."""
    name: str                    # "sb3_ppo", "sbx_sac", etc.
    framework: str               # "pytorch", "jax"
    on_policy: bool              # True=PPO (rollouts), False=SAC (replay buffer)
    supports_deterministic: bool # Whether deterministic=True is meaningful

class TrainingBackend(Protocol):
    """Interface for pluggable RL training algorithms."""

    @property
    def capabilities(self) -> BackendCapabilities: ...

    def create_model(self, envs: Any, seed: int, **kwargs) -> Any:
        """Create a new model for the given environment."""
        ...

    def load_model(self, path: str, envs: Any = None, device: str = "auto") -> Any:
        """Load a model from disk."""
        ...

    def save_model(self, model: Any, path: str) -> None:
        """Save a model to disk."""
        ...

    def predict(self, model: Any, obs: np.ndarray) -> np.ndarray:
        """Get action from observation. Always stochastic."""
        ...

    def learn(self, model: Any, total_timesteps: int, callback: Any = None,
              reset_num_timesteps: bool = True, progress_bar: bool = False) -> Any:
        """Train the model."""
        ...

    def get_policy_kwargs(self) -> dict:
        """Return the shared network architecture config."""
        ...

    def set_env(self, model: Any, envs: Any) -> None:
        """Update the model's environment (for level transitions)."""
        ...

    def reset_after_set_env(self, model: Any, envs: Any) -> None:
        """Reset _last_obs after set_env to prevent NoneType crash."""
        ...
```

### 1b. Implement SB3PPOBackend

**New file**: `src/atom/training/backends/sb3_ppo.py`

Wraps existing PPO logic from `stable_ppo_config.py` and `curriculum_components.py`:
- `create_model()` → current `ModelFactory.create_model()` for PPO
- `load_model()` → `PPO.load(path, device=device)`
- `save_model()` → `model.save(path)`
- `predict()` → `model.predict(obs, deterministic=False)`
- `learn()` → `model.learn(total_timesteps, callback, ...)`
- `get_policy_kwargs()` → `get_shared_policy_kwargs()`
- `set_env()` → `model.set_env(envs)` + reset _last_obs pattern
- `reset_after_set_env()` → the `model._last_obs = envs.reset()` pattern

### 1c. Update consumers to use the protocol

**`curriculum_components.py`** — `ModelFactory`:
- Accept a `TrainingBackend` instead of building PPO/SAC inline
- `create_model()` delegates to `backend.create_model()`

**`curriculum_trainer.py`**:
- Accept `backend: TrainingBackend` in constructor
- Replace all `self.model = PPO.load(...)` with `backend.load_model()`
- Replace all `self.model.save(...)` with `backend.save_model()`
- Replace all `self.model.predict(...)` with `backend.predict()`
- Replace all `self.model.set_env(...)` with `backend.set_env()` + `backend.reset_after_set_env()`
- Replace `PPO if self.algorithm == "ppo" else SAC` with `backend.load_model()`

**`population_trainer.py`**:
- Accept `backend: TrainingBackend` in constructor
- Replace `PPO("MlpPolicy", env, ...)` with `backend.create_model(env, seed)`
- Replace `fighter.model.predict(...)` with `backend.predict(fighter.model, obs)`

**`population_evolution.py`**:
- `_clone_and_mutate_model()` needs to work with both PyTorch and JAX params
- For Phase 1, keep the PyTorch mutation logic and add a backend-aware path

**NaN recovery** (`curriculum_components.py` `LevelRunner`):
- Currently calls `PPO.load()` directly
- Delegate to `backend.load_model()`
- Learning rate reduction: `model.learning_rate *= 0.5` → backend method

### 1d. Update CLI / entry point

**`train_progressive.py`**:
- Add `--backend` flag: `sb3_ppo` (default), `sbx_sac` (new)
- Construct the appropriate backend and pass to trainer

### 1e. Tests

- All existing tests pass (SB3PPOBackend should be drop-in)
- New test: `test_backend_protocol.py` — verify SB3PPOBackend satisfies protocol
- Run a quick smoke test to verify curriculum still works

### Files to modify in Phase 1

```
NEW:
  src/atom/training/backends/__init__.py
  src/atom/training/backends/protocol.py
  src/atom/training/backends/sb3_ppo.py
  tests/test_backend_protocol.py

MODIFY:
  src/atom/training/trainers/curriculum_components.py  — ModelFactory, LevelRunner
  src/atom/training/trainers/curriculum_trainer.py     — use backend throughout
  src/atom/training/trainers/population/population_trainer.py — use backend
  src/atom/training/trainers/population/population_evolution.py — model cloning
  apps/training/train_progressive.py                    — --backend flag
  src/atom/training/utils/stable_ppo_config.py          — move into sb3_ppo backend
```

**Estimated effort**: 2-3 days. No behavioral changes. Pure refactor.

---

## Phase 2: Add SBX SAC Backend (new functionality)

### 2a. Install SBX

Add `sbx-rl` to `requirements-colab.txt` and `requirements.txt`.

SBX API: https://github.com/araffin/sbx
- `from sbx import SAC, TQC, DroQ`
- Same `.learn()`, `.predict()`, `.save()`, `.load()` interface as SB3
- Uses Flax (JAX neural networks) instead of PyTorch
- VecEnv interface is identical to SB3

### 2b. Implement SBXSACBackend

**New file**: `src/atom/training/backends/sbx_sac.py`

```python
from sbx import SAC

class SBXSACBackend:
    def create_model(self, envs, seed, **kwargs):
        return SAC(
            "MlpPolicy",
            envs,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",
            policy_kwargs=self.get_policy_kwargs(),
            seed=seed,
        )

    def get_policy_kwargs(self):
        # SBX uses same net_arch format as SB3
        return {"net_arch": [256, 256]}

    def predict(self, model, obs):
        action, _ = model.predict(obs, deterministic=False)
        return action

    # ... etc, same interface as SB3PPOBackend
```

### 2c. Key differences to handle

**On-policy vs off-policy training loop:**
- PPO: `model.learn(512000)` collects rollouts of `n_steps * n_envs`, trains, repeats
- SAC: `model.learn(512000)` collects single steps into replay buffer, trains continuously

The curriculum callback (`CurriculumCallback`) hooks into PPO's rollout lifecycle:
- `_on_rollout_start()` — flushes holdouts, applies pool refresh
- `_on_rollout_end()` — logs timing

SAC doesn't have rollout boundaries. Instead:
- `_on_step()` fires every step (same as PPO)
- Need to add periodic flush/refresh logic (every N steps instead of every rollout)

**Solution**: Add a `steps_between_flushes` parameter. For PPO it's `n_steps * n_envs` (natural rollout boundary). For SAC it's configurable (e.g., every 2048 steps).

**Device management:**
- SB3 PPO: `device="cpu"` or `device="cuda"`
- SBX SAC: JAX handles device automatically via `jax.devices()`
- Backend abstracts this: `create_model()` handles device internally

**Model mutation for population evolution:**
- PPO (PyTorch): `model.policy.state_dict()` → mutate tensors → `load_state_dict()`
- SBX SAC (JAX/Flax): `model.policy.params` → mutate JAX arrays → replace params
- Need a backend method: `mutate_model(model, mutation_rate) -> model`

**Checkpoint format:**
- SB3: PyTorch `.zip` files
- SBX: JAX serialization (different format)
- Backend abstracts: `save_model()` / `load_model()` handle format internally
- Population models dir will have different file formats per backend
- Curriculum graduate must be loadable by population (same backend required)

### 2d. VecNormalize compatibility

SBX uses the same `VecNormalize` from SB3 (it's numpy-based, framework-agnostic). No changes needed.

### 2e. ONNX export

Current export uses `torch.onnx.export()`. SBX models can't export to ONNX directly.

Options:
1. Export as standalone JAX function (jax2tf → SavedModel → ONNX)
2. Export as Python file with embedded weights (current `export_fighters.py` approach)
3. Skip ONNX, keep Python-based export only

Recommendation: Keep Python-based export (option 2). Update `export_fighters.py` to detect backend and use appropriate predict call.

### 2f. Tests

- `test_backend_protocol.py` — verify SBXSACBackend satisfies protocol
- Integration test: curriculum L1 trains and graduates with SBX SAC
- Smoke test: population creates fighters and runs evaluation

### Files for Phase 2

```
NEW:
  src/atom/training/backends/sbx_sac.py

MODIFY:
  requirements.txt                    — add sbx-rl
  requirements-colab.txt              — add sbx-rl
  scripts/training/export_fighters.py — backend-aware export
  apps/training/train_progressive.py  — wire sbx_sac option
```

**Estimated effort**: 3-4 days after Phase 1.

---

## Phase 3 (future): Additional Backends

Once the protocol exists, adding new backends is straightforward:

### SBX TQC (Truncated Quantile Critics)
- Better Q-value estimation than SAC
- Same SBX API, just `from sbx import TQC`
- ~1 day to implement backend

### SBX DroQ (Dropout Q-Functions)
- SAC variant with dropout regularization
- Faster training with similar performance
- ~1 day to implement backend

### Model-Based (Dreamer / custom)
- Would implement the protocol with a learned world model
- `learn()` trains both the world model and the policy
- `predict()` could optionally do planning (look-ahead)
- Significantly more work (~2-4 weeks)

---

## Migration Strategy

1. **Phase 1 first** (refactor) — no risk, all tests pass, PPO still works
2. **Run 8 completes** — evaluate 256-256 + sparse rewards + anchors results
3. **Phase 2** (SBX SAC) — implement alongside PPO, A/B test on same curriculum
4. **Compare** — run identical curriculum with `--backend sb3_ppo` vs `--backend sbx_sac`
5. **If SAC wins** — make it the default, keep PPO as fallback

This way we never lose PPO capability, and every training run can specify which backend to use.

## Open Questions

1. **Population mixing**: Can we evolve a population where some fighters use PPO and others use SAC? Probably not worth the complexity — all fighters in a population should use the same backend.

2. **Curriculum → Population handoff**: The curriculum graduate model must be loadable by the population trainer. Both must use the same backend for a single run.

3. **Hyperparameter sharing**: `get_shared_policy_kwargs()` works for both SB3 and SBX (same `net_arch` format). But SAC has additional params (buffer_size, tau, etc.) that PPO doesn't. Each backend owns its full config.

4. **VecNormalize state**: The reward normalization running stats are numpy and framework-agnostic. A curriculum graduate trained with PPO could theoretically be fine-tuned with SAC if the VecNormalize stats are preserved. Worth testing but not required.
