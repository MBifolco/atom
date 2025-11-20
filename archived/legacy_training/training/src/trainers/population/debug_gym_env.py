#!/usr/bin/env python3
"""
Debug script to test if gym environment works properly.
"""

import sys
from pathlib import Path

# Add parent directories to path
atom_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(atom_root))

print(f"Python path includes: {atom_root}")

try:
    # Test import
    print("\n1. Testing import of gym_env...")
    from training.src.gym_env import AtomCombatEnv
    print("   ✓ Import successful")

    # Test creating environment
    print("\n2. Testing environment creation...")
    env = AtomCombatEnv(
        opponent_decision_func=lambda s: {"acceleration": 0, "stance": "neutral"}
    )
    print("   ✓ Environment created")

    # Test reset
    print("\n3. Testing environment reset...")
    obs, info = env.reset()
    print(f"   ✓ Reset successful")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation: {obs}")

    # Test step
    print("\n4. Testing environment step...")
    import numpy as np
    action = np.array([0.0, 0.0], dtype=np.float32)  # No acceleration, neutral stance
    obs, reward, done, truncated, info = env.step(action)
    print(f"   ✓ Step successful")
    print(f"   Reward: {reward}")
    print(f"   Done: {done}")
    print(f"   Info: {info}")

    # Test with SubprocVecEnv
    print("\n5. Testing with SubprocVecEnv...")
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def make_env():
        def _init():
            return AtomCombatEnv(
                opponent_decision_func=lambda s: {"acceleration": 0, "stance": "neutral"}
            )
        return _init

    vec_env = SubprocVecEnv([make_env() for _ in range(2)])
    print("   ✓ SubprocVecEnv created with 2 environments")

    obs = vec_env.reset()
    print(f"   ✓ Vector reset successful")
    print(f"   Observation shape: {obs.shape}")

    vec_env.close()
    print("   ✓ Vector environment closed")

    print("\n✅ All tests passed!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()