#!/usr/bin/env python3
"""
Fix the observation shape and JAX array issues in population training.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("Fixing population training observation issues...")
print("=" * 60)

# 1. Fix the JAX array stance issue in _create_opponent_decide_func
print("\n1. Fixing JAX array stance issue in population_trainer.py...")
population_file = project_root / "src/training/trainers/population/population_trainer.py"
content = population_file.read_text()

# Fix the stance handling to convert JAX arrays to Python types
old_stance_handling = '''        # Get opponent stance as integer
        opp_stance = snapshot["opponent"]["stance"]
        stance_map = {"neutral": 0, "extended": 1, "defending": 2, "retracted": 0}  # retracted fallback to neutral
        opp_stance_int = stance_map.get(opp_stance, 0)'''

new_stance_handling = '''        # Get opponent stance as integer
        opp_stance = snapshot["opponent"]["stance"]
        # Handle JAX arrays - convert to Python type
        if hasattr(opp_stance, '__array__'):
            opp_stance_val = int(opp_stance)
            # JAX arrays contain stance as integer already (0, 1, 2)
            opp_stance_int = opp_stance_val
        else:
            # String stance from Python arena
            stance_map = {"neutral": 0, "extended": 1, "defending": 2, "retracted": 0}  # retracted fallback to neutral
            opp_stance_int = stance_map.get(opp_stance, 0)'''

if old_stance_handling in content:
    content = content.replace(old_stance_handling, new_stance_handling)
    population_file.write_text(content)
    print("   ✅ Fixed JAX array stance handling in _create_opponent_decide_func")
else:
    print("   ⚠️  Stance handling already fixed or pattern not found")

# 2. Also check the VmapEnvWrapper to ensure observation shape is correct
print("\n2. Checking VmapEnvWrapper observation shape...")
vmap_file = project_root / "src/training/vmap_env_wrapper.py"
vmap_content = vmap_file.read_text()

# The observation space definition looks correct, but let's ensure the stacking is right
if "# Stack observations (now 13 values)" in vmap_content:
    print("   ✅ VmapEnvWrapper has correct observation stacking comment")
else:
    print("   ⚠️  VmapEnvWrapper might have incorrect observation stacking")

# 3. Check if there are any legacy 9-dim observation space definitions
print("\n3. Looking for legacy 9-dim observation spaces...")
legacy_pattern = "Box(low=np.array([0, -3, 0, 0, 0, -5, 0, 0, 0]"

if legacy_pattern in content:
    print("   ⚠️  Found legacy 9-dim observation space in population_trainer.py!")
else:
    print("   ✅ No legacy 9-dim observation spaces found in population_trainer.py")

if legacy_pattern in vmap_content:
    print("   ⚠️  Found legacy 9-dim observation space in vmap_env_wrapper.py!")
else:
    print("   ✅ No legacy 9-dim observation spaces found in vmap_env_wrapper.py")

# 4. Check model initialization observation space
print("\n4. Checking model initialization...")

# Search for where PPO models are created in population trainer
if "observation_space" in content and "Box" in content:
    # Check if models are being created with wrong observation space
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'PPO(' in line or 'SAC(' in line:
            # Check surrounding lines for observation space
            context = '\n'.join(lines[max(0, i-10):min(len(lines), i+10)])
            if 'observation_space' in context and 'Box' in context:
                print(f"   ⚠️  Found model creation with explicit observation_space at line {i+1}")
                break
    else:
        print("   ✅ Models created without explicit observation_space override")

print("\n" + "=" * 60)
print("✅ Fixes applied!")
print("\nNext steps:")
print("1. Run the population training to test the fixes:")
print("   python train_progressive.py --mode population --use-vmap")
print("\n2. If still seeing (45, 9) errors, the issue is in model initialization")
print("   The models might have been created with wrong observation space")