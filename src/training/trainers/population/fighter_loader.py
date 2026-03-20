"""
Utility module for loading and validating hardcoded fighters.
Provides robust loading with proper error handling.
"""

import sys
import importlib.util
from pathlib import Path
from typing import Callable, Optional, Dict, List
import traceback


class FighterLoadError(Exception):
    """Custom exception for fighter loading errors."""
    pass


def load_fighter(fighter_path: str, verbose: bool = False) -> Callable:
    """
    Load a hardcoded fighter from a Python file.

    Args:
        fighter_path: Path to the fighter Python file
        verbose: If True, print debug information

    Returns:
        The decide function from the fighter module

    Raises:
        FighterLoadError: If the fighter cannot be loaded
    """
    path = Path(fighter_path)

    # Check if file exists
    if not path.exists():
        raise FighterLoadError(f"Fighter file not found: {fighter_path}")

    if verbose:
        print(f"Loading fighter from: {path.absolute()}")

    try:
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("fighter", str(path))
        if spec is None:
            raise FighterLoadError(f"Could not load spec from {fighter_path}")

        module = importlib.util.module_from_spec(spec)
        if module is None:
            raise FighterLoadError(f"Could not create module from spec")

        # Execute the module
        spec.loader.exec_module(module)

        # Check for decide function
        if not hasattr(module, 'decide'):
            raise FighterLoadError(f"Fighter {fighter_path} missing 'decide' function")

        decide_func = getattr(module, 'decide')

        # Validate it's callable
        if not callable(decide_func):
            raise FighterLoadError(f"'decide' in {fighter_path} is not callable")

        if verbose:
            print(f"  ✓ Successfully loaded {path.name}")

        return decide_func

    except Exception as e:
        if isinstance(e, FighterLoadError):
            raise
        else:
            # Provide detailed error information
            error_msg = f"Failed to load fighter {fighter_path}: {str(e)}"
            if verbose:
                error_msg += f"\n{traceback.format_exc()}"
            raise FighterLoadError(error_msg)


def validate_fighter(decide_func: Callable, verbose: bool = False) -> bool:
    """
    Validate that a fighter function works with a test snapshot.

    Args:
        decide_func: The fighter's decide function
        verbose: If True, print debug information

    Returns:
        True if the fighter is valid, False otherwise
    """
    # Create a test snapshot
    test_snapshot = {
        "tick": 0,
        "you": {
            "position": 2.0,
            "velocity": 0.0,
            "hp": 100.0,
            "stamina": 8.0,
            "max_hp": 100.0,
            "max_stamina": 8.0,
            "stance": "neutral"
        },
        "opponent": {
            "distance": 8.48,
            "velocity": 0.0,
            "hp": 100.0,
            "stamina": 8.0,
            "max_hp": 100.0,
            "max_stamina": 8.0
        },
        "arena": {
            "width": 12.48
        }
    }

    try:
        # Call the decide function
        decision = decide_func(test_snapshot)

        # Validate the response
        if not isinstance(decision, dict):
            if verbose:
                print(f"  ✗ Fighter returned {type(decision)} instead of dict")
            return False

        # Check required fields
        if "acceleration" not in decision:
            if verbose:
                print(f"  ✗ Fighter decision missing 'acceleration' field")
            return False

        if "stance" not in decision:
            if verbose:
                print(f"  ✗ Fighter decision missing 'stance' field")
            return False

        # Validate acceleration is numeric
        accel = decision["acceleration"]
        if not isinstance(accel, (int, float)):
            if verbose:
                print(f"  ✗ Acceleration is {type(accel)}, not numeric")
            return False

        # Validate stance is string
        stance = decision["stance"]
        valid_stances = ["neutral", "extended", "retracted", "defending"]
        if stance not in valid_stances:
            if verbose:
                print(f"  ✗ Invalid stance: {stance}")
            return False

        if verbose:
            print(f"  ✓ Fighter validation passed")

        return True

    except Exception as e:
        if verbose:
            print(f"  ✗ Fighter validation failed: {str(e)}")
        return False


def load_hardcoded_fighters(base_path: Optional[str] = None, verbose: bool = True) -> Dict[str, Callable]:
    """
    Load all standard hardcoded fighters.

    Args:
        base_path: Base directory for the atom project (defaults to current working directory)
        verbose: If True, print loading information

    Returns:
        Dictionary mapping fighter names to their decide functions
    """
    if base_path is None:
        # Try to find the atom directory
        base_path = Path.cwd()
        while base_path.name != "atom" and base_path.parent != base_path:
            base_path = base_path.parent
        if base_path.name != "atom":
            base_path = Path.cwd()

    fighters_dir = Path(base_path) / "fighters" / "examples"

    if verbose:
        print(f"\nLoading hardcoded fighters from: {fighters_dir}")

    example_paths = sorted(
        path for path in fighters_dir.glob("*.py")
        if not path.name.startswith("__")
    )
    if verbose and not example_paths:
        print(f"  ✗ No example fighters found in {fighters_dir}")

    loaded_fighters = {}
    failed_fighters = []

    for filepath in example_paths:
        name = filepath.stem

        try:
            if verbose:
                print(f"\n{name.upper()}:")

            decide_func = load_fighter(str(filepath), verbose=verbose)

            # Validate the fighter
            if validate_fighter(decide_func, verbose=verbose):
                loaded_fighters[name] = decide_func
            else:
                failed_fighters.append(name)
                if verbose:
                    print(f"  ✗ {name} failed validation")

        except FighterLoadError as e:
            failed_fighters.append(name)
            if verbose:
                print(f"  ✗ Failed to load {name}: {e}")
        except Exception as e:
            failed_fighters.append(name)
            if verbose:
                print(f"  ✗ Unexpected error loading {name}: {e}")

    if verbose:
        print(f"\n" + "="*50)
        print(f"Successfully loaded: {list(loaded_fighters.keys())}")
        if failed_fighters:
            print(f"Failed to load: {failed_fighters}")
        print("="*50 + "\n")

    return loaded_fighters


def test_fighter_in_combat(decide_func: Callable, num_steps: int = 10, verbose: bool = True) -> bool:
    """
    Test a fighter through several combat steps.

    Args:
        decide_func: The fighter's decide function
        num_steps: Number of steps to simulate
        verbose: If True, print step-by-step information

    Returns:
        True if the fighter runs without errors
    """
    if verbose:
        print(f"Testing fighter for {num_steps} steps...")

    # Simulate a simple combat scenario
    position = 2.0
    opp_position = 10.0
    hp = 100.0
    opp_hp = 100.0

    for step in range(num_steps):
        distance = abs(opp_position - position)

        snapshot = {
            "tick": step,
            "you": {
                "position": position,
                "velocity": 0.5 if step % 2 == 0 else -0.3,
                "hp": hp,
                "stamina": 8.0 - (step * 0.5),
                "max_hp": 100.0,
                "max_stamina": 8.0,
                "stance": "neutral"
            },
            "opponent": {
                "distance": distance,
                "velocity": -0.2,
                "hp": opp_hp,
                "stamina": 7.0,
                "max_hp": 100.0,
                "max_stamina": 8.0
            },
            "arena": {
                "width": 12.48
            }
        }

        try:
            decision = decide_func(snapshot)

            if verbose and step < 3:  # Only print first few steps
                print(f"  Step {step}: dist={distance:.2f}, "
                      f"accel={decision['acceleration']:.2f}, "
                      f"stance={decision['stance']}")

            # Update positions for next step (simplified)
            position += decision["acceleration"] * 0.1
            position = max(0, min(12.48, position))
            opp_position -= 0.2  # Opponent moves closer

            # Simulate some damage
            if distance < 2:
                hp -= 2
                opp_hp -= 3

        except Exception as e:
            if verbose:
                print(f"  ✗ Error at step {step}: {e}")
            return False

    if verbose:
        print(f"  ✓ Fighter ran successfully for {num_steps} steps")

    return True


if __name__ == "__main__":
    # Test the loader
    print("Testing fighter loader utility...")

    # Try to load all hardcoded fighters
    fighters = load_hardcoded_fighters("/home/biff/eng/atom", verbose=True)

    # Test each loaded fighter
    for name, decide_func in fighters.items():
        print(f"\nTesting {name} in combat simulation:")
        test_fighter_in_combat(decide_func, num_steps=5, verbose=True)
