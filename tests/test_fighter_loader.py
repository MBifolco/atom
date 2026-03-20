"""
Tests for fighter loading utility.
"""

import pytest
import tempfile
from pathlib import Path
from src.training.trainers.population.fighter_loader import (
    load_fighter,
    validate_fighter,
    FighterLoadError,
    load_hardcoded_fighters
)


class TestLoadFighter:
    """Test fighter loading functionality."""

    def test_load_existing_fighter(self):
        """Test loading an actual fighter file."""
        # Use one of our real fighters
        fighter_path = "fighters/examples/boxer.py"

        if not Path(fighter_path).exists():
            pytest.skip("Boxer fighter not found")

        decide_func = load_fighter(fighter_path, verbose=False)

        assert callable(decide_func)

    def test_load_nonexistent_file_raises_error(self):
        """Test loading non-existent file raises FighterLoadError."""
        with pytest.raises(FighterLoadError) as exc_info:
            load_fighter("/tmp/nonexistent_fighter_xyz.py")

        assert "not found" in str(exc_info.value).lower()

    def test_load_file_missing_decide_raises_error(self):
        """Test file without decide function raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write a Python file without decide function
            f.write("# Fighter without decide function\n")
            f.write("def other_function():\n")
            f.write("    pass\n")
            temp_path = f.name

        try:
            with pytest.raises(FighterLoadError) as exc_info:
                load_fighter(temp_path)

            assert "missing 'decide'" in str(exc_info.value).lower()
        finally:
            Path(temp_path).unlink()

    def test_load_with_verbose(self):
        """Test loading with verbose mode."""
        # Create a valid fighter
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def decide(state):\n")
            f.write("    return {'acceleration': 0.0, 'stance': 'neutral'}\n")
            temp_path = f.name

        try:
            # Load with verbose (should not crash)
            decide_func = load_fighter(temp_path, verbose=True)
            assert callable(decide_func)
        finally:
            Path(temp_path).unlink()


class TestValidateFighter:
    """Test fighter validation."""

    def test_validate_working_fighter(self):
        """Test validating a working fighter."""
        def good_fighter(state):
            return {"acceleration": 0.5, "stance": "extended"}

        is_valid = validate_fighter(good_fighter, verbose=False)

        assert is_valid

    def test_validate_fighter_with_missing_acceleration(self):
        """Test fighter missing acceleration field fails validation."""
        def bad_fighter(state):
            return {"stance": "neutral"}  # Missing acceleration!

        is_valid = validate_fighter(bad_fighter, verbose=False)

        assert not is_valid

    def test_validate_fighter_with_missing_stance(self):
        """Test fighter missing stance field fails validation."""
        def bad_fighter(state):
            return {"acceleration": 0.5}  # Missing stance!

        is_valid = validate_fighter(bad_fighter, verbose=False)

        assert not is_valid

    def test_validate_fighter_that_crashes(self):
        """Test fighter that crashes fails validation."""
        def crashing_fighter(state):
            raise ValueError("Test crash")

        is_valid = validate_fighter(crashing_fighter, verbose=False)

        assert not is_valid

    def test_validate_with_verbose(self):
        """Test validation with verbose mode."""
        def good_fighter(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        # Should not crash with verbose
        is_valid = validate_fighter(good_fighter, verbose=True)
        assert is_valid


class TestLoadHardcodedFighters:
    """Test loading hardcoded fighters from directory."""

    def test_load_hardcoded_fighters_from_examples(self):
        """Test loading fighters from examples directory."""
        fighters_dict = load_hardcoded_fighters(base_path="fighters/examples", verbose=False)

        # Should return dict (may be empty if path issues)
        assert isinstance(fighters_dict, dict)

        # If any loaded, all should be callable
        for name, decide_func in fighters_dict.items():
            assert callable(decide_func)

    def test_load_hardcoded_fighters_with_verbose(self):
        """Test loading with verbose mode."""
        # Should not crash with verbose
        fighters = load_hardcoded_fighters(base_path="fighters/examples", verbose=True)

        assert isinstance(fighters, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
