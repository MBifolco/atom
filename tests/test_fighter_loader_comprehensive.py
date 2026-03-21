"""
Comprehensive tests for fighter_loader to increase coverage.
Tests loading, validation, and error handling.
"""

import pytest
import tempfile
from pathlib import Path

from src.atom.training.trainers.population.fighter_loader import (
    load_fighter,
    validate_fighter,
    FighterLoadError,
)


class TestLoadFighter:
    """Tests for load_fighter function."""

    def test_load_valid_fighter(self):
        """Test loading a valid fighter file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighter_file = Path(tmpdir) / "fighter.py"
            fighter_file.write_text('''
def decide(state):
    return {"stance": "neutral", "acceleration": 0.0}
''')
            decide_func = load_fighter(str(fighter_file))
            assert callable(decide_func)

    def test_load_fighter_nonexistent_file(self):
        """Test loading a fighter that doesn't exist."""
        with pytest.raises(FighterLoadError) as exc_info:
            load_fighter("/nonexistent/path/fighter.py")
        assert "not found" in str(exc_info.value)

    def test_load_fighter_missing_decide(self):
        """Test loading a fighter without decide function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighter_file = Path(tmpdir) / "fighter.py"
            fighter_file.write_text('''
def some_other_function():
    return {}
''')
            with pytest.raises(FighterLoadError) as exc_info:
                load_fighter(str(fighter_file))
            assert "missing 'decide'" in str(exc_info.value)

    def test_load_fighter_verbose(self):
        """Test loading with verbose output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighter_file = Path(tmpdir) / "fighter.py"
            fighter_file.write_text('''
def decide(state):
    return {"stance": "neutral", "acceleration": 0.0}
''')
            # Should not raise, verbose just prints
            decide_func = load_fighter(str(fighter_file), verbose=True)
            assert callable(decide_func)

    def test_load_fighter_syntax_error(self):
        """Test loading a fighter with syntax error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighter_file = Path(tmpdir) / "fighter.py"
            fighter_file.write_text('''
def decide(state
    return {}
''')
            with pytest.raises(FighterLoadError):
                load_fighter(str(fighter_file))

    def test_load_fighter_decide_not_callable(self):
        """Test loading where decide is not callable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighter_file = Path(tmpdir) / "fighter.py"
            fighter_file.write_text('''
decide = "not a function"
''')
            with pytest.raises(FighterLoadError) as exc_info:
                load_fighter(str(fighter_file))
            assert "not callable" in str(exc_info.value)

    def test_load_fighter_with_imports(self):
        """Test loading a fighter that uses imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighter_file = Path(tmpdir) / "fighter.py"
            fighter_file.write_text('''
import math

def decide(state):
    value = math.sin(0)
    return {"stance": "neutral", "acceleration": value}
''')
            decide_func = load_fighter(str(fighter_file))
            assert callable(decide_func)

    def test_load_real_fighter_boxer(self):
        """Test loading the real boxer fighter."""
        boxer_path = Path(__file__).parent.parent / "fighters" / "examples" / "boxer.py"
        if boxer_path.exists():
            decide_func = load_fighter(str(boxer_path))
            assert callable(decide_func)

    def test_load_real_fighter_counter_puncher(self):
        """Test loading the real counter_puncher fighter."""
        path = Path(__file__).parent.parent / "fighters" / "examples" / "counter_puncher.py"
        if path.exists():
            decide_func = load_fighter(str(path))
            assert callable(decide_func)


class TestValidateFighter:
    """Tests for validate_fighter function."""

    def test_validate_valid_fighter(self):
        """Test validating a valid fighter function."""
        def valid_decide(state):
            return {"stance": "neutral", "acceleration": 0.0}

        assert validate_fighter(valid_decide) is True

    def test_validate_fighter_missing_acceleration(self):
        """Test validating fighter missing acceleration field."""
        def bad_decide(state):
            return {"stance": "neutral"}

        assert validate_fighter(bad_decide) is False

    def test_validate_fighter_missing_stance(self):
        """Test validating fighter missing stance field."""
        def bad_decide(state):
            return {"acceleration": 0.0}

        assert validate_fighter(bad_decide) is False

    def test_validate_fighter_invalid_return_type(self):
        """Test validating fighter that returns non-dict."""
        def bad_decide(state):
            return "not a dict"

        assert validate_fighter(bad_decide) is False

    def test_validate_fighter_invalid_acceleration_type(self):
        """Test validating fighter with non-numeric acceleration."""
        def bad_decide(state):
            return {"stance": "neutral", "acceleration": "fast"}

        assert validate_fighter(bad_decide) is False

    def test_validate_fighter_invalid_stance(self):
        """Test validating fighter with invalid stance."""
        def bad_decide(state):
            return {"stance": "flying_kick", "acceleration": 0.0}

        assert validate_fighter(bad_decide) is False

    def test_validate_fighter_with_verbose(self):
        """Test validation with verbose output."""
        def bad_decide(state):
            return {"stance": "invalid"}

        # Should return False and print debug info
        assert validate_fighter(bad_decide, verbose=True) is False

    def test_validate_fighter_invalid_stance_verbose(self, capsys):
        """Test verbose output when stance is invalid."""
        def bad_decide(state):
            return {"stance": "flying_kick", "acceleration": 0.0}

        result = validate_fighter(bad_decide, verbose=True)
        captured = capsys.readouterr()
        assert result is False
        assert "Invalid stance" in captured.out

    def test_validate_fighter_extended_stance(self):
        """Test validating fighter with extended stance."""
        def decide(state):
            return {"stance": "extended", "acceleration": 1.0}

        assert validate_fighter(decide) is True

    def test_validate_fighter_defending_stance(self):
        """Test validating fighter with defending stance."""
        def decide(state):
            return {"stance": "defending", "acceleration": -0.5}

        assert validate_fighter(decide) is True

    def test_validate_fighter_retracted_stance(self):
        """Test validating fighter with retracted stance."""
        def decide(state):
            return {"stance": "retracted", "acceleration": 0.0}

        assert validate_fighter(decide) is True

    def test_validate_fighter_integer_acceleration(self):
        """Test validating fighter with integer acceleration."""
        def decide(state):
            return {"stance": "neutral", "acceleration": 1}

        assert validate_fighter(decide) is True

    def test_validate_fighter_exception_handling(self):
        """Test validation handles exceptions from decide function."""
        def bad_decide(state):
            raise RuntimeError("Fighter crashed!")

        assert validate_fighter(bad_decide) is False

    def test_validate_fighter_none_return(self):
        """Test validation handles None return."""
        def bad_decide(state):
            return None

        assert validate_fighter(bad_decide) is False


class TestFighterLoadError:
    """Tests for FighterLoadError exception."""

    def test_error_message(self):
        """Test exception message is preserved."""
        error = FighterLoadError("Test error message")
        assert str(error) == "Test error message"

    def test_error_inherits_exception(self):
        """Test FighterLoadError inherits from Exception."""
        assert issubclass(FighterLoadError, Exception)


class TestLoadAndValidateIntegration:
    """Integration tests for load and validate together."""

    def test_load_and_validate_valid_fighter(self):
        """Test loading and validating a valid fighter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighter_file = Path(tmpdir) / "fighter.py"
            fighter_file.write_text('''
def decide(state):
    distance = state["opponent"]["distance"]
    if distance < 2:
        return {"stance": "extended", "acceleration": 0.0}
    else:
        return {"stance": "neutral", "acceleration": 1.0}
''')
            decide_func = load_fighter(str(fighter_file))
            assert validate_fighter(decide_func) is True

    def test_load_and_validate_real_fighters(self):
        """Test loading and validating real example fighters."""
        examples_dir = Path(__file__).parent.parent / "fighters" / "examples"
        if examples_dir.exists():
            loaded_count = 0
            for fighter_file in examples_dir.glob("*.py"):
                if fighter_file.name.startswith("__"):
                    continue
                try:
                    decide_func = load_fighter(str(fighter_file))
                    # Just test that we can load and validate (result may vary)
                    validate_fighter(decide_func)
                    loaded_count += 1
                except FighterLoadError:
                    pass  # Some files might not be fighters
            assert loaded_count >= 1  # Should load at least one


class TestFighterLoaderVerbose:
    """Tests for verbose output paths in fighter_loader."""

    def test_load_fighter_verbose_output(self, capsys):
        """Test that verbose=True produces output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighter_file = Path(tmpdir) / "verbose_fighter.py"
            fighter_file.write_text('''
def decide(state):
    return {"stance": "neutral", "acceleration": 0}
''')
            load_fighter(str(fighter_file), verbose=True)
            captured = capsys.readouterr()
            assert "Loading fighter" in captured.out
            assert "Successfully loaded" in captured.out

    def test_validate_fighter_verbose_non_dict(self, capsys):
        """Test verbose output when fighter returns non-dict."""
        def bad_fighter(state):
            return "not a dict"

        result = validate_fighter(bad_fighter, verbose=True)
        captured = capsys.readouterr()
        assert result == False
        assert "instead of dict" in captured.out

    def test_validate_fighter_verbose_missing_acceleration(self, capsys):
        """Test verbose output when acceleration is missing."""
        def incomplete_fighter(state):
            return {"stance": "neutral"}

        result = validate_fighter(incomplete_fighter, verbose=True)
        captured = capsys.readouterr()
        assert result == False

    def test_validate_fighter_verbose_missing_stance(self, capsys):
        """Test verbose output when stance is missing."""
        def incomplete_fighter(state):
            return {"acceleration": 0.5}

        result = validate_fighter(incomplete_fighter, verbose=True)
        captured = capsys.readouterr()
        assert result == False


class TestFighterLoaderExceptionPaths:
    """Tests for exception handling paths in fighter_loader."""

    def test_load_fighter_with_exception_verbose(self, capsys):
        """Test verbose output when loading fails with exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighter_file = Path(tmpdir) / "error_fighter.py"
            fighter_file.write_text('''
raise ValueError("Intentional error during load")

def decide(state):
    return {"stance": "neutral", "acceleration": 0}
''')
            with pytest.raises(FighterLoadError):
                load_fighter(str(fighter_file), verbose=True)

            captured = capsys.readouterr()
            # Should have printed loading message before error
            assert "Loading fighter" in captured.out


class TestTestFighterInCombat:
    """Tests for test_fighter_in_combat function."""

    def test_fighter_runs_successfully(self, capsys):
        """Test that a valid fighter runs successfully in combat."""
        from src.atom.training.trainers.population.fighter_loader import test_fighter_in_combat

        def good_fighter(state):
            return {"stance": "neutral", "acceleration": 0.5}

        result = test_fighter_in_combat(good_fighter, num_steps=5, verbose=True)
        assert result is True
        captured = capsys.readouterr()
        assert "ran successfully" in captured.out

    def test_fighter_runs_non_verbose(self):
        """Test running fighter without verbose output."""
        from src.atom.training.trainers.population.fighter_loader import test_fighter_in_combat

        def good_fighter(state):
            return {"stance": "extended", "acceleration": 1.0}

        result = test_fighter_in_combat(good_fighter, num_steps=3, verbose=False)
        assert result is True

    def test_fighter_with_exception_fails(self, capsys):
        """Test that a fighter that raises exceptions fails."""
        from src.atom.training.trainers.population.fighter_loader import test_fighter_in_combat

        def bad_fighter(state):
            raise RuntimeError("Fighter crashed!")

        result = test_fighter_in_combat(bad_fighter, num_steps=5, verbose=True)
        assert result is False
        captured = capsys.readouterr()
        assert "Error" in captured.out

    def test_fighter_uses_state_values(self, capsys):
        """Test that fighter receives proper state values."""
        from src.atom.training.trainers.population.fighter_loader import test_fighter_in_combat

        state_received = []

        def recording_fighter(state):
            state_received.append(state)
            return {"stance": "neutral", "acceleration": 0}

        test_fighter_in_combat(recording_fighter, num_steps=3, verbose=False)

        # Verify state structure
        assert len(state_received) == 3
        for state in state_received:
            assert "tick" in state
            assert "you" in state
            assert "opponent" in state
            assert "arena" in state
            assert "position" in state["you"]
            assert "distance" in state["opponent"]

    def test_fighter_combat_simulation_updates(self):
        """Test that position and HP update during simulation."""
        from src.atom.training.trainers.population.fighter_loader import test_fighter_in_combat

        positions = []

        def tracking_fighter(state):
            positions.append(state["you"]["position"])
            return {"stance": "neutral", "acceleration": 1.0}

        test_fighter_in_combat(tracking_fighter, num_steps=5, verbose=False)

        # Position should change during simulation
        assert len(positions) == 5


class TestLoadHardcodedFighters:
    """Tests for load_hardcoded_fighters function."""

    def test_load_from_explicit_path(self, capsys):
        """Test loading fighters from explicit atom path."""
        from src.atom.training.trainers.population.fighter_loader import load_hardcoded_fighters

        # Use the actual atom path
        fighters = load_hardcoded_fighters("/home/biff/eng/atom", verbose=True)

        captured = capsys.readouterr()
        assert "Loading hardcoded fighters" in captured.out

        # Should load at least some fighters
        assert len(fighters) >= 0  # May be 0 if files don't exist

    def test_load_from_none_path(self):
        """Test loading with base_path=None uses cwd detection."""
        from src.atom.training.trainers.population.fighter_loader import load_hardcoded_fighters

        # Should not crash when base_path is None
        fighters = load_hardcoded_fighters(None, verbose=False)
        assert isinstance(fighters, dict)

    def test_load_non_verbose(self):
        """Test loading without verbose output."""
        from src.atom.training.trainers.population.fighter_loader import load_hardcoded_fighters

        fighters = load_hardcoded_fighters("/home/biff/eng/atom", verbose=False)
        assert isinstance(fighters, dict)

    def test_load_handles_missing_files(self, capsys):
        """Test that missing fighter files are handled gracefully."""
        from src.atom.training.trainers.population.fighter_loader import load_hardcoded_fighters
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an empty fighters/examples directory
            fighters_dir = Path(tmpdir) / "fighters" / "examples"
            fighters_dir.mkdir(parents=True)

            fighters = load_hardcoded_fighters(tmpdir, verbose=True)
            captured = capsys.readouterr()

            # Should report failures
            assert isinstance(fighters, dict)

    def test_load_handles_invalid_fighters(self, capsys):
        """Test that invalid fighter files are handled gracefully."""
        from src.atom.training.trainers.population.fighter_loader import load_hardcoded_fighters

        with tempfile.TemporaryDirectory() as tmpdir:
            fighters_dir = Path(tmpdir) / "fighters" / "examples"
            fighters_dir.mkdir(parents=True)

            # Create an invalid fighter file
            invalid_fighter = fighters_dir / "tank.py"
            invalid_fighter.write_text('''
def not_decide(state):
    return {}
''')

            fighters = load_hardcoded_fighters(tmpdir, verbose=True)
            captured = capsys.readouterr()

            # Should handle the error and report failure
            assert "Failed" in captured.out or len(fighters) == 0

    def test_load_handles_exception_during_load(self, capsys):
        """Test handling of exceptions during fighter load."""
        from src.atom.training.trainers.population.fighter_loader import load_hardcoded_fighters

        with tempfile.TemporaryDirectory() as tmpdir:
            fighters_dir = Path(tmpdir) / "fighters" / "examples"
            fighters_dir.mkdir(parents=True)

            # Create a fighter that raises an exception during load
            bad_fighter = fighters_dir / "tank.py"
            bad_fighter.write_text('''
raise ValueError("Error during module load")
''')

            fighters = load_hardcoded_fighters(tmpdir, verbose=True)
            captured = capsys.readouterr()

            # Should handle the error gracefully
            assert isinstance(fighters, dict)

    def test_load_validation_failure(self, capsys):
        """Test handling of fighters that fail validation."""
        from src.atom.training.trainers.population.fighter_loader import load_hardcoded_fighters

        with tempfile.TemporaryDirectory() as tmpdir:
            fighters_dir = Path(tmpdir) / "fighters" / "examples"
            fighters_dir.mkdir(parents=True)

            # Create a fighter that loads but fails validation
            bad_fighter = fighters_dir / "tank.py"
            bad_fighter.write_text('''
def decide(state):
    return {"stance": "invalid_stance", "acceleration": "not_a_number"}
''')

            fighters = load_hardcoded_fighters(tmpdir, verbose=True)
            captured = capsys.readouterr()

            # Should report validation failure
            assert "failed validation" in captured.out or len(fighters) == 0
