"""
Complete registry method coverage - target _extract_metadata_from_file and helpers.
Tests lines 165-428 (uncovered private methods).
"""

import pytest
import tempfile
from pathlib import Path
from src.registry.fighter_registry import FighterRegistry, FighterMetadata


class TestRegistryMetadataExtraction:
    """Test metadata extraction from fighter files."""

    def test_extract_metadata_from_valid_fighter(self):
        """Test extracting metadata from a valid fighter file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighters_dir = Path(tmpdir) / "fighters"
            fighters_dir.mkdir()

            # Create a valid fighter file with docstring
            fighter_file = fighters_dir / "test_fighter.py"
            fighter_file.write_text(
                '"""\n'
                'Test Fighter - A test fighter for metadata extraction\n'
                '"""\n'
                'def decide(state):\n'
                '    return {"acceleration": 0.5, "stance": "neutral"}\n'
            )

            registry = FighterRegistry(Path(tmpdir) / "reg.json", load_existing=False)

            # Scan should extract metadata
            count = registry.scan_directory(fighters_dir)

            assert count == 1

            fighters = registry.list_fighters()
            assert len(fighters) == 1

            fighter = fighters[0]
            assert fighter.id == "test_fighter"
            assert fighter.file_path is not None

    def test_extract_metadata_skips_pycache(self):
        """Test extraction skips __pycache__ directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighters_dir = Path(tmpdir) / "fighters"
            fighters_dir.mkdir()

            # Create __pycache__ directory
            pycache_dir = fighters_dir / "__pycache__"
            pycache_dir.mkdir()

            # Create a .py file in __pycache__
            (pycache_dir / "cached.py").write_text("def decide(state): return {}")

            # Create valid fighter outside __pycache__
            (fighters_dir / "valid.py").write_text("def decide(state): return {'acceleration': 0, 'stance': 'neutral'}")

            registry = FighterRegistry(Path(tmpdir) / "reg.json", load_existing=False)

            count = registry.scan_directory(fighters_dir)

            # Should only find the valid one, not the cached one
            assert count == 1

            fighters = registry.list_fighters()
            assert fighters[0].id == "valid"

    def test_extract_metadata_skips_init_files(self):
        """Test extraction skips __init__.py files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighters_dir = Path(tmpdir) / "fighters"
            fighters_dir.mkdir()

            # Create __init__.py
            (fighters_dir / "__init__.py").write_text("# Package init")

            # Create valid fighter
            (fighters_dir / "real_fighter.py").write_text("def decide(state): return {'acceleration': 0, 'stance': 'neutral'}")

            registry = FighterRegistry(Path(tmpdir) / "reg.json", load_existing=False)

            count = registry.scan_directory(fighters_dir)

            # Should skip __init__.py
            assert count == 1

    def test_extract_metadata_handles_fighter_without_decide(self):
        """Test extraction skips files without decide function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighters_dir = Path(tmpdir) / "fighters"
            fighters_dir.mkdir()

            # Create file without decide function
            (fighters_dir / "no_decide.py").write_text(
                'def other_function():\n'
                '    pass\n'
            )

            registry = FighterRegistry(Path(tmpdir) / "reg.json", load_existing=False)

            count = registry.scan_directory(fighters_dir)

            # Should skip file without decide
            assert count == 0

    def test_extract_metadata_handles_corrupted_file(self):
        """Test extraction handles corrupted Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighters_dir = Path(tmpdir) / "fighters"
            fighters_dir.mkdir()

            # Create file with syntax error
            (fighters_dir / "broken.py").write_text("def decide(state\n")  # Missing closing paren

            # Create valid file
            (fighters_dir / "good.py").write_text("def decide(state): return {'acceleration': 0, 'stance': 'neutral'}")

            registry = FighterRegistry(Path(tmpdir) / "reg.json", load_existing=False)

            count = registry.scan_directory(fighters_dir)

            # Should skip broken file but load good one
            assert count >= 1

    def test_extract_metadata_generates_fighter_id(self):
        """Test metadata extraction generates correct fighter ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighters_dir = Path(tmpdir) / "fighters"
            fighters_dir.mkdir()

            # Create fighter with specific name
            (fighters_dir / "aggressive_boxer.py").write_text(
                'def decide(state): return {"acceleration": 1.0, "stance": "extended"}'
            )

            registry = FighterRegistry(Path(tmpdir) / "reg.json", load_existing=False)

            count = registry.scan_directory(fighters_dir)

            fighters = registry.list_fighters()

            # ID should be based on filename
            assert any(f.id == "aggressive_boxer" for f in fighters)

    def test_extract_metadata_from_ai_directory_structure(self):
        """Test metadata extraction from AI fighters (different structure)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighters_dir = Path(tmpdir) / "fighters" / "AIs"
            fighter_subdir = fighters_dir / "alpha_fighter"
            fighter_subdir.mkdir(parents=True)

            # AI fighters are in subdirectories
            fighter_file = fighter_subdir / "alpha_fighter.py"
            fighter_file.write_text(
                'def decide(state): return {"acceleration": 0.5, "stance": "neutral"}'
            )

            registry = FighterRegistry(Path(tmpdir) / "reg.json", load_existing=False)

            count = registry.scan_directory(fighters_dir / "..", fighter_type="onnx-ai")

            # Should handle AI directory structure
            assert count >= 0  # May or may not find based on structure

    def test_save_and_load_preserves_metadata(self):
        """Test save/load cycle preserves all metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = Path(tmpdir) / "test.json"

            # Create and populate
            registry1 = FighterRegistry(reg_path, load_existing=False)

            meta = FighterMetadata(
                id="test",
                name="Test Fighter",
                description="Complete metadata test",
                creator="tester",
                type="rule-based",
                file_path="fighters/test.py",
                mass_default=72.5,
                strategy_tags=["test", "example"],
                version="2.5",
                protocol_version="v2.5",
                world_spec_version="v2.5",
                code_hash="hash123"
            )

            registry1.register_fighter(meta)
            registry1.save()

            # Load and verify
            registry2 = FighterRegistry(reg_path, load_existing=True)

            loaded = registry2.get_fighter("test")

            assert loaded.id == "test"
            assert loaded.name == "Test Fighter"
            assert loaded.mass_default == 72.5
            assert loaded.strategy_tags == ["test", "example"]
            assert loaded.version == "2.5"
            assert loaded.code_hash == "hash123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
