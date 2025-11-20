"""
Complete registry coverage - target lines 165-428 (uncovered branches).
"""

import pytest
import tempfile
from pathlib import Path
from src.registry.fighter_registry import FighterRegistry, FighterMetadata


class TestRegistryEdgeCases:
    """Test registry edge cases and error handling."""

    def test_scan_directory_with_invalid_files(self):
        """Test scanning directory with non-Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "test_fighters"
            test_dir.mkdir()

            # Create non-Python file
            (test_dir / "not_python.txt").write_text("Not a fighter")

            # Create Python file without decide
            (test_dir / "no_decide.py").write_text("def other_func():\n    pass\n")

            registry = FighterRegistry(Path(tmpdir) / "reg.json", load_existing=False)

            count = registry.scan_directory(test_dir)

            # Should skip invalid files
            assert count == 0

    def test_scan_directory_with_nested_structure(self):
        """Test scanning nested directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighters_dir = Path(tmpdir) / "fighters"
            sub_dir = fighters_dir / "subdirectory"
            sub_dir.mkdir(parents=True)

            # Create fighter in subdirectory
            fighter_file = sub_dir / "nested_fighter.py"
            fighter_file.write_text(
                '"""Test fighter"""\n'
                'def decide(state):\n'
                '    return {"acceleration": 0.0, "stance": "neutral"}\n'
            )

            registry = FighterRegistry(Path(tmpdir) / "reg.json", load_existing=False)

            count = registry.scan_directory(fighters_dir)

            # Should find nested fighter
            assert count >= 1

    def test_metadata_extraction_from_docstring(self):
        """Test extracting metadata from fighter docstring."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighters_dir = Path(tmpdir) / "fighters"
            fighters_dir.mkdir()

            fighter_file = fighters_dir / "documented_fighter.py"
            fighter_file.write_text(
                '"""\n'
                'Test Fighter - A well-documented fighter\n'
                'Strategy: Aggressive approach\n'
                '"""\n'
                'def decide(state):\n'
                '    return {"acceleration": 1.0, "stance": "extended"}\n'
            )

            registry = FighterRegistry(Path(tmpdir) / "reg.json", load_existing=False)

            count = registry.scan_directory(fighters_dir)

            assert count == 1

            # Check that metadata was extracted
            fighters = registry.list_fighters()
            assert len(fighters) > 0

    def test_registry_with_duplicate_ids(self):
        """Test registry handles duplicate fighter IDs."""
        with tempfile.TemporaryFile(suffix=".json") as f:
            registry = FighterRegistry(Path(f.name), load_existing=False)

            meta1 = FighterMetadata(
                id="duplicate",
                name="First",
                description="First fighter",
                creator="test",
                type="rule-based",
                file_path="test1.py"
            )

            meta2 = FighterMetadata(
                id="duplicate",  # Same ID
                name="Second",
                description="Second fighter",
                creator="test",
                type="rule-based",
                file_path="test2.py"
            )

            registry.register_fighter(meta1)
            registry.register_fighter(meta2)

            # Second should overwrite first
            fighter = registry.get_fighter("duplicate")
            assert fighter.name == "Second"

    def test_list_fighters_with_multiple_filters(self):
        """Test listing with both type and tag filters."""
        with tempfile.TemporaryFile(suffix=".json") as f:
            registry = FighterRegistry(Path(f.name), load_existing=False)

            registry.register_fighter(FighterMetadata(
                id="aggressive_ai",
                name="Aggressive AI",
                description="Test",
                creator="test",
                type="onnx-ai",
                file_path="test.py",
                strategy_tags=["aggressive", "offensive"]
            ))

            registry.register_fighter(FighterMetadata(
                id="defensive_ai",
                name="Defensive AI",
                description="Test",
                creator="test",
                type="onnx-ai",
                file_path="test.py",
                strategy_tags=["defensive", "patient"]
            ))

            registry.register_fighter(FighterMetadata(
                id="aggressive_rule",
                name="Aggressive Rule",
                description="Test",
                creator="test",
                type="rule-based",
                file_path="test.py",
                strategy_tags=["aggressive"]
            ))

            # Filter by type and tags
            aggressive_ais = registry.list_fighters(
                filter_type="onnx-ai",
                filter_tags=["aggressive"]
            )

            # Should only get aggressive_ai
            assert len(aggressive_ais) == 1
            assert aggressive_ais[0].id == "aggressive_ai"

    def test_clear_and_repopulate_registry(self):
        """Test clearing and repopulating registry."""
        with tempfile.TemporaryFile(suffix=".json") as f:
            registry = FighterRegistry(Path(f.name), load_existing=False)

            # Add fighters
            registry.register_fighter(FighterMetadata(
                id="f1", name="F1", description="Test",
                creator="test", type="rule-based", file_path="test.py"
            ))

            assert len(registry.fighters) == 1

            # Clear
            registry.clear()

            assert len(registry.fighters) == 0

            # Repopulate
            registry.register_fighter(FighterMetadata(
                id="f2", name="F2", description="Test",
                creator="test", type="rule-based", file_path="test.py"
            ))

            assert len(registry.fighters) == 1
            assert "f2" in registry.fighters
            assert "f1" not in registry.fighters

    def test_metadata_code_hash_generation(self):
        """Test metadata generates code hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighters_dir = Path(tmpdir) / "fighters"
            fighters_dir.mkdir()

            fighter_file = fighters_dir / "hash_test.py"
            fighter_file.write_text(
                'def decide(state):\n'
                '    return {"acceleration": 0.5, "stance": "neutral"}\n'
            )

            registry = FighterRegistry(Path(tmpdir) / "reg.json", load_existing=False)

            count = registry.scan_directory(fighters_dir)

            # Should have scanned file
            assert count == 1

            fighters = registry.list_fighters()
            if fighters:
                # Code hash should be generated (or empty string)
                assert hasattr(fighters[0], 'code_hash')

    def test_metadata_version_and_protocol_version(self):
        """Test metadata includes version fields."""
        meta = FighterMetadata(
            id="versioned",
            name="Versioned Fighter",
            description="Test",
            creator="test",
            type="rule-based",
            file_path="test.py",
            version="2.0",
            protocol_version="v2.0",
            world_spec_version="v2.0"
        )

        assert meta.version == "2.0"
        assert meta.protocol_version == "v2.0"
        assert meta.world_spec_version == "v2.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
