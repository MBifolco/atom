"""
Tests for fighter registry system.
"""

import pytest
import tempfile
import json
from pathlib import Path
from src.registry import FighterRegistry
from src.atom.registry.fighter_registry import FighterMetadata


class TestFighterMetadata:
    """Test FighterMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating fighter metadata."""
        meta = FighterMetadata(
            id="test_fighter",
            name="Test Fighter",
            description="A test fighter",
            creator="test",
            type="rule-based",
            file_path="fighters/test.py"
        )

        assert meta.id == "test_fighter"
        assert meta.name == "Test Fighter"
        assert meta.mass_default == 70.0  # Default
        assert meta.strategy_tags == []

    def test_metadata_to_dict(self):
        """Test converting metadata to dict."""
        meta = FighterMetadata(
            id="test",
            name="Test",
            description="Test fighter",
            creator="test",
            type="rule-based",
            file_path="test.py"
        )

        d = meta.to_dict()

        assert d["id"] == "test"
        assert d["name"] == "Test"
        assert "created_date" in d

    def test_metadata_from_dict(self):
        """Test creating metadata from dict."""
        d = {
            "id": "test",
            "name": "Test",
            "description": "Test fighter",
            "creator": "test",
            "type": "rule-based",
            "file_path": "test.py",
            "mass_default": 65.0,
            "strategy_tags": ["aggressive"],
            "created_date": "2025-01-01T00:00:00"
        }

        meta = FighterMetadata.from_dict(d)

        assert meta.id == "test"
        assert meta.mass_default == 65.0
        assert "aggressive" in meta.strategy_tags


class TestFighterRegistry:
    """Test FighterRegistry class."""

    def test_registry_initialization(self):
        """Test registry initializes with empty fighters dict."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            registry_path = Path(f.name)

        try:
            registry = FighterRegistry(registry_path, load_existing=False)

            assert registry.registry_path == registry_path
            assert len(registry.fighters) == 0
        finally:
            if registry_path.exists():
                registry_path.unlink()

    def test_register_fighter(self):
        """Test registering a fighter."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            registry_path = Path(f.name)

        try:
            registry = FighterRegistry(registry_path, load_existing=False)

            meta = FighterMetadata(
                id="boxer",
                name="Boxer",
                description="Boxing style",
                creator="system",
                type="rule-based",
                file_path="fighters/examples/boxer.py"
            )

            registry.register_fighter(meta)

            assert len(registry.fighters) == 1
            assert "boxer" in registry.fighters
        finally:
            if registry_path.exists():
                registry_path.unlink()

    def test_get_fighter(self):
        """Test retrieving fighter by ID."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            registry_path = Path(f.name)

        try:
            registry = FighterRegistry(registry_path, load_existing=False)

            meta = FighterMetadata(
                id="test", name="Test", description="Test",
                creator="test", type="rule-based", file_path="test.py"
            )

            registry.register_fighter(meta)

            retrieved = registry.get_fighter("test")

            assert retrieved is not None
            assert retrieved.id == "test"
            assert retrieved.name == "Test"

            # Non-existent fighter
            none_fighter = registry.get_fighter("nonexistent")
            assert none_fighter is None
        finally:
            if registry_path.exists():
                registry_path.unlink()

    def test_list_fighters(self):
        """Test listing all fighters."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            registry_path = Path(f.name)

        try:
            registry = FighterRegistry(registry_path, load_existing=False)

            # Add multiple fighters
            for i in range(3):
                meta = FighterMetadata(
                    id=f"fighter_{i}",
                    name=f"Fighter {i}",
                    description="Test",
                    creator="test",
                    type="rule-based",
                    file_path=f"test_{i}.py"
                )
                registry.register_fighter(meta)

            fighters = registry.list_fighters()

            assert len(fighters) == 3
        finally:
            if registry_path.exists():
                registry_path.unlink()

    def test_list_fighters_filtered_by_type(self):
        """Test filtering fighters by type."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            registry_path = Path(f.name)

        try:
            registry = FighterRegistry(registry_path, load_existing=False)

            # Add different types
            registry.register_fighter(FighterMetadata(
                id="rule1", name="Rule 1", description="Test",
                creator="test", type="rule-based", file_path="test.py"
            ))
            registry.register_fighter(FighterMetadata(
                id="ai1", name="AI 1", description="Test",
                creator="test", type="onnx-ai", file_path="test.py"
            ))

            rule_based = registry.list_fighters(filter_type="rule-based")
            ai_fighters = registry.list_fighters(filter_type="onnx-ai")

            assert len(rule_based) == 1
            assert len(ai_fighters) == 1
            assert rule_based[0].id == "rule1"
            assert ai_fighters[0].id == "ai1"
        finally:
            if registry_path.exists():
                registry_path.unlink()

    def test_list_fighters_filtered_by_tags(self):
        """Test filtering fighters by strategy tags."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            registry_path = Path(f.name)

        try:
            registry = FighterRegistry(registry_path, load_existing=False)

            registry.register_fighter(FighterMetadata(
                id="aggressive", name="Aggressive", description="Test",
                creator="test", type="rule-based", file_path="test.py",
                strategy_tags=["aggressive", "offensive"]
            ))
            registry.register_fighter(FighterMetadata(
                id="defensive", name="Defensive", description="Test",
                creator="test", type="rule-based", file_path="test.py",
                strategy_tags=["defensive", "patient"]
            ))

            aggressive = registry.list_fighters(filter_tags=["aggressive"])
            defensive = registry.list_fighters(filter_tags=["defensive"])

            assert len(aggressive) == 1
            assert len(defensive) == 1
        finally:
            if registry_path.exists():
                registry_path.unlink()

    def test_clear_registry(self):
        """Test clearing all fighters from registry."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            registry_path = Path(f.name)

        try:
            registry = FighterRegistry(registry_path, load_existing=False)

            # Add fighter
            registry.register_fighter(FighterMetadata(
                id="test", name="Test", description="Test",
                creator="test", type="rule-based", file_path="test.py"
            ))

            assert len(registry.fighters) == 1

            # Clear
            registry.clear()

            assert len(registry.fighters) == 0
        finally:
            if registry_path.exists():
                registry_path.unlink()

    def test_save_and_load_registry(self):
        """Test saving and loading registry from JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            registry_path = Path(f.name)

        try:
            # Create and populate registry
            registry = FighterRegistry(registry_path, load_existing=False)

            registry.register_fighter(FighterMetadata(
                id="boxer", name="Boxer", description="Boxing style",
                creator="system", type="rule-based",
                file_path="fighters/examples/boxer.py",
                strategy_tags=["technical", "stamina-aware"]
            ))

            # Save
            registry.save()

            # Load into new registry
            registry2 = FighterRegistry(registry_path, load_existing=True)

            assert len(registry2.fighters) == 1
            assert "boxer" in registry2.fighters
            assert registry2.fighters["boxer"].strategy_tags == ["technical", "stamina-aware"]
        finally:
            if registry_path.exists():
                registry_path.unlink()

    def test_scan_directory(self):
        """Test scan_directory method exists and is callable."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            registry_path = Path(f.name)

        try:
            registry = FighterRegistry(registry_path, load_existing=False)

            # Test with non-existent directory (should handle gracefully)
            fake_dir = Path("/tmp/nonexistent_fighters_dir_xyz")
            count = registry.scan_directory(fake_dir, fighter_type="rule-based")

            # Should return 0 for non-existent directory
            assert count == 0
        finally:
            if registry_path.exists():
                registry_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
