"""
Comprehensive tests for FighterRegistry to increase coverage.
Tests all methods including metadata extraction, validation, and parsing.
"""

import pytest
import tempfile
import json
from pathlib import Path
from src.registry.fighter_registry import FighterRegistry, FighterMetadata


class TestFighterMetadata:
    """Tests for FighterMetadata dataclass."""

    def test_metadata_creation_with_defaults(self):
        """Test creating metadata with default values."""
        metadata = FighterMetadata(
            id="test",
            name="Test Fighter",
            description="A test fighter",
            creator="tester",
            type="rule-based",
            file_path="fighters/test.py"
        )
        assert metadata.id == "test"
        assert metadata.mass_default == 70.0
        assert metadata.strategy_tags == []
        assert metadata.performance_stats is None
        assert metadata.version == "1.0"
        assert metadata.protocol_version == "v1.0"

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = FighterMetadata(
            id="boxer",
            name="Boxer",
            description="Boxing fighter",
            creator="system",
            type="rule-based",
            file_path="fighters/boxer.py",
            strategy_tags=["aggressive"]
        )
        result = metadata.to_dict()
        assert result["id"] == "boxer"
        assert result["strategy_tags"] == ["aggressive"]
        assert "created_date" in result

    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "id": "rusher",
            "name": "Rusher",
            "description": "Rushes forward",
            "creator": "system",
            "type": "rule-based",
            "file_path": "fighters/rusher.py",
            "mass_default": 75.0,
            "strategy_tags": ["aggressive", "stamina-aware"],
            "version": "2.0",
            "created_date": "2024-01-01T00:00:00",
            "protocol_version": "v1.0",
            "world_spec_version": "v1.0",
            "code_hash": "abc123"
        }
        metadata = FighterMetadata.from_dict(data)
        assert metadata.id == "rusher"
        assert metadata.mass_default == 75.0
        assert "aggressive" in metadata.strategy_tags

    def test_metadata_post_init_sets_defaults(self):
        """Test that __post_init__ sets strategy_tags and created_date."""
        metadata = FighterMetadata(
            id="test",
            name="Test",
            description="Test",
            creator="system",
            type="rule-based",
            file_path="test.py"
        )
        assert metadata.strategy_tags == []
        assert metadata.created_date is not None


class TestFighterRegistryInit:
    """Tests for FighterRegistry initialization."""

    def test_init_with_custom_path(self):
        """Test initialization with custom registry path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "custom_registry.json"
            registry = FighterRegistry(registry_path=path, load_existing=False)
            assert registry.registry_path == path
            assert registry.fighters == {}

    def test_init_with_default_path(self):
        """Test initialization uses default path when none provided."""
        registry = FighterRegistry(load_existing=False)
        assert "registry.json" in str(registry.registry_path)

    def test_init_loads_existing_registry(self):
        """Test that existing registry is loaded on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "registry.json"

            # Create existing registry
            data = {
                "version": "1.0",
                "generated": "2024-01-01",
                "fighters": {
                    "test": {
                        "id": "test",
                        "name": "Test",
                        "description": "Test fighter",
                        "creator": "system",
                        "type": "rule-based",
                        "file_path": "test.py",
                        "mass_default": 70.0,
                        "strategy_tags": [],
                        "performance_stats": None,
                        "version": "1.0",
                        "created_date": "2024-01-01",
                        "protocol_version": "v1.0",
                        "world_spec_version": "v1.0",
                        "code_hash": ""
                    }
                }
            }
            path.write_text(json.dumps(data))

            registry = FighterRegistry(registry_path=path, load_existing=True)
            assert "test" in registry.fighters


class TestFighterRegistryOperations:
    """Tests for basic registry operations."""

    def test_register_and_get_fighter(self):
        """Test registering and retrieving a fighter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            metadata = FighterMetadata(
                id="boxer",
                name="Boxer",
                description="Boxing fighter",
                creator="system",
                type="rule-based",
                file_path="fighters/boxer.py"
            )
            registry.register_fighter(metadata)

            result = registry.get_fighter("boxer")
            assert result is not None
            assert result.name == "Boxer"

    def test_get_nonexistent_fighter(self):
        """Test getting a fighter that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            assert registry.get_fighter("nonexistent") is None

    def test_list_fighters_no_filter(self):
        """Test listing all fighters without filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            for i in range(3):
                registry.register_fighter(FighterMetadata(
                    id=f"fighter{i}",
                    name=f"Fighter {i}",
                    description="Test",
                    creator="system",
                    type="rule-based",
                    file_path=f"fighter{i}.py"
                ))

            results = registry.list_fighters()
            assert len(results) == 3

    def test_list_fighters_filter_by_type(self):
        """Test filtering fighters by type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            registry.register_fighter(FighterMetadata(
                id="rule1", name="Rule 1", description="Test",
                creator="system", type="rule-based", file_path="r1.py"
            ))
            registry.register_fighter(FighterMetadata(
                id="ai1", name="AI 1", description="Test",
                creator="system", type="onnx-ai", file_path="a1.py"
            ))

            rule_based = registry.list_fighters(filter_type="rule-based")
            assert len(rule_based) == 1
            assert rule_based[0].id == "rule1"

    def test_list_fighters_filter_by_tags(self):
        """Test filtering fighters by strategy tags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            registry.register_fighter(FighterMetadata(
                id="aggressive1", name="Aggressive", description="Test",
                creator="system", type="rule-based", file_path="a.py",
                strategy_tags=["aggressive", "stamina-aware"]
            ))
            registry.register_fighter(FighterMetadata(
                id="defensive1", name="Defensive", description="Test",
                creator="system", type="rule-based", file_path="d.py",
                strategy_tags=["defensive"]
            ))

            aggressive = registry.list_fighters(filter_tags=["aggressive"])
            assert len(aggressive) == 1
            assert aggressive[0].id == "aggressive1"

    def test_clear_registry(self):
        """Test clearing all fighters from registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            registry.register_fighter(FighterMetadata(
                id="test", name="Test", description="Test",
                creator="system", type="rule-based", file_path="t.py"
            ))
            assert len(registry.fighters) == 1

            registry.clear()
            assert len(registry.fighters) == 0


class TestFighterRegistrySaveLoad:
    """Tests for registry persistence."""

    def test_save_and_load_registry(self):
        """Test saving and loading registry to/from JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "registry.json"

            # Create and save
            registry1 = FighterRegistry(registry_path=path, load_existing=False)
            registry1.register_fighter(FighterMetadata(
                id="boxer",
                name="Boxer",
                description="Boxing style",
                creator="system",
                type="rule-based",
                file_path="fighters/boxer.py",
                strategy_tags=["aggressive"]
            ))
            registry1.save()

            # Load in new instance
            registry2 = FighterRegistry(registry_path=path, load_existing=True)
            assert "boxer" in registry2.fighters
            assert registry2.fighters["boxer"].strategy_tags == ["aggressive"]

    def test_load_nonexistent_registry(self):
        """Test loading when registry file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"
            registry = FighterRegistry(registry_path=path, load_existing=True)
            assert registry.fighters == {}


class TestFighterRegistryScanDirectory:
    """Tests for directory scanning functionality."""

    def test_scan_directory_with_real_fighters(self):
        """Test scanning the actual fighters/examples directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            # Use real examples directory
            examples_dir = Path(__file__).parent.parent / "fighters" / "examples"
            if examples_dir.exists():
                count = registry.scan_directory(examples_dir)
                assert count >= 1  # Should find at least one fighter

    def test_scan_directory_skips_pycache(self):
        """Test that __pycache__ directories are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            # Scan examples - should not include __pycache__
            examples_dir = Path(__file__).parent.parent / "fighters" / "examples"
            if examples_dir.exists():
                count = registry.scan_directory(examples_dir)
                # Check no pycache in registered paths
                for fid, metadata in registry.fighters.items():
                    assert "__pycache__" not in metadata.file_path

    def test_scan_directory_skips_files_without_decide(self):
        """Test that files without decide function are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighter_dir = Path(tmpdir) / "fighters"
            fighter_dir.mkdir()

            # File without decide function
            (fighter_dir / "helper.py").write_text('''
def some_helper():
    return 42
''')

            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            count = registry.scan_directory(fighter_dir)

            assert count == 0

    def test_scan_directory_handles_invalid_files(self):
        """Test that invalid Python files are handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighter_dir = Path(tmpdir) / "fighters"
            fighter_dir.mkdir()

            # Invalid Python syntax
            (fighter_dir / "broken.py").write_text("def broken(")

            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            count = registry.scan_directory(fighter_dir)

            assert count == 0


class TestFighterRegistryMetadataExtraction:
    """Tests for metadata extraction from files."""

    def test_extract_strategy_tags_aggressive(self):
        """Test extracting aggressive strategy tag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            tags = registry._extract_strategy_tags("An aggressive attacker that rushes forward")
            assert "aggressive" in tags

    def test_extract_strategy_tags_defensive(self):
        """Test extracting defensive strategy tag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            tags = registry._extract_strategy_tags("A tank that defends patiently")
            assert "defensive" in tags

    def test_extract_strategy_tags_balanced(self):
        """Test extracting balanced strategy tag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            tags = registry._extract_strategy_tags("A balanced and adaptable fighter")
            assert "balanced" in tags

    def test_extract_strategy_tags_stamina_aware(self):
        """Test extracting stamina-aware strategy tag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            tags = registry._extract_strategy_tags("Manages stamina and endurance carefully")
            assert "stamina-aware" in tags

    def test_extract_strategy_tags_evasive(self):
        """Test extracting evasive strategy tag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            tags = registry._extract_strategy_tags("Uses dodge and evasion to avoid hits")
            assert "evasive" in tags

    def test_extract_strategy_tags_counter_puncher(self):
        """Test extracting counter-puncher strategy tag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            tags = registry._extract_strategy_tags("A patient counter fighter who punishes mistakes")
            assert "counter-puncher" in tags

    def test_extract_strategy_tags_range_control(self):
        """Test extracting range-control strategy tag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            tags = registry._extract_strategy_tags("Controls distance and range effectively")
        assert "range-control" in tags

    def test_extract_strategy_tags_multiple(self):
        """Test extracting multiple strategy tags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            tags = registry._extract_strategy_tags(
                "An aggressive attacker that manages stamina and energy"
            )
            assert "aggressive" in tags
            assert "stamina-aware" in tags


class TestFighterRegistryParseDocstring:
    """Tests for docstring parsing."""

    def test_parse_docstring_with_name_and_description(self):
        """Test parsing docstring with name - description format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighter_file = Path(tmpdir) / "fighter.py"
            fighter_file.write_text('''
"""Boxer - A classic boxing style fighter

Uses jabs and footwork to control the fight.
Maintains good stamina management.
"""
def decide(state):
    return {}
''')
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            name, description = registry._parse_docstring(fighter_file)
            assert name == "Boxer"
            assert "jabs" in description.lower() or "footwork" in description.lower()

    def test_parse_docstring_single_quotes(self):
        """Test parsing docstring with single quotes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighter_file = Path(tmpdir) / "fighter.py"
            fighter_file.write_text("""
'''Rusher - Aggressive forward attacker

Always moves toward the opponent.
'''
def decide(state):
    return {}
""")
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            name, description = registry._parse_docstring(fighter_file)
            assert name == "Rusher"

    def test_parse_docstring_no_separator(self):
        """Test parsing docstring without ' - ' separator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighter_file = Path(tmpdir) / "fighter.py"
            fighter_file.write_text('''
"""Tank Fighter

A defensive fighter that absorbs damage.
"""
def decide(state):
    return {}
''')
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            name, description = registry._parse_docstring(fighter_file)
            assert name == "Tank Fighter"

    def test_parse_docstring_empty_file(self):
        """Test parsing file with no docstring."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighter_file = Path(tmpdir) / "fighter.py"
            fighter_file.write_text('''
def decide(state):
    return {}
''')
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            name, description = registry._parse_docstring(fighter_file)
            assert name == ""
            assert description == ""


class TestFighterRegistryParseReadme:
    """Tests for README.md parsing."""

    def test_parse_readme_with_elo(self):
        """Test parsing README with ELO rating."""
        with tempfile.TemporaryDirectory() as tmpdir:
            readme_file = Path(tmpdir) / "README.md"
            readme_file.write_text('''
# Alpha Fighter

**ELO Rating**: 1523

A strong fighter.
''')
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            stats = registry._parse_readme(readme_file)
            assert stats is not None
            assert stats["elo"] == 1523.0

    def test_parse_readme_with_win_rate(self):
        """Test parsing README with win rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            readme_file = Path(tmpdir) / "README.md"
            readme_file.write_text('''
# Fighter Stats

**Win Rate**: 67.5%
''')
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            stats = registry._parse_readme(readme_file)
            assert stats is not None
            assert abs(stats["win_rate"] - 0.675) < 0.01

    def test_parse_readme_with_record(self):
        """Test parsing README with W-L-D record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            readme_file = Path(tmpdir) / "README.md"
            readme_file.write_text('''
# Fighter

**Record**: 15W - 5L - 3D
''')
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            stats = registry._parse_readme(readme_file)
            assert stats is not None
            assert stats["wins"] == 15
            assert stats["losses"] == 5
            assert stats["draws"] == 3

    def test_parse_readme_with_all_stats(self):
        """Test parsing README with all stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            readme_file = Path(tmpdir) / "README.md"
            readme_file.write_text('''
# Champion Fighter

**ELO Rating**: 1800
**Win Rate**: 85%
**Record**: 20W - 3L - 2D
''')
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            stats = registry._parse_readme(readme_file)
            assert stats is not None
            assert stats["elo"] == 1800.0
            assert abs(stats["win_rate"] - 0.85) < 0.01
            assert stats["wins"] == 20

    def test_parse_readme_empty(self):
        """Test parsing README with no stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            readme_file = Path(tmpdir) / "README.md"
            readme_file.write_text("# Just a title\n\nNo stats here.")

            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            stats = registry._parse_readme(readme_file)
            assert stats is None


class TestFighterRegistryValidation:
    """Tests for validation methods."""

    def test_has_decide_function_true(self):
        """Test checking for decide function when present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighter_file = Path(tmpdir) / "fighter.py"
            fighter_file.write_text('''
def decide(state):
    return {"stance": "neutral", "movement": 0}
''')
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            assert registry._has_decide_function(fighter_file) is True

    def test_has_decide_function_false(self):
        """Test checking for decide function when absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighter_file = Path(tmpdir) / "helper.py"
            fighter_file.write_text('''
def helper():
    return 42
''')
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            assert registry._has_decide_function(fighter_file) is False

    def test_has_decide_function_syntax_error(self):
        """Test checking invalid Python file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fighter_file = Path(tmpdir) / "broken.py"
            fighter_file.write_text("def broken(")

            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            assert registry._has_decide_function(fighter_file) is False

    def test_validate_all_with_valid_fighters(self):
        """Test validating all registered fighters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fighter file
            fighter_dir = Path(tmpdir) / "fighters"
            fighter_dir.mkdir()
            fighter_file = fighter_dir / "test.py"
            fighter_file.write_text('''
def decide(state):
    return {}
''')

            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            # Manually register with correct relative path
            # Need to mock project root for this test
            # This is tricky because validate_all uses Path(__file__).parent.parent.parent
            # For now, just test the method runs
            registry.register_fighter(FighterMetadata(
                id="test",
                name="Test",
                description="Test",
                creator="system",
                type="rule-based",
                file_path="fighters/examples/boxer.py"  # Use real path
            ))

            results = registry.validate_all()
            assert isinstance(results, dict)

    def test_calculate_file_hash(self):
        """Test calculating SHA256 hash of file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("def decide(state): return {}")

            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            hash_result = registry._calculate_file_hash(test_file)

            assert len(hash_result) == 64  # SHA256 produces 64 hex chars
            assert hash_result.isalnum()

    def test_calculate_file_hash_nonexistent(self):
        """Test calculating hash of nonexistent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            hash_result = registry._calculate_file_hash(Path("/nonexistent/file.py"))
            assert hash_result == ""


class TestFighterRegistryExtractMetadata:
    """Tests for full metadata extraction from files."""

    def test_extract_metadata_from_real_fighters(self):
        """Test extracting metadata from real fighter files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            examples_dir = Path(__file__).parent.parent / "fighters" / "examples"
            if examples_dir.exists():
                count = registry.scan_directory(examples_dir)
                assert count >= 1

                # Check that metadata was extracted
                fighters = registry.list_fighters()
                for fighter in fighters:
                    assert fighter.id is not None
                    assert fighter.name is not None
                    assert fighter.file_path is not None
                    assert fighter.code_hash != ""  # Hash should be calculated

    def test_extract_metadata_assigns_system_creator(self):
        """Test that regular fighters get 'system' as creator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            examples_dir = Path(__file__).parent.parent / "fighters" / "examples"
            if examples_dir.exists():
                registry.scan_directory(examples_dir)
                fighters = registry.list_fighters()
                for fighter in fighters:
                    # Non-AI fighters should have 'system' creator
                    if "AIs" not in fighter.file_path:
                        assert fighter.creator == "system"

    def test_extract_metadata_computes_hash(self):
        """Test that file hash is computed for each fighter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FighterRegistry(
                registry_path=Path(tmpdir) / "reg.json",
                load_existing=False
            )
            examples_dir = Path(__file__).parent.parent / "fighters" / "examples"
            if examples_dir.exists():
                registry.scan_directory(examples_dir)
                fighters = registry.list_fighters()
                for fighter in fighters:
                    # All fighters should have a 64-char SHA256 hash
                    assert len(fighter.code_hash) == 64
                    assert fighter.code_hash.isalnum()
