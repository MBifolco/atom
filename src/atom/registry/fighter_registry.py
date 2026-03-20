"""
Fighter Registry System

Manages the discovery, registration, and metadata for all fighters
in the Atom Combat platform.
"""

import json
import importlib.util
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib


@dataclass
class FighterMetadata:
    """
    Metadata for a registered fighter.

    Attributes:
        id: Unique identifier (e.g., "rusher", "alpha_g3")
        name: Display name
        description: Fighter description/strategy
        creator: Creator name or "system" for built-in fighters
        type: Fighter type ("rule-based", "onnx-ai", "custom")
        file_path: Path to fighter file (relative to project root)
        mass_default: Default mass for this fighter
        strategy_tags: List of strategy descriptors (e.g., ["aggressive", "stamina-aware"])
        performance_stats: Optional performance metrics (ELO, win_rate, etc.)
        version: Fighter version string
        created_date: ISO timestamp of creation
        protocol_version: Combat protocol version (e.g., "v1.0")
        world_spec_version: World config version (e.g., "v1.0")
        code_hash: SHA256 hash of fighter code for verification
    """
    id: str
    name: str
    description: str
    creator: str
    type: str
    file_path: str
    mass_default: float = 70.0
    strategy_tags: List[str] = None
    performance_stats: Optional[Dict[str, Any]] = None
    version: str = "1.0"
    created_date: str = None
    protocol_version: str = "v1.0"
    world_spec_version: str = "v1.0"
    code_hash: str = ""

    def __post_init__(self):
        if self.strategy_tags is None:
            self.strategy_tags = []
        if self.created_date is None:
            self.created_date = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FighterMetadata':
        """Create from dictionary."""
        return cls(**data)


class FighterRegistry:
    """
    Fighter Registry - manages all registered fighters.

    The registry provides:
    - Fighter discovery and scanning
    - Metadata storage and retrieval
    - Fighter validation
    - JSON persistence
    """

    def __init__(self, registry_path: Optional[Path] = None, load_existing: bool = True):
        """
        Initialize the fighter registry.

        Args:
            registry_path: Path to registry.json file. If None, uses default location.
            load_existing: Whether to load existing registry if it exists.
        """
        if registry_path is None:
            # Default to fighters/registry.json
            project_root = Path(__file__).parent.parent.parent.parent
            registry_path = project_root / "fighters" / "registry.json"

        self.registry_path = Path(registry_path)
        self.fighters: Dict[str, FighterMetadata] = {}

        # Load existing registry if it exists and requested
        if load_existing and self.registry_path.exists():
            self.load()

    def register_fighter(self, metadata: FighterMetadata) -> None:
        """
        Register a fighter in the registry.

        Args:
            metadata: Fighter metadata to register
        """
        self.fighters[metadata.id] = metadata

    def get_fighter(self, fighter_id: str) -> Optional[FighterMetadata]:
        """
        Get fighter metadata by ID.

        Args:
            fighter_id: Fighter ID

        Returns:
            FighterMetadata if found, None otherwise
        """
        return self.fighters.get(fighter_id)

    def list_fighters(self,
                     filter_type: Optional[str] = None,
                     filter_tags: Optional[List[str]] = None) -> List[FighterMetadata]:
        """
        List all fighters with optional filtering.

        Args:
            filter_type: Filter by fighter type (e.g., "rule-based", "onnx-ai")
            filter_tags: Filter by strategy tags (e.g., ["aggressive"])

        Returns:
            List of fighter metadata matching filters
        """
        results = list(self.fighters.values())

        if filter_type:
            results = [f for f in results if f.type == filter_type]

        if filter_tags:
            results = [f for f in results
                      if any(tag in f.strategy_tags for tag in filter_tags)]

        return results

    def clear(self) -> None:
        """Clear all fighters from the registry."""
        self.fighters = {}

    def scan_directory(self, directory: Path, fighter_type: str = "rule-based") -> int:
        """
        Scan a directory for fighter files and register them.

        Args:
            directory: Directory to scan for .py files
            fighter_type: Type to assign to discovered fighters

        Returns:
            Number of fighters registered
        """
        count = 0

        # Handle both flat directory and nested structure
        for py_file in directory.rglob("*.py"):
            # Skip __init__.py and __pycache__
            if py_file.name.startswith("__") or "__pycache__" in str(py_file):
                continue

            try:
                metadata = self._extract_metadata_from_file(py_file, fighter_type)
                if metadata:
                    self.register_fighter(metadata)
                    count += 1
            except Exception as e:
                print(f"Warning: Could not process {py_file}: {e}")
                continue

        return count

    def _extract_metadata_from_file(self, file_path: Path, fighter_type: str) -> Optional[FighterMetadata]:
        """
        Extract metadata from a fighter Python file.

        Attempts to extract information from:
        1. Module docstring
        2. README.md in same directory
        3. File name and structure

        Args:
            file_path: Path to fighter .py file
            fighter_type: Type to assign

        Returns:
            FighterMetadata if valid fighter, None otherwise
        """
        # Verify it has a decide function
        if not self._has_decide_function(file_path):
            return None

        # Get project root to make relative paths
        project_root = Path(__file__).parent.parent.parent.parent
        relative_path = file_path.relative_to(project_root)

        # Generate ID from path (e.g., "examples/rusher" -> "rusher")
        # For AIs: "AIs/Alpha/Alpha.py" -> "alpha"
        if "AIs" in file_path.parts:
            # AI fighter - use parent directory name
            fighter_id = file_path.parent.name.lower()
        else:
            # Regular fighter - use filename
            fighter_id = file_path.stem.lower()

        # Extract name and description from docstring
        name, description = self._parse_docstring(file_path)
        if not name:
            name = file_path.stem.replace("_", " ").title()
        if not description:
            description = f"A {fighter_type} fighter"

        # Check for README.md for AI fighters
        readme_path = file_path.parent / "README.md"
        performance_stats = None
        if readme_path.exists():
            performance_stats = self._parse_readme(readme_path)

        # Determine creator
        creator = "system"
        if "AIs" in file_path.parts:
            creator = "population-training"

        # Calculate code hash
        code_hash = self._calculate_file_hash(file_path)

        # Extract strategy tags from description
        strategy_tags = self._extract_strategy_tags(description)

        return FighterMetadata(
            id=fighter_id,
            name=name,
            description=description,
            creator=creator,
            type=fighter_type,
            file_path=str(relative_path),
            strategy_tags=strategy_tags,
            performance_stats=performance_stats,
            code_hash=code_hash
        )

    def _has_decide_function(self, file_path: Path) -> bool:
        """Check if Python file has a decide function."""
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location("temp_module", file_path)
            if not spec or not spec.loader:
                return False

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            return hasattr(module, 'decide') and callable(getattr(module, 'decide'))
        except Exception:
            return False

    def _parse_docstring(self, file_path: Path) -> tuple[str, str]:
        """
        Parse name and description from module docstring.

        Expected format:
        '''
        Fighter Name - Description

        Longer description...
        '''

        Returns:
            (name, description) tuple
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Find first docstring
            if '"""' in content:
                start = content.index('"""') + 3
                end = content.index('"""', start)
                docstring = content[start:end].strip()
            elif "'''" in content:
                start = content.index("'''") + 3
                end = content.index("'''", start)
                docstring = content[start:end].strip()
            else:
                return ("", "")

            lines = [line.strip() for line in docstring.split('\n') if line.strip()]
            if not lines:
                return ("", "")

            # First line is usually "Name - Description" or just "Name"
            first_line = lines[0]
            if " - " in first_line:
                name = first_line.split(" - ")[0].strip()
            else:
                name = first_line.strip()

            # Rest is description
            description = ' '.join(lines[1:]) if len(lines) > 1 else first_line

            return (name, description)
        except Exception:
            return ("", "")

    def _parse_readme(self, readme_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse performance stats from README.md.

        Looks for sections like:
        - **ELO Rating**: 1234
        - **Win Rate**: 67.5%
        - **Record**: 10W - 5L - 2D

        Returns:
            Dictionary of stats if found
        """
        try:
            with open(readme_path, 'r') as f:
                content = f.read()

            stats = {}

            # Look for ELO rating
            if "ELO Rating" in content:
                import re
                match = re.search(r'ELO Rating.*?(\d+(?:\.\d+)?)', content)
                if match:
                    stats['elo'] = float(match.group(1))

            # Look for Win Rate
            if "Win Rate" in content:
                import re
                match = re.search(r'Win Rate.*?(\d+(?:\.\d+)?)%', content)
                if match:
                    stats['win_rate'] = float(match.group(1)) / 100.0

            # Look for Record
            if "Record" in content:
                import re
                match = re.search(r'Record.*?(\d+)W.*?(\d+)L.*?(\d+)D', content)
                if match:
                    stats['wins'] = int(match.group(1))
                    stats['losses'] = int(match.group(2))
                    stats['draws'] = int(match.group(3))

            return stats if stats else None
        except Exception:
            return None

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file content."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""

    def _extract_strategy_tags(self, description: str) -> List[str]:
        """Extract strategy tags from description."""
        tags = []
        description_lower = description.lower()

        # Common strategy keywords
        if any(word in description_lower for word in ['aggress', 'attack', 'rush']):
            tags.append('aggressive')
        if any(word in description_lower for word in ['defend', 'tank', 'passive']):
            tags.append('defensive')
        if any(word in description_lower for word in ['balance', 'adapt', 'versatile']):
            tags.append('balanced')
        if any(word in description_lower for word in ['stamina', 'endurance', 'energy']):
            tags.append('stamina-aware')
        if any(word in description_lower for word in ['evade', 'dodge', 'mobile']):
            tags.append('evasive')
        if any(word in description_lower for word in ['counter', 'punish', 'patient']):
            tags.append('counter-puncher')
        if any(word in description_lower for word in ['range', 'distance', 'zone']):
            tags.append('range-control')

        return tags

    def save(self) -> None:
        """Save registry to JSON file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "generated": datetime.now().isoformat(),
            "fighters": {fid: metadata.to_dict()
                        for fid, metadata in self.fighters.items()}
        }

        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load registry from JSON file."""
        if not self.registry_path.exists():
            return

        with open(self.registry_path, 'r') as f:
            data = json.load(f)

        self.fighters = {
            fid: FighterMetadata.from_dict(metadata_dict)
            for fid, metadata_dict in data.get("fighters", {}).items()
        }

    def validate_all(self) -> Dict[str, bool]:
        """
        Validate that all registered fighters can be loaded.

        Returns:
            Dictionary mapping fighter_id to validation status (True/False)
        """
        results = {}
        project_root = Path(__file__).parent.parent.parent.parent

        for fighter_id, metadata in self.fighters.items():
            file_path = project_root / metadata.file_path
            results[fighter_id] = self._has_decide_function(file_path)

        return results
