"""Guardrail tests for preferred import paths in current repo surfaces."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_SURFACE_PATHS = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "docs" / "README.md",
    PROJECT_ROOT / "docs" / "PLATFORM_ARCHITECTURE.md",
    PROJECT_ROOT / "docs" / "BOXING_COMBAT_SYSTEM.md",
    PROJECT_ROOT / "docs" / "PROGRESSIVE_TRAINING.md",
    PROJECT_ROOT / "docs" / "QUICK_REFERENCE.md",
    PROJECT_ROOT / "docs" / "REPLAY_MONTAGE.md",
    PROJECT_ROOT / "docs" / "REFACTORING_PLAN_DETAILED.md",
    PROJECT_ROOT / "docs" / "REPO_REORGANIZATION_PLAN.md",
    PROJECT_ROOT / "scripts" / "README.md",
    PROJECT_ROOT / "web" / "README.md",
    PROJECT_ROOT / "apps",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "web",
]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".sh"}
LEGACY_IMPORT_PREFIXES = (
    "from src.arena",
    "from src.protocol",
    "from src.orchestrator",
    "from src.evaluator",
    "from src.renderer",
    "from src.telemetry",
    "from src.training",
    "from src.registry",
    "from src.coaching",
    "import src.arena",
    "import src.protocol",
    "import src.orchestrator",
    "import src.evaluator",
    "import src.renderer",
    "import src.telemetry",
    "import src.training",
    "import src.registry",
    "import src.coaching",
)
SKIP_DIR_NAMES = {"__pycache__", "original_vision", "analysis", "future_implementation"}



def _iter_surface_files():
    for path in CURRENT_SURFACE_PATHS:
        if path.is_file():
            yield path
            continue

        for child in path.rglob("*"):
            if not child.is_file():
                continue
            if child.suffix not in TEXT_SUFFIXES:
                continue
            if any(part in SKIP_DIR_NAMES for part in child.parts):
                continue
            yield child



def test_current_surfaces_prefer_src_atom_imports():
    offenders = []

    for file_path in _iter_surface_files():
        for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith(LEGACY_IMPORT_PREFIXES):
                offenders.append(f"{file_path.relative_to(PROJECT_ROOT)}:{line_number}: {stripped}")

    assert not offenders, "Found legacy imports in current surfaces:\n" + "\n".join(offenders)
