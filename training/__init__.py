"""Training module for Atom Combat."""

import sys
from pathlib import Path

# Setup project root path for all training modules
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))