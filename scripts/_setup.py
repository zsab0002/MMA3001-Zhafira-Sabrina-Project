"""Make ``src/`` importable when scripts are run directly (``python scripts/xx.py``).

Installing the package with ``pip install -e .`` makes this unnecessary, but
the fallback keeps the workflow working either way.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
