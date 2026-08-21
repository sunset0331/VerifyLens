# conftest.py
# -----------
# Pytest configuration for VerifyLens.
# Adds the repo root to sys.path so that `src.*` imports work
# regardless of where pytest is invoked from.

import sys
from pathlib import Path

# Insert the repo root into sys.path so that
# `from src.document.schema import ...` resolves correctly.
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
