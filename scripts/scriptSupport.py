"""Shared helpers for command-line scripts."""

import sys
from pathlib import Path


def bootstrap_src_path() -> Path:
    """
    Add the local `src/` folder to Python's import path.

    Both command scripts need this same setup step, so we keep it
    in one helper instead of repeating the code in every script.
    """

    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    return project_root
