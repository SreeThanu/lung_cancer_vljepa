"""
Path resolution utilities for the lung_cancer_vljepa project.

Usage:
    from src.utils.paths import find_repo_root, get_data_dir
    repo = find_repo_root()
    data = get_data_dir(repo)

Set LUNG_DATA_DIR env var to override the default data directory.
This is required when data and code live at different paths (e.g., Mac T7 SSD
vs. lab machine where data is stored under <repo_root>/data/).
"""

import os
from pathlib import Path


def find_repo_root() -> Path:
    """
    Walk up from this file until finding a directory that contains both src/
    and configs/ subdirectories. That is the repository root.

    Falls back to CWD search if the file-based walk fails (e.g., interactive
    sessions, notebooks).
    """
    # Walk up from this source file
    candidate = Path(__file__).resolve().parent
    while candidate != candidate.parent:
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate
        candidate = candidate.parent

    # Fall back: walk up from current working directory
    candidate = Path.cwd()
    while candidate != candidate.parent:
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate
        candidate = candidate.parent

    raise RuntimeError(
        "Cannot find repo root. Expected a directory containing both src/ and "
        "configs/. Set LUNG_DATA_DIR to point directly to the data directory if "
        "the repo layout differs from the default."
    )


def get_data_dir(repo_root: Path) -> Path:
    """
    Return the processed data directory root.

    Priority:
      1. LUNG_DATA_DIR environment variable (absolute path to data root)
      2. <repo_root>/data  (standard layout on lab machine)
    """
    env = os.environ.get("LUNG_DATA_DIR")
    if env:
        p = Path(env)
        if not p.exists():
            raise RuntimeError(
                f"LUNG_DATA_DIR={env!r} does not exist. "
                "Check the environment variable."
            )
        return p
    return repo_root / "data"
