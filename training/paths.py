"""Filesystem paths and `sys.path` bootstrap for the training framework.

The repo's extractor / fusion modules use bare imports
(`from nela_extractor import ...`, `from combination_all import ...`) that only
resolve when their directories are on `sys.path`. Importing this module adds
them, alongside the repo root (needed for `from data.preprocessing import ...`).
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- key directories -------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
EXTRACTORS_DIR = REPO_ROOT / "extractors"
FUSION_DIR = REPO_ROOT / "fusion"

DATASET_DIR = REPO_ROOT / "data" / "dataset_ready_final"
FEATURE_DIR = REPO_ROOT / "data" / "features"

MODELS_DIR = REPO_ROOT / "models"
READY_MODELS_DIR = MODELS_DIR / "ready_models"   # committed, validated models
TEST_MODELS_DIR = MODELS_DIR / "test_models"     # gitignored, experimental runs

# --- sys.path bootstrap ----------------------------------------------------
for _path in (REPO_ROOT, EXTRACTORS_DIR, FUSION_DIR):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


def resolve_device(device: str = "auto") -> str:
    """Map ``"auto"`` to ``"cuda"`` when a GPU is available, else ``"cpu"``."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # torch missing / broken -> safe default
        return "cpu"
