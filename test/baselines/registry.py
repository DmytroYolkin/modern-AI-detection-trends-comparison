"""Detector registry with lazy imports.

Each entry is a ``(module_path, class_name)`` tuple so the heavy dependencies
of a given detector (transformers, ollama, requests, ...) are only imported
when that detector is actually requested. This means
``compare_baselines.py --detectors radar`` will not try to import the GPTZero
HTTP client or the Binoculars 7B model loader.
"""

from __future__ import annotations

import importlib
from typing import Type

from .base import BaselineDetector

# name -> (module suffix, class name). Module suffix is relative to this package.
REGISTRY: dict[str, tuple[str, str]] = {
    "fast_detect_gpt": ("fast_detect_gpt", "FastDetectGPT"),
    "detect_gpt":      ("detect_gpt",      "DetectGPT"),
    "binoculars":      ("binoculars",      "Binoculars"),
    "r_detect":        ("r_detect",        "RDetect"),
    "radar":           ("radar",           "RADAR"),
    "raidar":          ("raidar",          "RAIDAR"),
}


def get_detector(name: str) -> Type[BaselineDetector]:
    """Resolve a registry name to its detector class (lazy-import on first hit)."""
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown detector {name!r}. Known: {sorted(REGISTRY)}"
        )
    module_suffix, class_name = REGISTRY[name]
    module = importlib.import_module(f".{module_suffix}", package=__package__)
    return getattr(module, class_name)
