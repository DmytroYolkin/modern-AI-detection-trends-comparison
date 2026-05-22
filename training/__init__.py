"""Training framework for the multi-extractor human-vs-AI text detector.

This package turns the three feature extractors (NELA, StyleDecipher, TRACE)
and the `MultiFeatureFusion` backbone into a complete, runnable training
pipeline:

    build_dataset.py    -> cache extractor features to data/features/*.npz
    feature_dataset.py  -> torch Dataset + feature normalisation over the cache
    model.py            -> FusionClassifier (fusion backbone + classifier head)
    train.py            -> training loop / CLI
    config.py           -> TrainConfig hyper-parameter container

Importing this package also bootstraps `sys.path` (see `paths`) so the repo's
`extractors/` and `fusion/` modules can be imported with their original
bare imports.
"""

from . import paths  # noqa: F401  -- side effect: extends sys.path

__all__ = ["paths"]
