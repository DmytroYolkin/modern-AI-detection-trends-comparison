"""Tests and evaluation scripts for the AI-detection model.

- `evaluate.py`           -- score a trained checkpoint on the test split
- `test_fusion_model.py`  -- unit tests for the fusion backbone + classifier
- `test_feature_dataset.py` -- unit tests for the cached-feature dataset/normalizer
- `test_extractors.py`    -- smoke tests for the three feature extractors

Run the unit tests with:  python -m pytest test/
"""
