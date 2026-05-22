"""Unit tests for the cached-feature dataset and normaliser.

Builds a tiny synthetic `.npz` cache on the fly, so no extractors are needed.

    python -m pytest test/test_feature_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from training.extractor_pipeline import edit_similarity, ngram_overlap
from training.feature_dataset import FeatureNormalizer, FusionFeatureDataset

NELA_DIM, STYLE_DIM, TRACE_DIM = 87, 10, 128


def _make_npz(path: Path, n: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    np.savez_compressed(
        path,
        nela=rng.normal(5.0, 3.0, size=(n, NELA_DIM)).astype(np.float32),
        style=rng.random((n, STYLE_DIM)).astype(np.float32),
        trace=rng.normal(0.0, 1.0, size=(n, TRACE_DIM)).astype(np.float32),
        label=rng.integers(0, 2, size=n).astype(np.int64),
        style_ok=rng.integers(0, 2, size=n).astype(bool),
        ids=np.array([f"r_{i:04d}" for i in range(n)], dtype=object),
        sources=np.array(["use"] * n, dtype=object),
    )


def test_dataset_loads_and_yields_tensors(tmp_path):
    npz = tmp_path / "train.npz"
    _make_npz(npz, n=12)
    ds = FusionFeatureDataset(npz)

    assert len(ds) == 12
    assert ds.dims == {"nela": NELA_DIM, "style": STYLE_DIM, "trace": TRACE_DIM}

    nela, style, trace, label = ds[0]
    assert nela.shape == (NELA_DIM,) and style.shape == (STYLE_DIM,)
    assert trace.shape == (TRACE_DIM,)
    assert isinstance(label.item(), int)
    assert label.dtype == torch.long


def test_class_counts_sum_to_length(tmp_path):
    npz = tmp_path / "train.npz"
    _make_npz(npz, n=30)
    ds = FusionFeatureDataset(npz)
    assert sum(ds.class_counts().values()) == len(ds)
    assert 0.0 <= ds.style_coverage() <= 1.0


def test_normalizer_standardises_train_features(tmp_path):
    npz = tmp_path / "train.npz"
    _make_npz(npz, n=200)
    ds = FusionFeatureDataset(npz)

    normalizer = FeatureNormalizer.fit(ds.raw_arrays())
    ds.apply_normalizer(normalizer)

    # after standardisation the train features are ~zero-mean, ~unit-std
    assert np.allclose(ds.nela.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(ds.nela.std(axis=0), 1.0, atol=1e-3)
    # raw arrays must remain untouched
    assert not np.allclose(ds.raw_arrays()["nela"].mean(axis=0), 0.0, atol=1e-4)


def test_normalizer_state_dict_roundtrip(tmp_path):
    npz = tmp_path / "train.npz"
    _make_npz(npz, n=50)
    ds = FusionFeatureDataset(npz)

    normalizer = FeatureNormalizer.fit(ds.raw_arrays())
    restored = FeatureNormalizer.from_state_dict(normalizer.state_dict())

    sample = ds.raw_arrays()["trace"]
    assert np.allclose(
        normalizer.transform("trace", sample),
        restored.transform("trace", sample),
        atol=1e-6,
    )
    assert FeatureNormalizer.from_state_dict(None) is None


def test_styledecipher_similarity_helpers():
    """The offline StyleDecipher similarity primitives behave sensibly."""
    text = "the quick brown fox jumps over the lazy dog"
    assert ngram_overlap(text, text, 1) == 1.0
    assert ngram_overlap(text, text, 2) == 1.0
    assert ngram_overlap("a b c", "x y z", 1) == 0.0
    assert edit_similarity(text, text) == 1.0
    assert 0.0 <= edit_similarity(text, "completely different string") < 1.0
