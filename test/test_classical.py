"""Unit tests for the classical-classifier track.

Synthetic features only -- no extractors, no neural net needed.

    python -m pytest test/test_classical.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from training.classical import (
    BACKENDS,
    ClassicalClassifier,
    block_importances,
    flatten_features,
)

NELA_DIM, STYLE_DIM, TRACE_DIM = 87, 10, 128
DIMS = {"nela": NELA_DIM, "style": STYLE_DIM, "trace": TRACE_DIM}


def _make_xy(n=200, seed=0):
    """Synthetic data with a learnable signal (label depends on a few features)."""
    rng = np.random.default_rng(seed)
    nela = rng.normal(0, 1, (n, NELA_DIM)).astype(np.float32)
    style = rng.normal(0, 1, (n, STYLE_DIM)).astype(np.float32)
    trace = rng.normal(0, 1, (n, TRACE_DIM)).astype(np.float32)
    y = (nela[:, 0] + style[:, 0] - trace[:, 0] > 0).astype(np.int64)
    return flatten_features(nela, style, trace), y


def test_flatten_features_width():
    X, _ = _make_xy(n=5)
    assert X.shape == (5, NELA_DIM + STYLE_DIM + TRACE_DIM)
    assert X.dtype == np.float32


def test_random_forest_fit_predict():
    """random_forest is always available (sklearn) -- fit, predict, learn signal."""
    X, y = _make_xy(n=300)
    clf = ClassicalClassifier("random_forest", n_estimators=50).fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == (300,)
    assert set(np.unique(pred)).issubset({0, 1})
    assert (pred == y).mean() > 0.8          # signal is learnable
    proba = clf.predict_proba(X)
    assert proba.shape == (300, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_feature_importances_and_blocks():
    X, y = _make_xy(n=300)
    clf = ClassicalClassifier("random_forest", n_estimators=50).fit(X, y)
    imp = clf.feature_importances()
    assert imp is not None and imp.shape == (NELA_DIM + STYLE_DIM + TRACE_DIM,)

    blocks = block_importances(imp, DIMS)
    assert set(blocks) == {"nela", "style", "trace"}
    assert abs(sum(blocks.values()) - 1.0) < 1e-6


def test_invalid_backend_rejected():
    with pytest.raises(ValueError):
        ClassicalClassifier("not_a_backend")


def test_available_backends_nonempty():
    avail = ClassicalClassifier.available_backends()
    assert avail                              # at least the sklearn ones
    assert all(b in BACKENDS for b in avail)


def test_checkpoint_roundtrip(tmp_path):
    X, y = _make_xy(n=200)
    clf = ClassicalClassifier("random_forest", n_estimators=50).fit(X, y)
    before = clf.predict(X)

    path = tmp_path / "clf.joblib"
    clf.save(path, dims=DIMS, metrics={"val": {"macro_f1": 0.9}})
    reloaded, payload = ClassicalClassifier.load(path)

    assert np.array_equal(before, reloaded.predict(X))
    assert payload["backend"] == "random_forest"
    assert payload["dims"] == DIMS
    assert payload["metrics"]["val"]["macro_f1"] == 0.9


def test_xgboost_fit_predict():
    """XGBoost path -- skipped automatically if the package is absent."""
    pytest.importorskip("xgboost")
    X, y = _make_xy(n=300)
    clf = ClassicalClassifier("xgboost", n_estimators=50).fit(X, y)
    assert (clf.predict(X) == y).mean() > 0.8
    assert clf.feature_importances().shape == (NELA_DIM + STYLE_DIM + TRACE_DIM,)


def test_mlp_fit_predict():
    """sklearn MLPClassifier backend -- fit, predict, learn the signal."""
    assert "mlp" in BACKENDS
    X, y = _make_xy(n=300)
    clf = ClassicalClassifier("mlp", n_estimators=120).fit(X, y)  # n_estimators -> epochs
    pred = clf.predict(X)
    assert pred.shape == (300,)
    assert set(np.unique(pred)).issubset({0, 1})
    assert (pred == y).mean() > 0.75              # signal is learnable
    proba = clf.predict_proba(X)
    assert proba.shape == (300, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    # an MLP exposes no per-feature importance
    assert clf.feature_importances() is None


def test_mlp_class_weighting_via_oversample():
    """MLP takes no sample_weight -> class_weighting must fall back cleanly."""
    rng = np.random.default_rng(1)
    # heavily imbalanced labels (mostly class 1)
    nela = rng.normal(0, 1, (200, NELA_DIM)).astype(np.float32)
    style = rng.normal(0, 1, (200, STYLE_DIM)).astype(np.float32)
    trace = rng.normal(0, 1, (200, TRACE_DIM)).astype(np.float32)
    X = flatten_features(nela, style, trace)
    y = np.array([0] * 20 + [1] * 180, dtype=np.int64)
    clf = ClassicalClassifier("mlp", n_estimators=60, class_weighting=True).fit(X, y)
    assert clf.predict(X).shape == (200,)
