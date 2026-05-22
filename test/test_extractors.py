"""Smoke tests for the three feature extractors via `FeaturePipeline`.

These exercise real models, so some are gated:

  * NELA  -- light (small NLTK data download); runs by default.
  * TRACE / StyleDecipher embedding model -- download ~400MB transformer
    weights, so they only run when RUN_EXTRACTOR_TESTS=1 is set.

    python -m pytest test/test_extractors.py                 # NELA only
    RUN_EXTRACTOR_TESTS=1 python -m pytest test/test_extractors.py   # all
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from training.extractor_pipeline import (
    STYLE_DIM,
    TRACE_DIM,
    FeaturePipeline,
    build_rewrite_clusters,
)

HEAVY = os.environ.get("RUN_EXTRACTOR_TESTS") == "1"
heavy_only = pytest.mark.skipif(
    not HEAVY, reason="set RUN_EXTRACTOR_TESTS=1 to run heavy extractor tests")

SAMPLE_TEXT = (
    "Artificial intelligence is rapidly transforming various industries. "
    "From healthcare to finance, machine learning models are becoming more "
    "sophisticated, but this also raises real ethical and societal challenges."
)


@dataclass
class FakeSample:
    """Minimal stand-in for `TextSample` (only the attrs the pipeline reads)."""

    record_id: str
    text: str
    is_rewrite: bool = False
    source_text_id: str | None = None


def test_nela_extraction_dim():
    """NELA returns the fixed 87-dim feature vector."""
    pytest.importorskip("nela_features")
    pipeline = FeaturePipeline(styledecipher_mode="off")
    vec = pipeline.nela_features(SAMPLE_TEXT)
    assert vec.shape == (FeaturePipeline.NELA_DIM,)
    assert vec.dtype.name == "float32"
    import numpy as np

    assert np.isfinite(vec).all()


def test_styledecipher_off_mode_returns_zero_vector():
    """`off` mode emits a zero StyleDecipher vector without loading any model."""
    pipeline = FeaturePipeline(styledecipher_mode="off")
    vec, ok = pipeline.style_features(FakeSample("r_0001", SAMPLE_TEXT))
    assert vec.shape == (STYLE_DIM,)
    assert not ok
    assert (vec == 0).all()


def test_rewrite_cluster_grouping():
    """A human text and its rewrites land in one cluster keyed by the source id."""
    human = FakeSample("h_0001", "original text")
    rw1 = FakeSample("r_0001", "rewrite one", is_rewrite=True, source_text_id="h_0001")
    rw2 = FakeSample("r_0002", "rewrite two", is_rewrite=True, source_text_id="h_0001")
    clusters, member_to_cluster = build_rewrite_clusters([[human, rw1, rw2]])

    assert set(clusters["h_0001"]) == {"h_0001", "r_0001", "r_0002"}
    assert member_to_cluster["r_0001"] == "h_0001"
    assert member_to_cluster["h_0001"] == "h_0001"


@heavy_only
def test_styledecipher_cached_mode_produces_features():
    """`cached` mode computes real similarity features against cluster siblings."""
    human = FakeSample("h_0001", SAMPLE_TEXT)
    rw = FakeSample("r_0001", SAMPLE_TEXT + " It is a transformative era.",
                    is_rewrite=True, source_text_id="h_0001")
    pipeline = FeaturePipeline.from_datasets([[human, rw]], styledecipher_mode="cached")

    vec, ok = pipeline.style_features(human)
    assert ok
    assert vec.shape == (STYLE_DIM,)


@heavy_only
def test_trace_embedding_dim():
    """TRACE returns the fixed 128-dim author-style embedding."""
    pipeline = FeaturePipeline(styledecipher_mode="off")
    vec = pipeline.trace_features(FakeSample("h_0001", SAMPLE_TEXT))
    assert vec.shape == (TRACE_DIM,)


@heavy_only
def test_full_pipeline_extract():
    """End-to-end: one sample -> finite (nela, style, trace) triple."""
    import numpy as np

    human = FakeSample("h_0001", SAMPLE_TEXT)
    rw = FakeSample("r_0001", SAMPLE_TEXT + " A new era indeed.",
                    is_rewrite=True, source_text_id="h_0001")
    pipeline = FeaturePipeline.from_datasets([[human, rw]], styledecipher_mode="cached")

    feats = pipeline.extract(human)
    assert feats.nela.shape == (FeaturePipeline.NELA_DIM,)
    assert feats.style.shape == (STYLE_DIM,)
    assert feats.trace.shape == (TRACE_DIM,)
    for arr in (feats.nela, feats.style, feats.trace):
        assert np.isfinite(arr).all()
