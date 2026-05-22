"""Unit tests for the fusion backbone and `FusionClassifier`.

Dependency-light: only needs torch, so these always run.

    python -m pytest test/test_fusion_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

import training  # noqa: F401  -- bootstraps sys.path for the bare import below
from combination_all import MultiFeatureFusion
from training.config import FUSION_METHODS
from training.model import FusionClassifier

BATCH = 8
NELA_DIM, STYLE_DIM, TRACE_DIM = 87, 10, 128


def _mock_batch():
    return (
        torch.randn(BATCH, NELA_DIM),
        torch.randn(BATCH, STYLE_DIM),
        torch.randn(BATCH, TRACE_DIM),
    )


@pytest.mark.parametrize("method", FUSION_METHODS)
def test_fusion_backbone_output_shape(method):
    """Every fusion strategy produces a (batch, hidden_dim) embedding."""
    fusion = MultiFeatureFusion(NELA_DIM, STYLE_DIM, TRACE_DIM,
                                hidden_dim=64, fusion_method=method)
    out = fusion(*_mock_batch())
    assert out.shape == (BATCH, fusion.out_dim)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("method", FUSION_METHODS)
def test_classifier_forward_shape(method):
    """The classifier maps the feature triple to (batch, num_classes) logits."""
    model = FusionClassifier(NELA_DIM, STYLE_DIM, TRACE_DIM,
                             fusion_method=method, hidden_dim=64, head_hidden_dim=32)
    logits = model(*_mock_batch())
    assert logits.shape == (BATCH, 2)
    assert torch.isfinite(logits).all()


def test_predict_proba_is_distribution():
    model = FusionClassifier(NELA_DIM, STYLE_DIM, TRACE_DIM, hidden_dim=64)
    probs = model.predict_proba(*_mock_batch())
    assert probs.shape == (BATCH, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(BATCH), atol=1e-5)
    assert (probs >= 0).all()


def test_zero_style_features_stay_finite():
    """All-zero StyleDecipher vectors (uncovered samples) must not break the net."""
    model = FusionClassifier(NELA_DIM, STYLE_DIM, TRACE_DIM, hidden_dim=64)
    nela, _, trace = _mock_batch()
    style = torch.zeros(BATCH, STYLE_DIM)
    assert torch.isfinite(model(nela, style, trace)).all()


def test_backward_pass_produces_gradients():
    model = FusionClassifier(NELA_DIM, STYLE_DIM, TRACE_DIM, hidden_dim=64)
    logits = model(*_mock_batch())
    loss = torch.nn.functional.cross_entropy(logits, torch.randint(0, 2, (BATCH,)))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_invalid_fusion_method_rejected():
    with pytest.raises(ValueError):
        FusionClassifier(NELA_DIM, STYLE_DIM, TRACE_DIM, fusion_method="nonsense")


def test_checkpoint_roundtrip(tmp_path):
    """Saving then loading reproduces identical predictions."""
    model = FusionClassifier(NELA_DIM, STYLE_DIM, TRACE_DIM,
                             fusion_method="attention", hidden_dim=64, head_hidden_dim=32)
    model.eval()
    batch = _mock_batch()
    with torch.no_grad():
        before = model(*batch)

    path = tmp_path / "ckpt.pt"
    model.save(path, metrics={"val": {"macro_f1": 0.5}})
    reloaded, payload = FusionClassifier.load(path)
    with torch.no_grad():
        after = reloaded(*batch)

    assert torch.allclose(before, after, atol=1e-6)
    assert payload["hparams"]["fusion_method"] == "attention"
    assert payload["metrics"]["val"]["macro_f1"] == 0.5
