"""Fusion-based human-vs-AI text classifier.

`FusionClassifier` wraps the repo's `MultiFeatureFusion` backbone
(`fusion/combination_all.py`) with a classifier head. The backbone fuses the
three extractor feature vectors into one embedding; the head maps that
embedding to class logits.

`combination_all.py` is intentionally left unchanged -- it is a clean,
reusable feature backbone. The only thing it lacked for training is a
prediction head, which lives here so the backbone stays task-agnostic.
"""

from __future__ import annotations

from typing import Sequence

from . import paths  # noqa: F401  -- must precede the bare `combination_all` import

import torch
import torch.nn as nn

from combination_all import MultiFeatureFusion

from .config import FUSION_METHODS


class FusionClassifier(nn.Module):
    """`MultiFeatureFusion` backbone + an MLP classification head.

    Modality dropout
    ----------------
    Each forward pass in training mode independently zeroes each feature block
    (NELA / Style / TRACE) with the per-modality probability stored in
    ``self.modality_dropout``. Inverted-dropout scaling preserves the expected
    activation magnitude so evaluation (no dropout) sees the same scale.

    The rates are intentionally **asymmetric**: NELA is dropped at a higher
    probability than Style or TRACE so the optimiser cannot collapse onto
    NELA's surface features alone. This is the standard "modality dropout"
    technique from multimodal learning (Neverova et al., "ModDrop", TPAMI 2016;
    Hussen Abdelaziz et al. 2020).

    Setting all three rates to 0 disables the regularizer (the default in
    `TrainConfig` is enabled, but ablation runs can pass zeros).
    """

    def __init__(
        self,
        nela_dim: int,
        style_dim: int,
        trace_dim: int,
        *,
        fusion_method: str = "gating",
        hidden_dim: int = 256,
        head_hidden_dim: int = 128,
        dropout: float = 0.3,
        num_classes: int = 2,
        modality_dropout_nela: float = 0.0,
        modality_dropout_style: float = 0.0,
        modality_dropout_trace: float = 0.0,
    ) -> None:
        super().__init__()
        if fusion_method not in FUSION_METHODS:
            raise ValueError(
                f"fusion_method must be one of {FUSION_METHODS}, got {fusion_method!r}"
            )
        for name, p in (("nela", modality_dropout_nela),
                        ("style", modality_dropout_style),
                        ("trace", modality_dropout_trace)):
            if not 0.0 <= p < 1.0:
                raise ValueError(
                    f"modality_dropout_{name} must be in [0, 1), got {p}"
                )

        self.fusion = MultiFeatureFusion(
            nela_dim=nela_dim,
            style_dim=style_dim,
            trace_dim=trace_dim,
            hidden_dim=hidden_dim,
            fusion_method=fusion_method,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.fusion.out_dim, head_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(head_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

        self.modality_dropout = {
            "nela":  float(modality_dropout_nela),
            "style": float(modality_dropout_style),
            "trace": float(modality_dropout_trace),
        }

        # Stored so a checkpoint can rebuild the exact architecture.
        self.hparams = dict(
            nela_dim=nela_dim,
            style_dim=style_dim,
            trace_dim=trace_dim,
            fusion_method=fusion_method,
            hidden_dim=hidden_dim,
            head_hidden_dim=head_hidden_dim,
            dropout=dropout,
            num_classes=num_classes,
            modality_dropout_nela=float(modality_dropout_nela),
            modality_dropout_style=float(modality_dropout_style),
            modality_dropout_trace=float(modality_dropout_trace),
        )

    def _apply_modality_dropout(self, x: torch.Tensor, p: float) -> torch.Tensor:
        """Per-sample, whole-block Bernoulli mask with inverted-dropout scaling.

        Returns ``x`` unchanged when not training or when ``p`` is 0.
        """
        if not self.training or p <= 0.0:
            return x
        # One mask value per row (broadcasts over the feature dim).
        keep = torch.bernoulli(
            torch.full((x.size(0), 1), 1.0 - p, device=x.device, dtype=x.dtype)
        )
        return x * keep / (1.0 - p)

    def forward(self, nela: torch.Tensor, style: torch.Tensor, trace: torch.Tensor) -> torch.Tensor:
        """Return raw class logits of shape ``(batch, num_classes)``."""
        nela  = self._apply_modality_dropout(nela,  self.modality_dropout["nela"])
        style = self._apply_modality_dropout(style, self.modality_dropout["style"])
        trace = self._apply_modality_dropout(trace, self.modality_dropout["trace"])
        return self.head(self.fusion(nela, style, trace))

    @torch.no_grad()
    def predict_proba(self, nela: torch.Tensor, style: torch.Tensor, trace: torch.Tensor) -> torch.Tensor:
        """Class probabilities (softmax over logits)."""
        was_training = self.training
        self.eval()
        probs = torch.softmax(self.forward(nela, style, trace), dim=-1)
        self.train(was_training)
        return probs

    # ---- persistence ------------------------------------------------------

    def save(
        self,
        path,
        *,
        normalizer=None,
        metrics: dict | None = None,
        train_config: dict | None = None,
        label_names: Sequence[str] = ("human", "ai"),
    ) -> None:
        """Save weights + everything needed to reload and run the model."""
        payload = {
            "state_dict": self.state_dict(),
            "hparams": self.hparams,
            "normalizer": normalizer.state_dict() if normalizer is not None else None,
            "metrics": metrics or {},
            "train_config": train_config or {},
            "label_names": list(label_names),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> tuple["FusionClassifier", dict]:
        """Rebuild a model from a checkpoint. Returns ``(model, payload)``."""
        payload = torch.load(path, map_location=map_location, weights_only=False)
        hp = payload["hparams"]
        model = cls(
            hp["nela_dim"],
            hp["style_dim"],
            hp["trace_dim"],
            fusion_method=hp["fusion_method"],
            hidden_dim=hp["hidden_dim"],
            head_hidden_dim=hp["head_hidden_dim"],
            dropout=hp["dropout"],
            num_classes=hp["num_classes"],
            # Older checkpoints predate modality dropout -- default to 0 so the
            # rebuilt model is bit-for-bit identical at inference time.
            modality_dropout_nela=hp.get("modality_dropout_nela", 0.0),
            modality_dropout_style=hp.get("modality_dropout_style", 0.0),
            modality_dropout_trace=hp.get("modality_dropout_trace", 0.0),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, payload
