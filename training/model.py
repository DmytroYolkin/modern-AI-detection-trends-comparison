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
    """`MultiFeatureFusion` backbone + an MLP classification head."""

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
    ) -> None:
        super().__init__()
        if fusion_method not in FUSION_METHODS:
            raise ValueError(
                f"fusion_method must be one of {FUSION_METHODS}, got {fusion_method!r}"
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
        )

    def forward(self, nela: torch.Tensor, style: torch.Tensor, trace: torch.Tensor) -> torch.Tensor:
        """Return raw class logits of shape ``(batch, num_classes)``."""
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
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, payload
