"""Hyper-parameter container for a training run."""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Fusion strategies supported by `fusion/combination_all.py::MultiFeatureFusion`.
FUSION_METHODS = ("concat", "mlp", "attention", "gating")


@dataclass
class TrainConfig:
    """Everything needed to reproduce a single training run.

    The defaults are a sensible starting point for the ~13k-record dataset in
    `data/dataset_ready_final/`.
    """

    # --- model -------------------------------------------------------------
    fusion_method: str = "gating"        # concat | mlp | attention | gating
    hidden_dim: int = 256                # MultiFeatureFusion projection width
    head_hidden_dim: int = 128           # classifier-head hidden width
    dropout: float = 0.3
    num_classes: int = 2                 # human (0) vs ai (1)

    # --- optimisation ------------------------------------------------------
    epochs: int = 40
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_patience: int = 3                 # ReduceLROnPlateau patience
    early_stopping_patience: int = 8     # epochs without val-F1 gain -> stop

    # --- data --------------------------------------------------------------
    class_weighting: bool = True         # weight CE loss by inverse class freq
    normalize_features: bool = True      # standardise features (fit on train)

    # --- misc --------------------------------------------------------------
    seed: int = 42
    device: str = "auto"                 # auto | cpu | cuda

    def __post_init__(self) -> None:
        if self.fusion_method not in FUSION_METHODS:
            raise ValueError(
                f"fusion_method must be one of {FUSION_METHODS}, "
                f"got {self.fusion_method!r}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_args(cls, args) -> "TrainConfig":
        """Build a config from an `argparse.Namespace` (unknown attrs ignored)."""
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in vars(args).items() if k in fields and v is not None})
