"""Torch dataset + feature normalisation over the cached `.npz` feature files.

`build_dataset.py` writes one `data/features/<split>.npz` per split.
`FusionFeatureDataset` loads such a file for training/evaluation;
`FeatureNormalizer` standardises features (fit on train, reused everywhere).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

MODALITIES = ("nela", "style", "trace")


class FeatureNormalizer:
    """Per-feature standardisation (zero mean, unit variance) per modality.

    Fit once on the training split, then reused (saved into the checkpoint) so
    val/test/inference all see the exact same transform.
    """

    def __init__(self, stats: dict | None = None) -> None:
        # stats: modality -> (mean: np.ndarray, std: np.ndarray)
        self.stats = stats or {}

    @classmethod
    def fit(cls, arrays: dict) -> "FeatureNormalizer":
        stats = {}
        for m in MODALITIES:
            a = np.asarray(arrays[m], dtype=np.float64)
            mean = a.mean(axis=0)
            std = a.std(axis=0)
            std[std < 1e-6] = 1.0          # guard constant / all-zero features
            stats[m] = (mean.astype(np.float32), std.astype(np.float32))
        return cls(stats)

    def transform(self, modality: str, arr: np.ndarray) -> np.ndarray:
        if modality not in self.stats:
            return np.asarray(arr, dtype=np.float32)
        mean, std = self.stats[modality]
        return ((np.asarray(arr, dtype=np.float32) - mean) / std).astype(np.float32)

    def state_dict(self) -> dict:
        return {
            m: {"mean": mean.tolist(), "std": std.tolist()}
            for m, (mean, std) in self.stats.items()
        }

    @classmethod
    def from_state_dict(cls, sd: dict | None) -> "FeatureNormalizer | None":
        if not sd:
            return None
        stats = {
            m: (np.asarray(v["mean"], np.float32), np.asarray(v["std"], np.float32))
            for m, v in sd.items()
        }
        return cls(stats)


class FusionFeatureDataset(TorchDataset):
    """Loads one cached split; yields ``(nela, style, trace, label)`` tensors."""

    def __init__(self, npz_path, normalizer: FeatureNormalizer | None = None) -> None:
        data = np.load(npz_path, allow_pickle=True)
        self._raw = {
            "nela": data["nela"].astype(np.float32),
            "style": data["style"].astype(np.float32),
            "trace": data["trace"].astype(np.float32),
        }
        self.labels = data["label"].astype(np.int64)
        n = len(self.labels)
        self.ids = data["ids"] if "ids" in data.files else np.arange(n)
        self.sources = (
            data["sources"] if "sources" in data.files
            else np.array(["?"] * n, dtype=object)
        )
        self.style_ok = (
            data["style_ok"].astype(bool) if "style_ok" in data.files
            else np.zeros(n, dtype=bool)
        )

        # working arrays (normalised when a normalizer is attached)
        self.normalizer: FeatureNormalizer | None = None
        self.nela = self._raw["nela"]
        self.style = self._raw["style"]
        self.trace = self._raw["trace"]
        if normalizer is not None:
            self.apply_normalizer(normalizer)

    # ---- normalisation ---------------------------------------------------

    def apply_normalizer(self, normalizer: FeatureNormalizer) -> "FusionFeatureDataset":
        self.normalizer = normalizer
        self.nela = normalizer.transform("nela", self._raw["nela"])
        self.style = normalizer.transform("style", self._raw["style"])
        self.trace = normalizer.transform("trace", self._raw["trace"])
        return self

    def raw_arrays(self) -> dict:
        """Un-normalised feature arrays (use this to *fit* a normalizer)."""
        return self._raw

    # ---- introspection ---------------------------------------------------

    @property
    def dims(self) -> dict:
        return {
            "nela": self._raw["nela"].shape[1],
            "style": self._raw["style"].shape[1],
            "trace": self._raw["trace"].shape[1],
        }

    def class_counts(self) -> dict:
        return {int(c): int((self.labels == c).sum()) for c in np.unique(self.labels)}

    def style_coverage(self) -> float:
        return float(self.style_ok.mean()) if len(self.style_ok) else 0.0

    # ---- torch Dataset protocol ------------------------------------------

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.nela[idx]),
            torch.from_numpy(self.style[idx]),
            torch.from_numpy(self.trace[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
