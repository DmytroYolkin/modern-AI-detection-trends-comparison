"""Shared interface for every baseline detector wrapper.

A detector implementation must:

1. Subclass :class:`BaselineDetector`.
2. Set the class-level ``name`` attribute (used as the registry key and as the
   stem of the output ``<name>.metrics.json`` file).
3. Override :meth:`load` for any heavy initialisation (loading weights, opening
   an HTTP session). The runner calls ``load()`` exactly once before the first
   ``predict()`` so the cost is excluded from per-sample latency.
4. Override :meth:`predict` to return a :class:`DetectorResult` with
   ``score_ai`` in ``[0, 1]`` and a hard ``label`` in ``{"human", "ai"}``.

Detectors that have native batch APIs (HuggingFace classifiers, the commercial
APIs that accept arrays) should also override :meth:`predict_batch` for speed;
the default implementation falls back to a serial loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator


@dataclass
class DetectorResult:
    """One detector's verdict on one text.

    Attributes
    ----------
    score_ai
        Probability (or pseudo-probability) that the input is AI-generated,
        normalised to ``[0, 1]``. Used downstream for ROC-AUC and the strict-FPR
        threshold scan. If the detector only returns a hard label, set this to
        ``1.0`` for ``"ai"`` and ``0.0`` for ``"human"``.
    label
        Hard classification at the detector's own default threshold. Used for
        accuracy / macro-F1 at the default operating point.
    raw
        The unmodified response payload (HTTP JSON, model logits, perturbation
        scores). Persisted alongside the aggregate metrics so a run can be
        re-thresholded later without rerunning the detector.
    """

    score_ai: float
    label: str
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score_ai <= 1.0:
            raise ValueError(f"score_ai must be in [0, 1], got {self.score_ai}")
        if self.label not in ("human", "ai"):
            raise ValueError(f"label must be 'human' or 'ai', got {self.label!r}")


class BaselineDetector:
    """Abstract base for every detector wrapper."""

    #: registry key + output filename stem.
    name: str = "base"

    #: human-readable list of pip packages / env vars / external services the
    #: wrapper needs in order to run. Surfaced by ``compare_baselines.py
    #: --list`` so the user can see what to install before launching.
    requires: tuple[str, ...] = ()

    def __init__(self, **kwargs: Any) -> None:
        self.config: dict[str, Any] = dict(kwargs)
        self._loaded = False

    # ------------------------------------------------------------------ lifecycle
    def load(self) -> None:
        """Heavy initialisation hook. Idempotent.

        Default no-op; override to load weights, open an HTTP session, etc.
        """
        self._loaded = True

    def close(self) -> None:
        """Release resources (HTTP sessions, GPU memory). Default no-op."""
        self._loaded = False

    # ------------------------------------------------------------------ prediction
    def predict(self, text: str) -> DetectorResult:
        raise NotImplementedError

    def predict_batch(self, texts: Iterable[str]) -> Iterator[DetectorResult]:
        """Default serial batch loop. Override for native batching."""
        for text in texts:
            yield self.predict(text)

    # ------------------------------------------------------------------ describe
    def describe(self) -> dict[str, Any]:
        """Compact dict written into each output ``metrics.json``.

        Override to add detector-specific fields (the scoring model, the
        rewriter LM, the number of perturbations, the API endpoint, ...).
        """
        return {"detector": self.name, "config": dict(self.config)}
