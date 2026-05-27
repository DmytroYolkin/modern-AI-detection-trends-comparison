"""External AI-text detectors used as baselines in the comparison study.

Each module wraps a published detector behind a uniform
:class:`BaselineDetector` interface so the runner in
``test/compare_baselines.py`` can score them all on the same test split as the
in-house models in ``models/ready_models/``.

The detectors split into two groups, all open-source / locally-runnable:

* **Paper-method** (Fast-DetectGPT, DetectGPT, Binoculars, RAIDAR, R-Detect)
  -- the detector is an *algorithm* applied on top of one or two
  general-purpose LMs (and, for R-Detect, a vendored pretrained deep
  kernel). The wrappers implement the method per the published paper /
  reference implementation; verify against the upstream repo before
  publishing.
* **Pretrained classifier** (RADAR) -- a HuggingFace sequence-classification
  checkpoint. The wrapper is a thin loader.

Commercial API detectors (GPTZero, Originality.ai, ZeroGPT) were removed:
GPTZero and Originality are paid per-text and the per-run cost relative to
the value of one extra data point isn't justifiable; ZeroGPT's advertised
"free tier" applies only to its web demo -- the API requires purchased
credits.

The interface and the registry are the only things ``test/compare_baselines.py``
imports; the actual detector classes are lazy-imported on demand so a missing
heavy dependency only breaks the detector that needs it.
"""

from .base import BaselineDetector, DetectorResult
from .registry import REGISTRY, get_detector

__all__ = ["BaselineDetector", "DetectorResult", "REGISTRY", "get_detector"]
