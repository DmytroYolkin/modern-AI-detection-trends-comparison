"""R-Detect: Deep Kernel Relative Test for Machine-generated Text Detection.

Paper:    https://openreview.net/forum?id=z9j7wctoGV  (Zhang et al., ICLR 2024)
Ref impl: https://github.com/xLearn-AU/R-Detect (vendored at
          ``test/third_party/r_detect/``)

Method
------
For a candidate text, the detector runs a non-parametric kernel relative
test in the latent space of a trained deep kernel ``net.pt``. It asks
whether the candidate's distribution is significantly closer to a sample of
human-written texts (HWT) or to a sample of machine-generated texts (MGT).
The test is repeated over ``rounds`` independent sub-samples and the
fraction of rounds in which the "closer to HWT" null is rejected is the
"rejection power" -- higher power -> more likely machine-generated.

Pretrained artefacts (loaded from the vendored repo as-is):

* ``net.pt`` -- the trained kernel network + ``sigma``, ``sigma0_u``, ``ep``
* the RoBERTa-base-openai-detector encoder (HuggingFace download on first use)

Reference samples
-----------------
The relative test needs reference feature tensors for HWT and MGT. On first
``load()`` the wrapper builds them from the project's own training split
(``data/dataset_ready_final/train.jsonl``) so R-Detect is calibrated against
the same distribution the in-house models trained on. The refs are cached
under ``data/baseline_artifacts/r_detect/`` and reused on subsequent runs.

Operational notes
-----------------
* The vendored repo loads ``./net.pt`` via a cwd-relative path. The wrapper
  ``chdir``'s into the vendored directory only for the duration of the
  one-time import/load phase, then restores cwd. ``predict()`` does not
  touch the filesystem so it runs in whatever cwd the caller is in.
* The vendored ``RobertaModelLoader`` hard-codes ``cache_dir=".cache"``,
  which fails on Windows under the vendored dir (HF Hub symlink handling).
  The wrapper monkey-patches it to use the default HF cache.
* The vendored ``meta_train.py`` imports ``pytorch_transformers`` (legacy).
  Install once: ``pip install pytorch_transformers``.
* The vendored ``utils.config`` is a module-level dict that ``NetLoader``
  reads at import time. The wrapper sets it before importing anything from
  the vendored package.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .base import BaselineDetector, DetectorResult

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REPO_DIR = _PROJECT_ROOT / "test" / "third_party" / "r_detect"
_REFS_DIR = _PROJECT_ROOT / "data" / "baseline_artifacts" / "r_detect"


class RDetectVendorMissing(RuntimeError):
    pass


class RDetect(BaselineDetector):
    name = "r_detect"
    requires = (
        "torch", "transformers", "nltk",
        "pytorch_transformers (pip install pytorch_transformers)",
        f"vendored repo at {_REPO_DIR.relative_to(_PROJECT_ROOT)}",
    )

    def __init__(
        self,
        device: str = "auto",
        threshold: float = 0.5,
        rounds: int = 20,
        n_ref_samples: int = 1000,
        feature_ref_HWT_path: str | None = None,
        feature_ref_MGT_path: str | None = None,
        train_jsonl: str = "data/dataset_ready_final/train.jsonl",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            device=device,
            threshold=threshold,
            rounds=rounds,
            n_ref_samples=n_ref_samples,
            feature_ref_HWT_path=feature_ref_HWT_path,
            feature_ref_MGT_path=feature_ref_MGT_path,
            train_jsonl=train_jsonl,
            **kwargs,
        )
        self._tester = None

    # ------------------------------------------------------------------ lifecycle
    def load(self) -> None:
        if self._loaded:
            return
        if not _REPO_DIR.exists():
            raise RDetectVendorMissing(
                f"R-Detect vendored repo missing at {_REPO_DIR}.\n"
                f"Clone it with:\n  git clone https://github.com/xLearn-AU/R-Detect.git \"{_REPO_DIR}\""
            )

        hwt_path = Path(
            self.config["feature_ref_HWT_path"] or _REFS_DIR / "feature_ref_HWT.pt"
        )
        mgt_path = Path(
            self.config["feature_ref_MGT_path"] or _REFS_DIR / "feature_ref_MGT.pt"
        )

        if str(_REPO_DIR) not in sys.path:
            sys.path.insert(0, str(_REPO_DIR))

        # chdir into the vendored dir only for the import/load phase -- the
        # vendored ``NetLoader`` reads ``./net.pt`` at module import. After
        # ``load()`` returns we restore cwd so the runner's relative output
        # paths still resolve correctly.
        original_cwd = os.getcwd()
        try:
            os.chdir(_REPO_DIR)
            # Set config FIRST -- both NetLoader (at meta_train import) and the
            # patched RobertaModelLoader read it via utils.get_device().
            import utils as r_utils
            r_utils.config["use_gpu"] = self._use_gpu()
            r_utils.config["local_model"] = ""
            r_utils.config["feature_ref_HWT"] = str(hwt_path)
            r_utils.config["feature_ref_MGT"] = str(mgt_path)
            self._patch_roberta_loader()

            if not hwt_path.exists() or not mgt_path.exists():
                self._build_refs(hwt_path, mgt_path)

            from relative_tester import RelativeTester
            from utils import init_random_seeds
            # Seed the relative test's subsample randomness so re-runs of the
            # comparison reproduce. Upstream calls this only in main.py.
            init_random_seeds()
            self._tester = RelativeTester()
        finally:
            os.chdir(original_cwd)
        super().load()

    @staticmethod
    def _patch_roberta_loader() -> None:
        """Patch the vendored ``RobertaModelLoader`` for cache + device.

        Two upstream issues:
        1. Hard-coded ``cache_dir=".cache"`` resolves to a path inside the
           vendored dir on which Windows HF Hub fails to create symlinks.
           Force the default HF cache instead.
        2. The loaded RoBERTa model stays on CPU even when ``use_gpu`` is set,
           so any subsequent ``tokens.to('cuda')`` raises a device-mismatch
           error. Move the model to ``utils.get_device()`` after loading.
        """
        import roberta_model_loader as _rml
        if getattr(_rml.RobertaModelLoader, "_rdetect_wrapper_patched", False):
            return
        original_init = _rml.RobertaModelLoader.__init__

        def patched_init(self, model_name="roberta-base-openai-detector", cache_dir=None):
            original_init(self, model_name=model_name, cache_dir=cache_dir)
            from utils import get_device
            self.model = self.model.to(get_device()).eval()

        _rml.RobertaModelLoader.__init__ = patched_init
        _rml.RobertaModelLoader._rdetect_wrapper_patched = True

    def predict(self, text: str) -> DetectorResult:
        if not self._loaded:
            self.load()
        power = self._test_numeric(text)
        if power is None:
            return DetectorResult(
                score_ai=0.5, label="human",
                raw={"power": None, "note": "input too short for relative test"},
            )
        score = max(0.0, min(1.0, float(power)))
        label = "ai" if score >= self.config["threshold"] else "human"
        return DetectorResult(
            score_ai=score, label=label,
            raw={"power": float(power), "rounds": self.config["rounds"]},
        )

    def close(self) -> None:
        super().close()

    # ------------------------------------------------------------------ internals
    def _use_gpu(self) -> bool:
        if self.config["device"] == "cpu":
            return False
        if self.config["device"].startswith("cuda"):
            return True
        # auto
        import torch
        return torch.cuda.is_available()

    def _test_numeric(self, text: str) -> float | None:
        """Replicates :py:meth:`RelativeTester.test` but returns the numeric power."""
        import torch
        from MMD import MMD_3_Sample_Test

        tester = self._tester
        sents = tester.sents_split(text)
        feats = tester.feature_extractor.process_sents(sents, False)
        if len(feats) <= 1:
            return None
        min_len = 2
        rounds = int(self.config["rounds"])
        net_loader = tester.feature_extractor.net  # NetLoader instance
        h_u_list: list[int] = []
        for _ in range(rounds):
            cand = feats[torch.randperm(len(feats))[:min_len]]
            hwt = tester.feature_hwt_ref[torch.randperm(len(tester.feature_hwt_ref))[:min_len]]
            mgt = tester.feature_mgt_ref[torch.randperm(len(tester.feature_mgt_ref))[:min_len]]
            h_u, _p, _t, *_ = MMD_3_Sample_Test(
                net_loader.net(cand),
                net_loader.net(hwt),
                net_loader.net(mgt),
                cand.view(cand.shape[0], -1),
                hwt.view(hwt.shape[0], -1),
                mgt.view(mgt.shape[0], -1),
                net_loader.sigma,
                net_loader.sigma0_u,
                net_loader.ep,
                0.05,
            )
            h_u_list.append(int(h_u))
        return sum(h_u_list) / rounds

    def _build_refs(self, hwt_path: Path, mgt_path: Path) -> None:
        """Extract HWT / MGT sentence-level feature refs from the train split.

        Mirrors the vendored ``feature_ref_generater.py`` but reads from the
        project's own ``train.jsonl`` instead of HC3 so the test is calibrated
        against the same distribution the in-house models trained on.
        """
        import torch
        import nltk

        train_path = _PROJECT_ROOT / self.config["train_jsonl"]
        if not train_path.exists():
            raise FileNotFoundError(
                f"Cannot build R-Detect feature refs: train split not found at "
                f"{train_path}. Build it with: python -m training.build_dataset --splits train"
            )

        # Vendored module setup -- chdir + config + patch already done by load().
        from roberta_model_loader import RobertaModelLoader
        from meta_train import net
        from utils import FeatureExtractor

        feature_extractor = FeatureExtractor(RobertaModelLoader(), net)

        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

        hwt_texts: list[str] = []
        mgt_texts: list[str] = []
        with train_path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                (hwt_texts if rec["label"] == "human" else mgt_texts).append(rec["text"])

        hwt_path.parent.mkdir(parents=True, exist_ok=True)
        for texts, out_path, tag in (
            (hwt_texts, hwt_path, "HWT"),
            (mgt_texts, mgt_path, "MGT"),
        ):
            sents: list[str] = []
            for txt in texts:
                for s in nltk.sent_tokenize(txt):
                    if len(s.split()) > 5:
                        sents.append(s)
                        if len(sents) >= self.config["n_ref_samples"]:
                            break
                if len(sents) >= self.config["n_ref_samples"]:
                    break
            sents = sents[: self.config["n_ref_samples"]]
            print(f"R-Detect: extracting {len(sents)} {tag} reference features -> {out_path}")
            feats = [feature_extractor.process(s, False).detach().cpu() for s in sents]
            torch.save(torch.cat(feats, dim=0), out_path)
