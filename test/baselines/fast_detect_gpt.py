"""Fast-DetectGPT (Bao, Zhao, Teng, Yang, Wan 2024).

Paper:  https://arxiv.org/abs/2310.05130
Ref impl: https://github.com/baoguangsheng/fast-detect-gpt

The detector replaces DetectGPT's expensive perturbation step with a single
forward pass. For a candidate text ``x`` with token-level log-probabilities
``log p_theta(x_i | x_<i)`` under a *scoring* model and ``log p_phi(x_i | x_<i)``
under a *reference* model, the conditional-probability-curvature score is:

    s(x) = mean_i [ log p_theta(x_i | x_<i) ]
         - mean_i [ log p_phi  (x_i | x_<i) ]   (normalised by std)

Higher s -> more likely AI-generated. The hard label is produced by
thresholding s against a calibration value (the paper uses the same model as
both scorer and reference and reports a fixed threshold around zero on the
discrepancy score; this wrapper exposes ``threshold`` so it can be set per
domain).

Defaults are GPT-2 / GPT-2 for both roles -- small enough to run on CPU for a
smoke test. For published-quality results swap in the same scorer the paper
uses for your generator family (see the reference repo's ``--scoring_model``).
"""

from __future__ import annotations

import math
from typing import Any

from .base import BaselineDetector, DetectorResult


class FastDetectGPT(BaselineDetector):
    name = "fast_detect_gpt"
    requires = ("torch", "transformers")

    def __init__(
        self,
        scoring_model: str = "gpt2",
        reference_model: str = "gpt2",
        device: str = "auto",
        max_length: int = 512,
        threshold: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            scoring_model=scoring_model,
            reference_model=reference_model,
            device=device,
            max_length=max_length,
            threshold=threshold,
            **kwargs,
        )
        self._scorer = None
        self._scorer_tok = None
        self._ref = None
        self._ref_tok = None
        self._device = device

    def load(self) -> None:
        if self._loaded:
            return
        import torch  # noqa: F401  (validate the dep)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = _resolve_device(self.config["device"])

        self._scorer_tok = AutoTokenizer.from_pretrained(self.config["scoring_model"])
        self._scorer = (
            AutoModelForCausalLM.from_pretrained(self.config["scoring_model"])
            .to(self._device)
            .eval()
        )
        if self.config["reference_model"] == self.config["scoring_model"]:
            self._ref_tok = self._scorer_tok
            self._ref = self._scorer
        else:
            self._ref_tok = AutoTokenizer.from_pretrained(self.config["reference_model"])
            self._ref = (
                AutoModelForCausalLM.from_pretrained(self.config["reference_model"])
                .to(self._device)
                .eval()
            )
        super().load()

    def predict(self, text: str) -> DetectorResult:
        if not self._loaded:
            self.load()
        scorer_ll = _token_logprobs(self._scorer, self._scorer_tok, text,
                                    self.config["max_length"], self._device)
        ref_ll = _token_logprobs(self._ref, self._ref_tok, text,
                                 self.config["max_length"], self._device)
        # Discrepancy score: scoring-model log-likelihood minus reference-model
        # log-likelihood, normalised by the reference-model std. AI text scores
        # higher because the scoring model assigns it higher likelihood than the
        # reference model does (see paper eq. 3).
        diff = scorer_ll["mean"] - ref_ll["mean"]
        denom = ref_ll["std"] if ref_ll["std"] > 1e-8 else 1.0
        score = diff / denom
        prob_ai = 1.0 / (1.0 + math.exp(-score))  # squash to [0, 1] for ROC-AUC
        label = "ai" if score >= self.config["threshold"] else "human"
        return DetectorResult(
            score_ai=prob_ai,
            label=label,
            raw={
                "discrepancy": float(score),
                "scorer_ll_mean": scorer_ll["mean"],
                "ref_ll_mean": ref_ll["mean"],
                "ref_ll_std": ref_ll["std"],
            },
        )


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _token_logprobs(model, tokenizer, text: str, max_length: int, device: str) -> dict[str, float]:
    """Mean and std of per-token log-likelihoods under ``model``."""
    import torch

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[:, :-1, :]
        targets = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        gather = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
    return {"mean": float(gather.mean().item()), "std": float(gather.std().item())}
