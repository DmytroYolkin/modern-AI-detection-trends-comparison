"""DetectGPT (Mitchell, Lee, Khazatsky, Manning, Finn 2023).

Paper:  https://arxiv.org/abs/2301.11305
Ref impl: https://github.com/eric-mitchell/detect-gpt

For input text ``x`` and N perturbations ``x_tilde_1..N`` produced by masking
random spans and re-filling them with a mask-filling LM (T5), the score is:

    s(x) = log p(x) - mean_i log p(x_tilde_i)

A *positive* and *large* s means the candidate sits at a local maximum of the
scoring LM's log-probability surface -- i.e. perturbing it almost always lowers
the likelihood -- which is the published characterisation of LLM-generated
text. Hard label thresholds s against 0 by default.

This is the slowest detector in the suite (N forward passes through the mask
filler + N+1 forward passes through the scorer per sample). Defaults to N=10
which is at the low end of what the paper reports; raise ``n_perturbations``
for higher fidelity at the cost of linear-time slowdown.
"""

from __future__ import annotations

import math
import random
import re
from typing import Any

from .base import BaselineDetector, DetectorResult


class DetectGPT(BaselineDetector):
    name = "detect_gpt"
    requires = ("torch", "transformers")

    def __init__(
        self,
        scoring_model: str = "gpt2",
        mask_filling_model: str = "t5-base",
        n_perturbations: int = 10,
        mask_fraction: float = 0.15,
        span_length: int = 2,
        device: str = "auto",
        max_length: int = 512,
        threshold: float = 0.0,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            scoring_model=scoring_model,
            mask_filling_model=mask_filling_model,
            n_perturbations=n_perturbations,
            mask_fraction=mask_fraction,
            span_length=span_length,
            device=device,
            max_length=max_length,
            threshold=threshold,
            seed=seed,
            **kwargs,
        )
        self._scorer = None
        self._scorer_tok = None
        self._mask = None
        self._mask_tok = None
        self._device = device
        self._rng = random.Random(seed)

    def load(self) -> None:
        if self._loaded:
            return
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )
        from .fast_detect_gpt import _resolve_device

        self._device = _resolve_device(self.config["device"])
        self._scorer_tok = AutoTokenizer.from_pretrained(self.config["scoring_model"])
        self._scorer = (
            AutoModelForCausalLM.from_pretrained(self.config["scoring_model"])
            .to(self._device)
            .eval()
        )
        self._mask_tok = AutoTokenizer.from_pretrained(self.config["mask_filling_model"])
        self._mask = (
            AutoModelForSeq2SeqLM.from_pretrained(self.config["mask_filling_model"])
            .to(self._device)
            .eval()
        )
        super().load()

    def predict(self, text: str) -> DetectorResult:
        if not self._loaded:
            self.load()
        ll_orig = self._loglik(text)
        perturbed = [self._perturb(text) for _ in range(self.config["n_perturbations"])]
        ll_perturbed = [self._loglik(p) for p in perturbed]
        mean_ll_perturbed = sum(ll_perturbed) / len(ll_perturbed)
        score = ll_orig - mean_ll_perturbed
        prob_ai = 1.0 / (1.0 + math.exp(-score))
        label = "ai" if score >= self.config["threshold"] else "human"
        return DetectorResult(
            score_ai=prob_ai,
            label=label,
            raw={
                "score": float(score),
                "ll_original": float(ll_orig),
                "ll_perturbed_mean": float(mean_ll_perturbed),
                "n_perturbations": self.config["n_perturbations"],
            },
        )

    # ------------------------------------------------------------------ internals
    def _loglik(self, text: str) -> float:
        from .fast_detect_gpt import _token_logprobs
        return _token_logprobs(
            self._scorer, self._scorer_tok, text,
            self.config["max_length"], self._device,
        )["mean"]

    def _perturb(self, text: str) -> str:
        """Mask ``mask_fraction`` of word-spans and re-fill with the T5 mask LM."""
        import torch

        words = text.split()
        if not words:
            return text
        span = self.config["span_length"]
        n_spans = max(1, int(self.config["mask_fraction"] * len(words) / span))
        positions = sorted(
            self._rng.sample(range(max(1, len(words) - span)), min(n_spans, max(1, len(words) - span)))
        )
        # Deduplicate overlapping spans.
        masked_idx: list[tuple[int, int]] = []
        for p in positions:
            if not masked_idx or p >= masked_idx[-1][1]:
                masked_idx.append((p, p + span))

        masked: list[str] = []
        i = 0
        mask_id = 0
        for start, end in masked_idx:
            masked.extend(words[i:start])
            masked.append(f"<extra_id_{mask_id}>")
            mask_id += 1
            i = end
        masked.extend(words[i:])
        masked_text = " ".join(masked)

        enc = self._mask_tok(masked_text, return_tensors="pt", truncation=True,
                             max_length=self.config["max_length"]).to(self._device)
        with torch.no_grad():
            out = self._mask.generate(**enc, max_new_tokens=64, do_sample=True,
                                      top_p=0.95, num_return_sequences=1)
        fills = self._mask_tok.decode(out[0], skip_special_tokens=False)
        # T5 emits "<extra_id_0> tok... <extra_id_1> tok... </s>"; split and reinsert.
        parts = re.split(r"<extra_id_\d+>", fills)[1:]
        replacements = [p.replace("</s>", "").replace("<pad>", "").strip() for p in parts]
        out_words: list[str] = []
        i = 0
        mask_id = 0
        for start, end in masked_idx:
            out_words.extend(words[i:start])
            fill = replacements[mask_id] if mask_id < len(replacements) else " ".join(words[start:end])
            out_words.append(fill)
            mask_id += 1
            i = end
        out_words.extend(words[i:])
        return " ".join(out_words)
