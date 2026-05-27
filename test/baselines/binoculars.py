"""Binoculars (Hans, Schwarzschild, Cherepanova, Kazemi, Saha, Goldblum, Geiping, Goldstein 2024).

Paper:  https://arxiv.org/abs/2401.12070
Ref impl: https://github.com/ahans30/Binoculars

Two LMs are used: an *observer* (M_obs) and a *performer* (M_perf) from the
same family but at different scales (the paper uses Falcon-7B as observer,
Falcon-7B-instruct as performer). For a candidate ``x``:

    score(x) = ppl_obs(x) / x_ppl_obs_perf(x)

where ``ppl_obs(x)`` is the observer's per-token perplexity on ``x`` and
``x_ppl_obs_perf(x)`` is the cross-perplexity using ``M_perf`` as the
distribution under which the observer's tokens are evaluated. Lower scores
indicate AI-generated text. The paper's default threshold is approximately
0.901 ("accuracy" operating point) -- exposed here as ``threshold``.

**Resource note.** Both models are loaded simultaneously. A 7B/7B pair needs
~30 GB GPU memory in fp16 or ~15 GB in 4-bit. Override ``observer`` and
``performer`` with a smaller pair (e.g. GPT-2 / GPT-2-medium) for CPU smoke
tests.
"""

from __future__ import annotations

import math
from typing import Any

from .base import BaselineDetector, DetectorResult


class Binoculars(BaselineDetector):
    """Binoculars detector wrapper.

    Kwargs
    ------
    observer, performer
        HuggingFace model IDs for the observer / performer LM pair. Must share
        a tokenizer vocabulary (paper default: Falcon-7B + Falcon-7B-instruct).
    device, max_length, threshold, dtype
        Standard knobs; ``threshold`` is the operating point from the paper.
    load_in_4bit
        When True, both models are loaded with a ``BitsAndBytesConfig`` for
        4-bit NF4 weights (compute dtype fp16). This drops the 7B/7B pair from
        ~28 GB to ~8 GB GPU resident, making it fit on a 5090 (32 GB) alongside
        other detectors. Requires ``bitsandbytes`` installed. When False (the
        default), the existing fp16/bf16 path runs unchanged.
    """

    name = "binoculars"
    requires = ("torch", "transformers", "accelerate (recommended for 7B pair)",
                "bitsandbytes (for load_in_4bit=True)")

    def __init__(
        self,
        observer: str = "gpt2",                # paper default: "tiiuae/falcon-7b"
        performer: str = "gpt2-medium",        # paper default: "tiiuae/falcon-7b-instruct"
        device: str = "auto",
        max_length: int = 512,
        threshold: float = 0.901,
        dtype: str = "auto",                   # "auto" | "float16" | "bfloat16" | "float32"
        load_in_4bit: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            observer=observer,
            performer=performer,
            device=device,
            max_length=max_length,
            threshold=threshold,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            **kwargs,
        )
        self._obs = None
        self._obs_tok = None
        self._perf = None
        self._perf_tok = None
        self._device = device

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .fast_detect_gpt import _resolve_device

        self._device = _resolve_device(self.config["device"])

        load_kwargs_common: dict[str, Any]
        if self.config.get("load_in_4bit"):
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            load_kwargs_common = {"quantization_config": quant_config}
            # bitsandbytes places weights on GPU itself; skip the .to(device) call.
            move_to_device = False
        else:
            dtype = _resolve_dtype(self.config["dtype"], self._device)
            load_kwargs_common = {"torch_dtype": dtype}
            move_to_device = True

        self._obs_tok = AutoTokenizer.from_pretrained(self.config["observer"])
        obs = AutoModelForCausalLM.from_pretrained(
            self.config["observer"], **load_kwargs_common
        )
        if move_to_device:
            obs = obs.to(self._device)
        self._obs = obs.eval()
        self._perf_tok = AutoTokenizer.from_pretrained(self.config["performer"])
        perf = AutoModelForCausalLM.from_pretrained(
            self.config["performer"], **load_kwargs_common
        )
        if move_to_device:
            perf = perf.to(self._device)
        self._perf = perf.eval()
        # Binoculars requires both models share a tokenizer (per the paper); if
        # they do not, cross-perplexity is ill-defined. The default Falcon pair
        # satisfies this; for ad-hoc smaller pairs the user must verify.
        if self._obs_tok.vocab_size != self._perf_tok.vocab_size:
            raise RuntimeError(
                "Binoculars requires observer and performer to share a tokenizer; "
                f"got vocab sizes {self._obs_tok.vocab_size} vs {self._perf_tok.vocab_size}."
            )
        super().load()

    def predict(self, text: str) -> DetectorResult:
        if not self._loaded:
            self.load()
        import torch

        enc = self._obs_tok(text, return_tensors="pt", truncation=True,
                            max_length=self.config["max_length"]).to(self._device)
        input_ids = enc["input_ids"]
        with torch.no_grad():
            obs_logits = self._obs(input_ids).logits[:, :-1, :]
            perf_logits = self._perf(input_ids).logits[:, :-1, :]
        targets = input_ids[:, 1:]

        obs_lp = torch.log_softmax(obs_logits, dim=-1)
        perf_lp = torch.log_softmax(perf_logits, dim=-1)
        # ppl(observer): exp(-mean log p_obs(x_i | x_<i))
        gather_obs = obs_lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
        ppl_obs = math.exp(-float(gather_obs.mean().item()))
        # cross-ppl: H(p_obs, p_perf) = -sum p_obs log p_perf, averaged over positions.
        x_entropy = -(obs_lp.exp() * perf_lp).sum(dim=-1).mean()
        x_ppl = math.exp(float(x_entropy.item()))
        score = ppl_obs / x_ppl if x_ppl > 1e-8 else float("inf")

        # Lower score => more likely AI. Convert to P(ai) in [0,1] for ROC-AUC.
        # 1 / (1 + score) is a monotonic squash that matches the direction.
        prob_ai = 1.0 / (1.0 + score) if math.isfinite(score) else 0.0
        label = "ai" if score < self.config["threshold"] else "human"
        return DetectorResult(
            score_ai=prob_ai,
            label=label,
            raw={"binoculars": float(score), "ppl_obs": ppl_obs, "x_ppl": x_ppl},
        )


def _resolve_dtype(name: str, device: str):
    import torch
    if name != "auto":
        return getattr(torch, name)
    return torch.float16 if device.startswith("cuda") else torch.float32
