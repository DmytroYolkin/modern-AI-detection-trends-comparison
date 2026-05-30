"""RADAR (Hu, Chen, Ho 2023) -- adversarial AI-text detector.

Paper:  https://arxiv.org/abs/2307.03838
Checkpoint: https://huggingface.co/TrustSafeAI/RADAR-Vicuna-7B

RADAR ships as a single fine-tuned sequence-classification checkpoint on
HuggingFace, so this wrapper is a thin loader: tokenize, forward, softmax,
take the AI-probability column. The model is ~7 B parameters in fp32 (~14 GB
in fp16) -- the wrapper exposes ``dtype`` so it can be loaded in fp16 on a
single 24 GB GPU.

Label convention in the checkpoint: index 0 = human, index 1 = AI. Verified
against the model card.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator

from .base import BaselineDetector, DetectorResult


class RADAR(BaselineDetector):
    """RADAR detector wrapper.

    Kwargs
    ------
    checkpoint
        HuggingFace seq-classification checkpoint (paper: ``TrustSafeAI/RADAR-Vicuna-7B``).
    device, max_length, threshold, dtype, batch_size
        Standard knobs.
    load_in_4bit
        When True, the 7B checkpoint is loaded via ``BitsAndBytesConfig`` for
        4-bit NF4 weights (compute dtype fp16). Drops VRAM from ~14 GB (fp16)
        to ~3.5 GB, making it fit on an 8 GB laptop GPU. Requires
        ``bitsandbytes`` installed.
    """

    name = "radar"
    requires = ("torch", "transformers", "accelerate (recommended for 7B fp16)",
                "bitsandbytes (for load_in_4bit=True)")

    def __init__(
        self,
        checkpoint: str = "TrustSafeAI/RADAR-Vicuna-7B",
        device: str = "auto",
        max_length: int = 512,
        dtype: str = "auto",
        threshold: float = 0.5,
        batch_size: int = 8,
        load_in_4bit: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            checkpoint=checkpoint,
            device=device,
            max_length=max_length,
            dtype=dtype,
            threshold=threshold,
            batch_size=batch_size,
            load_in_4bit=load_in_4bit,
            **kwargs,
        )
        self._model = None
        self._tok = None
        self._device = device

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from .fast_detect_gpt import _resolve_device
        from .binoculars import _resolve_dtype

        self._device = _resolve_device(self.config["device"])
        self._tok = AutoTokenizer.from_pretrained(self.config["checkpoint"])

        if self.config.get("load_in_4bit"):
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            # bitsandbytes places weights on GPU itself; skip the .to(device) call.
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.config["checkpoint"], quantization_config=quant_config,
            ).eval()
        else:
            dtype = _resolve_dtype(self.config["dtype"], self._device)
            self._model = (
                AutoModelForSequenceClassification
                .from_pretrained(self.config["checkpoint"], torch_dtype=dtype)
                .to(self._device)
                .eval()
            )
        super().load()

    def predict(self, text: str) -> DetectorResult:
        return next(self.predict_batch([text]))

    def predict_batch(self, texts: Iterable[str]) -> Iterator[DetectorResult]:
        if not self._loaded:
            self.load()
        import torch

        batch: list[str] = []
        for text in texts:
            batch.append(text)
            if len(batch) >= self.config["batch_size"]:
                yield from self._forward(batch)
                batch = []
        if batch:
            yield from self._forward(batch)

    def _forward(self, batch: list[str]) -> Iterator[DetectorResult]:
        import torch

        enc = self._tok(batch, return_tensors="pt", truncation=True, padding=True,
                        max_length=self.config["max_length"]).to(self._device)
        with torch.no_grad():
            logits = self._model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].tolist()
        for p in probs:
            yield DetectorResult(
                score_ai=float(p),
                label="ai" if p >= self.config["threshold"] else "human",
                raw={"prob_ai": float(p)},
            )
