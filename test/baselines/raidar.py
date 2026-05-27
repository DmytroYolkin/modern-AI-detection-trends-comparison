"""RAIDAR -- geneRative AI Detection viA Rewriting (Mao et al. 2024).

Paper:  https://arxiv.org/abs/2401.12970
Ref impl: https://github.com/cvlab-columbia/RAIDAR

Core idea: ask an LLM to *rewrite* the candidate text, then measure the edit
distance between the candidate and the rewrite. Human text gets edited more
heavily than AI text (the LLM rewriter is closer in style to AI-generated
input, so its rewrite of AI input changes less). Concretely the wrapper uses
the normalised character-level edit distance ``1 - lev(x, rewrite(x)) /
max(|x|, |rewrite(x)|)`` -- so the *similarity*-to-rewrite is the AI-score.

The reference paper trains a small classifier on top of several
edit-distance-derived features; this wrapper uses the single-similarity
threshold variant, which is the headline "training-free" mode in the paper.
Set ``rewriter`` to any local Ollama model -- the project already uses
ollama for the StyleDecipher cache, so the dependency is free here.
"""

from __future__ import annotations

from typing import Any

from .base import BaselineDetector, DetectorResult


class RAIDAR(BaselineDetector):
    name = "raidar"
    requires = ("ollama (running locally)", "python-Levenshtein")

    def __init__(
        self,
        rewriter: str = "llama3",
        rewrite_prompt: str = (
            "Rewrite the following text while preserving its meaning and length. "
            "Output only the rewrite, no preamble.\n\nTEXT:\n{text}\n\nREWRITE:"
        ),
        ollama_host: str = "http://localhost:11434",
        threshold: float = 0.65,
        timeout_s: float = 120.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            rewriter=rewriter,
            rewrite_prompt=rewrite_prompt,
            ollama_host=ollama_host,
            threshold=threshold,
            timeout_s=timeout_s,
            **kwargs,
        )
        self._client = None

    def load(self) -> None:
        if self._loaded:
            return
        import ollama  # noqa: F401
        from Levenshtein import distance  # noqa: F401

        import ollama as _ollama
        self._client = _ollama.Client(host=self.config["ollama_host"])
        super().load()

    def predict(self, text: str) -> DetectorResult:
        if not self._loaded:
            self.load()
        from Levenshtein import distance

        prompt = self.config["rewrite_prompt"].format(text=text)
        response = self._client.generate(
            model=self.config["rewriter"],
            prompt=prompt,
            options={"num_predict": max(256, len(text.split()))},
        )
        rewrite = (response.get("response") or "").strip()
        if not rewrite:
            similarity = 0.0
        else:
            similarity = 1.0 - distance(text, rewrite) / max(len(text), len(rewrite))
        prob_ai = max(0.0, min(1.0, similarity))
        label = "ai" if similarity >= self.config["threshold"] else "human"
        return DetectorResult(
            score_ai=prob_ai,
            label=label,
            raw={
                "similarity": float(similarity),
                "rewrite_len": len(rewrite),
                "text_len": len(text),
            },
        )
