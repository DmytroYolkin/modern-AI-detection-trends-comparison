"""Unified feature-extraction pipeline over the three extractors.

`FeaturePipeline` turns a `TextSample` (see `data/preprocessing/dataset.py`)
into the ``(nela, style, trace)`` feature triple consumed by the fusion model:

    NELA          -> 87  surface/linguistic features      (nela_extractor.py)
    StyleDecipher -> 10  rewrite-similarity statistics     (styledecipher_extractor.py)
    TRACE         -> 128 author-style embedding            (trace_user_profiler.py)

StyleDecipher modes
-------------------
StyleDecipher compares a text against LLM rewrites of it. There are three ways
to obtain those rewrites:

    "cached"  (default) -- the dataset already ships LLM paraphrase clusters
                           (`rewritten_texts.jsonl`). Each USE human essay and
                           its rewrites form a cluster; a sample's StyleDecipher
                           features come from comparing it to the *other*
                           members of its cluster. No LLM calls, fully offline.
    "ollama"            -- generate fresh rewrites on the fly via Ollama
                           (`styledecipher_extractor.generate_rewrites_multi_llm`).
                           Full coverage, but slow and needs a running Ollama.
    "off"               -- skip StyleDecipher; emit a zero vector.

In "cached" mode, samples outside any rewrite cluster (HC3 humans, AI-original
texts) get a zero style vector -- see `meta.json::style_coverage` after a build.
"""

from __future__ import annotations

import difflib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from . import paths  # noqa: F401  -- sys.path bootstrap for the extractor imports

# Feature dimensionalities -- must match `fusion/combination_all.py` defaults.
NELA_DIM = 87
STYLE_DIM = 10
TRACE_DIM = 128


# ===========================================================================
# StyleDecipher similarity helpers
# (mirror `styledecipher_extractor.py`, minus the Ollama dependency so the
#  "cached"/"off" modes stay fully offline)
# ===========================================================================

def _ngrams(text: str, n: int) -> set:
    tokens = text.lower().split()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def ngram_overlap(t1: str, t2: str, n: int) -> float:
    """Jaccard overlap of the n-gram sets of two texts."""
    n1, n2 = _ngrams(t1, n), _ngrams(t2, n)
    if not n1 and not n2:
        return 1.0
    if not n1 or not n2:
        return 0.0
    return len(n1 & n2) / len(n1 | n2)


try:  # python-Levenshtein is fast; fall back to difflib if it is absent
    from Levenshtein import distance as _lev_distance
except Exception:  # pragma: no cover - optional dependency
    _lev_distance = None


def edit_similarity(t1: str, t2: str) -> float:
    """1 - normalised edit distance (1.0 == identical)."""
    max_len = max(len(t1), len(t2))
    if max_len == 0:
        return 0.0
    if _lev_distance is not None:
        return 1.0 - _lev_distance(t1, t2) / max_len
    return difflib.SequenceMatcher(None, t1, t2).ratio()


# ===========================================================================
# Rewrite clusters (for StyleDecipher "cached" mode)
# ===========================================================================

def build_rewrite_clusters(datasets) -> tuple[dict, dict]:
    """Group each USE human essay with its LLM rewrites into a cluster.

    Returns ``(clusters, member_to_cluster)`` where ``clusters`` maps a cluster
    id (the source human record id) to ``{record_id: text}`` and
    ``member_to_cluster`` maps every member record id back to its cluster id.
    """
    if not isinstance(datasets, (list, tuple)):
        datasets = [datasets]

    clusters: dict[str, dict[str, str]] = defaultdict(dict)
    member_to_cluster: dict[str, str] = {}

    # 1. rewrites -> grouped under their source human text id
    for ds in datasets:
        for s in ds:
            if s.is_rewrite and s.source_text_id:
                clusters[s.source_text_id][s.record_id] = s.text
                member_to_cluster[s.record_id] = s.source_text_id

    # 2. add the source human texts themselves to their clusters
    for ds in datasets:
        for s in ds:
            if s.record_id in clusters:           # this human text has rewrites
                clusters[s.record_id][s.record_id] = s.text
                member_to_cluster[s.record_id] = s.record_id

    return dict(clusters), member_to_cluster


# ===========================================================================
# Extracted-feature container
# ===========================================================================

@dataclass(frozen=True)
class ExtractedFeatures:
    """The feature triple produced for one text sample."""

    nela: np.ndarray     # (NELA_DIM,)
    style: np.ndarray    # (STYLE_DIM,)
    trace: np.ndarray    # (TRACE_DIM,)
    style_ok: bool       # True when real StyleDecipher features were computed


# ===========================================================================
# Pipeline
# ===========================================================================

class FeaturePipeline:
    """Runs NELA + StyleDecipher + TRACE and returns one `ExtractedFeatures`.

    Extractors are imported and constructed lazily, so e.g. an "off"
    StyleDecipher build never touches `sentence-transformers`.
    """

    NELA_DIM = NELA_DIM
    STYLE_DIM = STYLE_DIM
    TRACE_DIM = TRACE_DIM

    def __init__(
        self,
        clusters: dict | None = None,
        member_to_cluster: dict | None = None,
        *,
        styledecipher_mode: str = "cached",
        trace_context: str = "single",
        device: str = "auto",
        sbert_model: str = "all-mpnet-base-v2",
        trace_model: str = "sentence-transformers/all-mpnet-base-v2",
        seed: int = 42,
    ) -> None:
        if styledecipher_mode not in ("cached", "ollama", "off"):
            raise ValueError(f"styledecipher_mode must be cached|ollama|off, got {styledecipher_mode!r}")
        if trace_context not in ("single", "author"):
            raise ValueError(f"trace_context must be single|author, got {trace_context!r}")

        self.clusters = clusters or {}
        self.member_to_cluster = member_to_cluster or {}
        self.styledecipher_mode = styledecipher_mode
        self.trace_context = trace_context
        self.device = paths.resolve_device(device)
        self.sbert_model_name = sbert_model
        self.trace_model_name = trace_model
        self.seed = seed

        # lazily-initialised extractor handles
        self._nela = None
        self._sbert = None
        self._trace = None
        self._ollama_rewriter = None

    @classmethod
    def from_datasets(cls, datasets, **kwargs) -> "FeaturePipeline":
        """Build a pipeline whose StyleDecipher cache is derived from `datasets`."""
        clusters, member_to_cluster = build_rewrite_clusters(datasets)
        return cls(clusters, member_to_cluster, **kwargs)

    # ---- lazy extractor construction -------------------------------------

    def _ensure_nela(self):
        if self._nela is None:
            from nela_extractor import NELAFeatureExtractor, ensure_nltk_tokenizers

            ensure_nltk_tokenizers()
            extractor = NELAFeatureExtractor()
            # `nela_features` derives its `num_dates` feature via the
            # `datefinder` library, which raises on out-of-range years (e.g. a
            # negative year parsed from odd text). Wrap count_dates so one bad
            # date yields 0 instead of killing all 87 NELA features.
            _orig_count_dates = extractor.Functions.count_dates

            def _safe_count_dates(text, words):
                try:
                    return _orig_count_dates(text, words)
                except Exception:
                    return 0.0

            extractor.Functions.count_dates = _safe_count_dates
            self._nela = extractor
        return self._nela

    def _ensure_sbert(self):
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer

            self._sbert = SentenceTransformer(self.sbert_model_name, device=self.device)
        return self._sbert

    def _ensure_trace(self):
        if self._trace is None:
            import torch

            from trace_user_profiler import TRACEUserProfileEmbedder

            # The TRACE projection head is randomly initialised; seed it so the
            # same embedding space is used across every split / build run.
            torch.manual_seed(self.seed)
            self._trace = TRACEUserProfileEmbedder(
                model_name=self.trace_model_name, device=self.device
            )
        return self._trace

    def _ensure_ollama_rewriter(self):
        if self._ollama_rewriter is None:
            from styledecipher_extractor import generate_rewrites_multi_llm

            self._ollama_rewriter = generate_rewrites_multi_llm
        return self._ollama_rewriter

    # ---- per-modality extraction -----------------------------------------

    def nela_features(self, text: str) -> np.ndarray:
        """87-dim NELA surface/linguistic feature vector."""
        extractor = self._ensure_nela()
        feats, _names = extractor.extract_all(text or " ")
        arr = np.nan_to_num(
            np.asarray(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        return _fit_dim(arr, self.NELA_DIM)

    def _rewrites_for(self, sample) -> list[str]:
        """Return the rewrites a sample's StyleDecipher features compare against."""
        if self.styledecipher_mode == "off":
            return []

        cluster_id = self.member_to_cluster.get(sample.record_id)
        if cluster_id is not None:
            cluster = self.clusters.get(cluster_id, {})
            return [t for rid, t in cluster.items() if rid != sample.record_id]

        if self.styledecipher_mode == "ollama":
            try:
                return self._ensure_ollama_rewriter()(sample.text)
            except Exception as exc:  # pragma: no cover - network/runtime dependent
                print(f"  [styledecipher] ollama rewrite failed for {sample.record_id}: {exc}")
        return []

    def style_features(self, sample) -> tuple[np.ndarray, bool]:
        """10-dim StyleDecipher vector: mean + std of 5 similarity metrics.

        Returns ``(vector, style_ok)``; ``style_ok`` is False (and the vector
        all-zeros) when no rewrites were available for the sample.
        """
        if self.styledecipher_mode == "off":
            return np.zeros(self.STYLE_DIM, dtype=np.float32), False

        rewrites = [r for r in self._rewrites_for(sample) if r and r.strip()]
        if not rewrites:
            return np.zeros(self.STYLE_DIM, dtype=np.float32), False

        sbert = self._ensure_sbert()
        embeddings = sbert.encode(
            [sample.text] + rewrites,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        base = embeddings[0]

        rows = []
        for i, rew in enumerate(rewrites, start=1):
            rows.append([
                ngram_overlap(sample.text, rew, 1),
                ngram_overlap(sample.text, rew, 2),
                ngram_overlap(sample.text, rew, 3),
                edit_similarity(sample.text, rew),
                float(np.dot(base, embeddings[i])),   # cosine (vectors normalised)
            ])
        rows = np.asarray(rows, dtype=np.float32)
        vec = np.concatenate([rows.mean(axis=0), rows.std(axis=0)])
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return _fit_dim(vec, self.STYLE_DIM), True

    def trace_features(self, sample, siblings=None) -> np.ndarray:
        """128-dim TRACE author-style embedding for the sample."""
        embedder = self._ensure_trace()
        if self.trace_context == "author" and siblings:
            texts = [sample.text] + [s.text for s in siblings]
        else:
            texts = [sample.text]
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return np.zeros(self.TRACE_DIM, dtype=np.float32)

        emb = np.asarray(embedder.get_author_embedding(texts), dtype=np.float32).reshape(-1)
        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        return _fit_dim(emb, self.TRACE_DIM)

    # ---- combined --------------------------------------------------------

    def extract(self, sample, siblings=None) -> ExtractedFeatures:
        """Run all three extractors for one `TextSample`.

        Each extractor is isolated: if one fails, only its block falls back to
        a zero vector -- the other two still contribute real features.
        """
        try:
            nela = self.nela_features(sample.text)
        except Exception as exc:
            print(f"  [nela] {sample.record_id} failed: {exc}")
            nela = np.zeros(self.NELA_DIM, dtype=np.float32)

        try:
            style, style_ok = self.style_features(sample)
        except Exception as exc:
            print(f"  [styledecipher] {sample.record_id} failed: {exc}")
            style, style_ok = np.zeros(self.STYLE_DIM, dtype=np.float32), False

        try:
            trace = self.trace_features(sample, siblings=siblings)
        except Exception as exc:
            print(f"  [trace] {sample.record_id} failed: {exc}")
            trace = np.zeros(self.TRACE_DIM, dtype=np.float32)

        return ExtractedFeatures(nela=nela, style=style, trace=trace, style_ok=style_ok)


def _fit_dim(arr: np.ndarray, dim: int) -> np.ndarray:
    """Defensively pad/trim a 1-D array to exactly `dim` elements."""
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.shape[0] == dim:
        return arr
    fixed = np.zeros(dim, dtype=np.float32)
    fixed[: min(arr.shape[0], dim)] = arr[: min(arr.shape[0], dim)]
    return fixed
