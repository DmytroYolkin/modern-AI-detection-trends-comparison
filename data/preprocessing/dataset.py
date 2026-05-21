"""
Classifier-ready preprocessing layer for the human-vs-AI text dataset.

Turns the raw `dataset_ready_final/*.jsonl` files into clean Python objects
that the feature extractors (NELA, StyleDecipher, TRACE) consume directly.

Feature extractors take a single `.text` as input. TRACE additionally needs
other texts written by the *same author* — that grouping is what this layer
exists to provide.

Author identity
---------------
Every record exposes one unified `author_id` for "give me other texts by the
same author", regardless of whether the text is human- or AI-written:

    USE human essay   ->  the human author            e.g. "0219"
    LLM rewrite       ->  the human author it was      e.g. "0219"
                          rewritten from (inherited)
    AI-original text  ->  the generating model         e.g. "chatgpt"
    HC3 human answer  ->  no author label -> falls back to its own record id
                          (a singleton group; HC3 carries no author info)

The raw components stay available separately, so both axes are queryable:

    .human_author_id  ->  USE author      (humans + their rewrites), else None
    .generator        ->  model that wrote the text (every AI record), else None

So: "who generated this AI text?"  -> `.generator`        / texts_by_generator()
    "which human does it belong to?" -> `.human_author_id` / texts_by_author()

Usage
-----
    from data.preprocessing import Dataset

    train = Dataset.load("train")            # train.jsonl  (val / test / None=merged)

    for sample in train:
        x        = sample.text                       # NELA / StyleDecipher input
        siblings = train.author_siblings(sample)     # TRACE: same-author context
        ...

    # author-disjoint splits keep an author wholly inside one split, so
    # siblings never leak across train/val/test.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "dataset_ready_final"


def _clean(value):
    """Treat null / empty string as 'absent'."""
    return value if value else None


@dataclass(frozen=True, slots=True)
class TextSample:
    """A single normalized text record.

    `text` is the feature-extractor input. `author_id` is the key for
    grouping texts by the same author (TRACE).
    """

    record_id: str
    text: str
    label: str                      # "human" | "ai"
    is_ai: bool
    split: str | None               # "train" | "val" | "test"
    source: str                     # use | argugpt | raid | hc3 | use_rewrite
    domain: str | None
    human_author_id: str | None     # USE author; on USE humans + their rewrites
    generator: str | None           # model that produced the text; None if human
    source_text_id: str | None      # rewrites: id of the original human text

    @property
    def author_id(self) -> str:
        """Unified author key for grouping 'texts by the same author'.

        Human author when known (USE humans + their rewrites), otherwise the
        generating model (AI-originals), otherwise the record's own id
        (authorless HC3 humans -> singleton group).
        """
        return self.human_author_id or self.generator or self.record_id

    @property
    def has_known_author(self) -> bool:
        """False for HC3 humans, which carry no author label."""
        return bool(self.human_author_id or self.generator)

    @property
    def is_rewrite(self) -> bool:
        return self.source == "use_rewrite"

    @classmethod
    def from_record(cls, r: dict) -> "TextSample":
        return cls(
            record_id=r["id"],
            text=r["text"],
            label=r["label"],
            is_ai=bool(r["is_ai"]),
            split=_clean(r.get("split")),
            source=r["source"],
            domain=_clean(r.get("domain")),
            human_author_id=_clean(r.get("author_id")),
            generator=_clean(r.get("model")),
            source_text_id=_clean(r.get("source_text_id")),
        )


class Dataset:
    """An indexed collection of `TextSample`s with author-grouping queries."""

    def __init__(self, samples: list[TextSample]):
        self.samples: list[TextSample] = list(samples)
        self._by_id: dict[str, TextSample] = {}
        self._by_author: dict[str, list[TextSample]] = defaultdict(list)
        self._by_generator: dict[str, list[TextSample]] = defaultdict(list)
        for s in self.samples:
            self._by_id[s.record_id] = s
            self._by_author[s.author_id].append(s)
            if s.generator:
                self._by_generator[s.generator].append(s)

    # ---- loading ----------------------------------------------------------

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Dataset":
        with open(path, encoding="utf-8") as f:
            return cls([
                TextSample.from_record(json.loads(line))
                for line in f if line.strip()
            ])

    @classmethod
    def load(cls, split: str | None = None,
             data_dir: str | Path | None = None) -> "Dataset":
        """Load a split ("train"/"val"/"test"); split=None loads merged.jsonl."""
        base = Path(data_dir) if data_dir else DATA_DIR
        return cls.from_jsonl(base / (f"{split}.jsonl" if split else "merged.jsonl"))

    # ---- container protocol ----------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, idx) -> TextSample:
        return self.samples[idx]

    def __repr__(self) -> str:
        return f"Dataset({len(self.samples)} samples, {len(self._by_author)} authors)"

    # ---- lookup -----------------------------------------------------------

    def get(self, record_id: str) -> TextSample:
        return self._by_id[record_id]

    # ---- author grouping (TRACE) -----------------------------------------

    def texts_by_author(self, author_id: str, *,
                         label: str | None = None) -> list[TextSample]:
        """All texts sharing a unified `author_id` (optionally one label only)."""
        group = self._by_author.get(author_id, [])
        if label is not None:
            group = [s for s in group if s.label == label]
        return list(group)

    def texts_by_generator(self, model: str) -> list[TextSample]:
        """All AI texts produced by a given model — the 'who generated it' axis."""
        return list(self._by_generator.get(model, []))

    def author_siblings(self, sample: TextSample, *,
                         include_self: bool = False,
                         same_label: bool = False) -> list[TextSample]:
        """Other texts by the same author — the context TRACE compares against."""
        out = []
        for s in self._by_author.get(sample.author_id, []):
            if not include_self and s.record_id == sample.record_id:
                continue
            if same_label and s.label != sample.label:
                continue
            out.append(s)
        return out

    def author_groups(self, *, min_size: int = 1) -> dict[str, list[TextSample]]:
        """Map author_id -> its texts, keeping only groups of >= min_size."""
        return {a: list(g) for a, g in self._by_author.items() if len(g) >= min_size}

    @property
    def authors(self) -> list[str]:
        return list(self._by_author.keys())

    # ---- filtering --------------------------------------------------------

    def filter(self, *, label: str | None = None, split: str | None = None,
               source: str | None = None, generator: str | None = None,
               is_ai: bool | None = None) -> "Dataset":
        """Return a new Dataset with the subset matching all given criteria."""
        def keep(s: TextSample) -> bool:
            return (
                (label is None or s.label == label)
                and (split is None or s.split == split)
                and (source is None or s.source == source)
                and (generator is None or s.generator == generator)
                and (is_ai is None or s.is_ai == is_ai)
            )
        return Dataset([s for s in self.samples if keep(s)])

    # ---- convenience ------------------------------------------------------

    @property
    def texts(self) -> list[str]:
        """Raw text inputs in order — direct feed for batched feature extraction."""
        return [s.text for s in self.samples]

    def summary(self) -> dict:
        sizes = [len(g) for g in self._by_author.values()]
        return {
            "records": len(self.samples),
            "by_label": dict(Counter(s.label for s in self.samples)),
            "by_split": dict(Counter(s.split for s in self.samples)),
            "by_source": dict(Counter(s.source for s in self.samples)),
            "authors": len(self._by_author),
            "authors_with_multiple_texts": sum(1 for n in sizes if n > 1),
            "max_texts_per_author": max(sizes, default=0),
        }


def load_splits(data_dir: str | Path | None = None) -> dict[str, Dataset]:
    """Load all three splits at once: {"train": ..., "val": ..., "test": ...}."""
    return {sp: Dataset.load(sp, data_dir) for sp in ("train", "val", "test")}


if __name__ == "__main__":
    for _split in ("train", "val", "test"):
        _ds = Dataset.load(_split)
        print(f"\n[{_split}]")
        print(json.dumps(_ds.summary(), indent=2))
