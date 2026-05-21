"""
Finalize the dataset:
- length filter (100..1000 words)
- unify schema across all files (same fields, missing = null)
- author-disjoint stratified train/val/test split (70/15/15)
- detailed stats per split x label x domain
- merged.jsonl reflecting the filtered + split data

Author-disjoint split is critical for TRACE: if texts of the same author
appeared in both train and val, the contrastive author-fingerprint model
would leak. We split *authors* into 70/15/15, then assign each text by its
author_id. AI texts (no author_id) are split independently.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

INPUT_DIR  = Path("/home/konrado/dataset_ready")
OUTPUT_DIR = Path("/home/konrado/dataset_ready")

MIN_WORDS = 100
MAX_WORDS = 1000
RANDOM_SEED = 42

CANONICAL_FIELDS = [
    "id",
    "text",
    "is_ai",
    "label",                  # redundant with is_ai, kept for readability
    "source",
    "domain",
    "author_id",
    "model",
    "exam_type",
    "prompt",
    "source_text_id",         # only set on rewrites
    "text_length_words",
    "split",                  # train | val | test
]


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def word_count(text: str) -> int:
    return len(text.split())


def normalise(record: dict) -> dict:
    out = {k: record.get(k) for k in CANONICAL_FIELDS}
    out["label"] = "ai" if record.get("is_ai") else "human"
    out["text_length_words"] = word_count(record["text"])
    return out


def author_disjoint_split(
    records: list[dict],
    rng: random.Random,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, str]:
    """Return mapping record_id -> split label (train/val/test).

    Records with author_id are grouped by author and assigned together.
    Records without author_id (AI texts from ArguGPT/RAID) are split
    independently at the record level.
    """
    by_author: dict[str, list[str]] = defaultdict(list)
    no_author: list[str] = []
    for r in records:
        if r["author_id"]:
            by_author[r["author_id"]].append(r["id"])
        else:
            no_author.append(r["id"])

    def _split_pool(pool: list, key: str) -> dict[str, str]:
        rng.shuffle(pool)
        n = len(pool)
        n_train = int(n * ratios[0])
        n_val   = int(n * ratios[1])
        result = {}
        for i, item in enumerate(pool):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            if key == "author":
                # item is author_id, expand to all its record_ids
                for rid in by_author[item]:
                    result[rid] = split
            else:
                result[item] = split
        return result

    authors = list(by_author.keys())
    auth_assign = _split_pool(authors, "author")
    rec_assign  = _split_pool(no_author, "record")
    return {**auth_assign, **rec_assign}


def main() -> None:
    rng = random.Random(RANDOM_SEED)

    human = load_jsonl(INPUT_DIR / "human_texts.jsonl")
    ai    = load_jsonl(INPUT_DIR / "ai_texts.jsonl")
    rew   = load_jsonl(INPUT_DIR / "rewritten_texts.jsonl")

    all_recs = human + ai + rew

    # 1. Normalise schema + word count
    normalised = [normalise(r) for r in all_recs]

    # 2. Length filter (count drops per group)
    drops = Counter()
    kept = []
    for r in normalised:
        wc = r["text_length_words"]
        if wc < MIN_WORDS or wc > MAX_WORDS:
            drops[(r["label"], r["domain"], r["source"])] += 1
            continue
        kept.append(r)

    # 3. Author-disjoint stratified split
    split_map = author_disjoint_split(kept, rng)
    for r in kept:
        r["split"] = split_map[r["id"]]

    # 4. Re-bucket per file (after filter)
    out_human = [r for r in kept if r["source"] == "use"]
    out_ai    = [r for r in kept if r["source"] in ("argugpt", "raid")]
    out_rew   = [r for r in kept if r["source"] == "use_rewrite"]
    out_merged = kept

    write_jsonl(OUTPUT_DIR / "human_texts.jsonl",     out_human)
    write_jsonl(OUTPUT_DIR / "ai_texts.jsonl",        out_ai)
    write_jsonl(OUTPUT_DIR / "rewritten_texts.jsonl", out_rew)
    write_jsonl(OUTPUT_DIR / "merged.jsonl",          out_merged)

    # 5. Per-split files (handy for downstream loaders)
    for split in ("train", "val", "test"):
        write_jsonl(
            OUTPUT_DIR / f"{split}.jsonl",
            [r for r in kept if r["split"] == split],
        )

    # 6. Sanity assertions
    ids = [r["id"] for r in kept]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for r in kept:
        assert r["text"], "empty text"
    # author-disjoint check
    train_authors = {r["author_id"] for r in kept if r["split"] == "train" and r["author_id"]}
    val_authors   = {r["author_id"] for r in kept if r["split"] == "val"   and r["author_id"]}
    test_authors  = {r["author_id"] for r in kept if r["split"] == "test"  and r["author_id"]}
    assert not (train_authors & val_authors),  "author leak train<->val"
    assert not (train_authors & test_authors), "author leak train<->test"
    assert not (val_authors   & test_authors), "author leak val<->test"

    # 7. Stats
    def lengths(rs):
        ws = sorted(r["text_length_words"] for r in rs)
        if not ws:
            return {"n": 0}
        n = len(ws)
        return {
            "n":      n,
            "min":    ws[0],
            "median": ws[n // 2],
            "mean":   round(sum(ws) / n, 1),
            "max":    ws[-1],
        }

    stats = {
        "min_words":    MIN_WORDS,
        "max_words":    MAX_WORDS,
        "total_after_filter": len(kept),
        "dropped_by_filter":  sum(drops.values()),
        "drops_by_group":     {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in drops.items()},
        "by_split": {
            split: {
                "total": sum(1 for r in kept if r["split"] == split),
                "by_label":  dict(Counter(r["label"]  for r in kept if r["split"] == split)),
                "by_domain": dict(Counter(r["domain"] for r in kept if r["split"] == split)),
                "by_source": dict(Counter(r["source"] for r in kept if r["split"] == split)),
                "lengths_human": lengths([r for r in kept if r["split"] == split and r["label"] == "human"]),
                "lengths_ai":    lengths([r for r in kept if r["split"] == split and r["label"] == "ai"]),
            }
            for split in ("train", "val", "test")
        },
        "authors": {
            "total":           len({r["author_id"] for r in kept if r["author_id"]}),
            "in_train":        len(train_authors),
            "in_val":          len(val_authors),
            "in_test":         len(test_authors),
            "rewrites_per_author_min_max": {
                "min": min(Counter(r["author_id"] for r in out_rew).values(), default=0),
                "max": max(Counter(r["author_id"] for r in out_rew).values(), default=0),
            },
        },
    }
    (OUTPUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
