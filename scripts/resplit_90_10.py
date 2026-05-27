"""Step-1 resplit: pool the existing filtered feature caches (USE-only, post
``--require-known-author --min-human-siblings 2``) into a single table and
draw a fresh author-disjoint 90/10 train/val split.

The previous train/val/test cache is preserved in
``data/features_step1_pre_resplit_backup_2026-05-27/``; this script overwrites
``data/features/{train,val}.npz`` and removes ``data/features/test.npz`` so
that no stale held-out split survives. The new test set is the arXiv set
(``data/testing_dataset/arxiv_final/``) and is built in a separate later step.

The split groups by ``author_id`` (loaded from
``data/dataset_ready_final/merged.jsonl``). Records whose author is ``None``
(should be zero after the filters but defended against) are assigned a unique
synthetic group key so they end up isolated rather than colliding.

Usage
-----
    python -m scripts.resplit_90_10
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FEATURE_DIR = REPO_ROOT / "data" / "features"
BACKUP_DIR = REPO_ROOT / "data" / "features_step1_pre_resplit_backup_2026-05-27"
MERGED_JSONL = REPO_ROOT / "data" / "dataset_ready_final" / "merged.jsonl"

OLD_SPLITS = ("train", "val", "test")
SEED = 42
TEST_SIZE = 0.10  # -> 10% val


def _load_split(name: str) -> dict:
    path = BACKUP_DIR / f"{name}.npz"
    if not path.exists():
        raise SystemExit(f"missing pre-resplit backup cache: {path}")
    data = np.load(path, allow_pickle=True)
    return {
        "nela": data["nela"].astype(np.float32),
        "style": data["style"].astype(np.float32),
        "trace": data["trace"].astype(np.float32),
        "label": data["label"].astype(np.int64),
        "style_ok": data["style_ok"].astype(bool),
        "ids": np.array([str(x) for x in data["ids"]], dtype=object),
        "sources": np.array([str(x) for x in data["sources"]], dtype=object),
    }


def _load_record_to_author() -> dict:
    """``record_id -> author_id`` map from merged.jsonl (author_id may be None)."""
    mapping = {}
    with MERGED_JSONL.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            rid = obj.get("id")
            if rid is None:
                continue
            mapping[str(rid)] = obj.get("author_id")
    return mapping


def _pool(splits: list[dict]) -> dict:
    keys = ("nela", "style", "trace", "label", "style_ok", "ids", "sources")
    return {k: np.concatenate([s[k] for s in splits], axis=0) for k in keys}


def _save_npz(path: Path, idx: np.ndarray, pool: dict) -> None:
    payload = {k: pool[k][idx] for k in pool}
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def _label_counts(labels: np.ndarray) -> dict:
    return {"human": int((labels == 0).sum()), "ai": int((labels == 1).sum())}


def _source_counts(sources: np.ndarray) -> dict:
    out: dict = {}
    for s in sources:
        out[str(s)] = out.get(str(s), 0) + 1
    return dict(sorted(out.items()))


def main() -> None:
    print("loading pre-resplit backup caches ...")
    parts = []
    for name in OLD_SPLITS:
        sp = _load_split(name)
        print(f"  {name:<5} {len(sp['ids']):>5} records "
              f"(label counts: {_label_counts(sp['label'])})")
        parts.append(sp)

    pool = _pool(parts)
    n = len(pool["ids"])
    print(f"pooled total: {n} records")

    print("building record_id -> author_id map from merged.jsonl ...")
    rid_to_author = _load_record_to_author()

    groups = []
    missing = 0
    none_author = 0
    for rid in pool["ids"]:
        rid = str(rid)
        if rid not in rid_to_author:
            # record not present in merged.jsonl -- shouldn't happen, but guard
            missing += 1
            groups.append(f"missing_{rid}")
            continue
        aid = rid_to_author[rid]
        if aid is None:
            none_author += 1
            groups.append(f"singleton_{rid}")
        else:
            groups.append(str(aid))
    groups = np.array(groups, dtype=object)
    if missing:
        print(f"  WARN: {missing} record_ids not found in merged.jsonl (treated as singletons)")
    if none_author:
        print(f"  WARN: {none_author} records had author_id=None (treated as singletons)")
    print(f"  unique group keys: {len(set(groups))}")

    print(f"running GroupShuffleSplit(n_splits=1, test_size={TEST_SIZE}, "
          f"random_state={SEED}) ...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, val_idx = next(splitter.split(np.zeros(n), pool["label"], groups))

    train_authors = set(groups[train_idx].tolist())
    val_authors = set(groups[val_idx].tolist())
    overlap = train_authors & val_authors
    print(f"  train rows: {len(train_idx)}  val rows: {len(val_idx)}")
    print(f"  train unique groups: {len(train_authors)}  "
          f"val unique groups: {len(val_authors)}")
    assert overlap == set(), (
        f"author-disjoint INVARIANT VIOLATED: {len(overlap)} overlapping group(s)")
    print("  assert set(train_groups) & set(val_groups) == set()  PASSED")

    # --- write new caches --------------------------------------------------
    train_path = FEATURE_DIR / "train.npz"
    val_path = FEATURE_DIR / "val.npz"
    _save_npz(train_path, train_idx, pool)
    _save_npz(val_path, val_idx, pool)
    print(f"wrote: {train_path}")
    print(f"wrote: {val_path}")

    # --- remove stale test.npz (moved to backup already) -------------------
    stale_test = FEATURE_DIR / "test.npz"
    if stale_test.exists():
        stale_test.unlink()
        print(f"removed stale: {stale_test}")

    # --- per-split summary -------------------------------------------------
    def _summary(idx: np.ndarray) -> dict:
        labels = pool["label"][idx]
        sources = pool["sources"][idx]
        ratio = (float((labels == 0).sum()) / max(int((labels == 1).sum()), 1))
        return {
            "records": int(len(idx)),
            "human": int((labels == 0).sum()),
            "ai": int((labels == 1).sum()),
            "human_to_ai_ratio": round(ratio, 4),
            "style_coverage": round(float(pool["style_ok"][idx].mean()), 4),
            "unique_authors": int(len(set(groups[idx].tolist()))),
            "source_counts": _source_counts(sources),
        }

    train_summary = _summary(train_idx)
    val_summary = _summary(val_idx)

    print("\n=== new split summary ===")
    for name, s in (("train", train_summary), ("val", val_summary)):
        print(f"  {name:<5} records={s['records']:>5}  human={s['human']:>4}  "
              f"ai={s['ai']:>4}  human:ai={s['human_to_ai_ratio']}  "
              f"unique_authors={s['unique_authors']}  "
              f"style_coverage={s['style_coverage']}")
        print(f"        sources={s['source_counts']}")

    # --- write meta.json --------------------------------------------------
    meta = {
        "dims": {"nela": 87, "style": 10, "trace": 128},
        "styledecipher_mode": "ollama",
        "trace_context": "author",
        "require_known_author": True,
        "min_human_siblings": 2,
        "limit": None,
        "seed": SEED,
        "splits": {"train": train_summary, "val": val_summary},
        "resplit_note": (
            "step-1 resplit: pooled prior train+val+test cache (USE-only, post-filter, "
            f"n={n}) and resplit author-disjoint 90/10, seed={SEED}; previous test split "
            "is *not* held out, new test set is data/testing_dataset/arxiv_final/"
        ),
        "rebuild_note": (
            "TRACE rebuilt in author mode; HC3 no-author humans dropped; "
            "NELA + StyleDecipher (ollama) preserved from prior cache"
        ),
        "pre_resplit_backup": str(BACKUP_DIR.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    (FEATURE_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote: {FEATURE_DIR / 'meta.json'}")


if __name__ == "__main__":
    main()
