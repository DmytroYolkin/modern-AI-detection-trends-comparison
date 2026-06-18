"""Stage 1 -- per-seed author-disjoint 70/15/15 re-split of the USE-only cache.

Adapted COPY of
``data/dataset_ready_final/scripts/pipeline/resplit_strict.py`` with three
deliberate changes for the multi-seed controlled experiment:

  1. **Seed is a parameter** (not the hardcoded SEED=42). One call per seed.
  2. **USE-only.** The on-disk ``data/features/`` cache is already the real
     post-filter author-mode cache (5,547 records: 963 human ``use`` +
     4,584 ``use_rewrite``). Grouping is by USE ``author_id`` only; rewrites
     inherit their source essay's author via ``author_id`` / ``source_text_id``.
     No HC3 / ArguGPT / RAID paths.
  3. **Never writes in place.** Reads feature vectors out of the *read-only*
     ``data/features/`` cache (``ids`` are joined back to the corpus jsonl only
     to recover ``author_id`` / ``text`` -- the npz itself carries neither) and
     writes the new per-split ``.npz`` into ``data/features_resplit/seed<k>/``.
     The source cache and the jsonl are opened read-only.

Why this is cheap and re-extraction-free: re-splitting only reassigns which
split each record id lands in. Feature vectors are split-invariant because
TRACE author-context is author-scoped and the splits are author-disjoint, so
this is pure id-reassignment -- zero Ollama / NELA / SBERT / TRACE reruns.

Author-disjoint guarantee: every record sharing a USE ``author_id`` (the source
essay + its rewrites, which inherit the essay's ``author_id``) is placed
atomically into one split; records sharing exact text are unioned into the same
group first as a belt-and-suspenders check.

Usage (from repo root):
    python -m scripts.controlled_resplit.resplit_seed --seed 0
    python -m scripts.controlled_resplit.resplit_seed --seed 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = REPO_ROOT / "data" / "features"                 # READ-ONLY source cache
DATASET_DIR = REPO_ROOT / "data" / "dataset_ready_final"      # READ-ONLY corpus jsonl
RESPLIT_ROOT = REPO_ROOT / "data" / "features_resplit"        # write target

TARGET_RATIOS = (0.70, 0.15, 0.15)
MIN_GROUP_SIZE = 2


# ---------------------------------------------------------------------------
# Load corpus metadata (author_id / source_text_id / text) keyed by record id
# ---------------------------------------------------------------------------

def load_corpus_meta(dataset_dir: Path) -> dict[str, dict]:
    """``id -> {author_id, source_text_id, source, text}`` from the corpus jsonl."""
    meta: dict[str, dict] = {}
    for sp in ("train", "val", "test"):
        p = dataset_dir / f"{sp}.jsonl"
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            meta[r["id"]] = {
                "author_id": r.get("author_id"),
                "source_text_id": r.get("source_text_id"),
                "source": r.get("source"),
                "text": r.get("text"),
            }
    return meta


# ---------------------------------------------------------------------------
# Load the cache rows (feature vectors live here; metadata joined from jsonl)
# ---------------------------------------------------------------------------

def load_cache_rows(feature_dir: Path, corpus_meta: dict[str, dict]) -> list[dict]:
    """Read every cached row from the read-only feature cache.

    Joins each row's id to the corpus jsonl so we recover ``author_id`` and
    ``text`` (the npz carries only nela/style/trace/label/style_ok/ids/sources).
    """
    rows: list[dict] = []
    seen: set[str] = set()
    for sp in ("train", "val", "test"):
        p = feature_dir / f"{sp}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        n = len(d["ids"])
        for i in range(n):
            rid = str(d["ids"][i])
            if rid in seen:
                continue
            seen.add(rid)
            cm = corpus_meta.get(rid, {})
            rows.append({
                "id": rid,
                "source": str(d["sources"][i]) if "sources" in d.files else cm.get("source", "?"),
                "author_id": cm.get("author_id"),
                "source_text_id": cm.get("source_text_id"),
                "text": cm.get("text"),
                # feature columns copied verbatim from the read-only cache
                "nela": d["nela"][i],
                "style": d["style"][i],
                "trace": d["trace"][i],
                "label": d["label"][i],
                "style_ok": d["style_ok"][i] if "style_ok" in d.files else False,
            })
    return rows


# ---------------------------------------------------------------------------
# Grouping -- USE author. Rewrites inherit their source essay's author_id.
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self) -> None:
        self.parent: dict = {}

    def add(self, x) -> None:
        self.parent.setdefault(x, x)

    def find(self, x):
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _group_key(row: dict, by_id: dict[str, dict]):
    """('use_author', author_id) for USE essays + rewrites; else a singleton."""
    src = row.get("source")
    aid = row.get("author_id")
    if src == "use" and aid is not None:
        return ("use_author", str(aid))
    if src == "use_rewrite":
        if aid is not None:
            return ("use_author", str(aid))
        sid = row.get("source_text_id")
        if sid and sid in by_id and by_id[sid].get("author_id") is not None:
            return ("use_author", str(by_id[sid]["author_id"]))
        return ("orphan_rewrite", row["id"])
    return ("singleton", row["id"])


def build_groups(rows: list[dict]) -> dict:
    by_id = {r["id"]: r for r in rows}
    groups: dict = defaultdict(list)
    for r in rows:
        groups[_group_key(r, by_id)].append(r)

    uf = UnionFind()
    for gk in groups:
        uf.add(gk)

    # union groups that share exact text (belt-and-suspenders against any
    # rewrite whose author_id failed to resolve)
    text_to_gks: dict = defaultdict(set)
    for gk, rs in groups.items():
        for r in rs:
            t = r.get("text")
            if t is not None:
                text_to_gks[str(t)].add(gk)
    for gks in text_to_gks.values():
        if len(gks) > 1:
            it = iter(gks)
            first = next(it)
            for other in it:
                uf.union(first, other)

    merged: dict = defaultdict(list)
    for gk, rs in groups.items():
        merged[uf.find(gk)].extend(rs)
    return merged


def drop_singletons(groups: dict, min_size: int) -> tuple[dict, int]:
    kept, dropped = {}, 0
    for gk, rs in groups.items():
        if len(rs) < min_size:
            dropped += len(rs)
            continue
        kept[gk] = rs
    return kept, dropped


# ---------------------------------------------------------------------------
# Stratified atomic split (by dominant source within group)
# ---------------------------------------------------------------------------

def stratify_and_split(groups: dict, ratios=TARGET_RATIOS, seed: int = 0) -> dict:
    rng = random.Random(seed)
    strata: dict = defaultdict(list)
    for rs in groups.values():
        counts: dict = defaultdict(int)
        for r in rs:
            counts[r.get("source", "unknown")] += 1
        dominant = max(counts, key=counts.get)
        strata[dominant].append(rs)

    splits = {"train": [], "val": [], "test": []}
    for gs in strata.values():
        rng.shuffle(gs)
        n = len(gs)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        splits["train"].extend(gs[:n_train])
        splits["val"].extend(gs[n_train:n_train + n_val])
        splits["test"].extend(gs[n_train + n_val:])

    return {sp: [r for g in gs for r in g] for sp, gs in splits.items()}


# ---------------------------------------------------------------------------
# Write npz (only the columns the training scripts read) + verify
# ---------------------------------------------------------------------------

FEATURE_COLS = ("nela", "style", "trace", "label", "style_ok")


def write_split_npz(rows: list[dict], out_path: Path) -> None:
    payload = {
        "nela": np.stack([np.asarray(r["nela"], np.float32) for r in rows]),
        "style": np.stack([np.asarray(r["style"], np.float32) for r in rows]),
        "trace": np.stack([np.asarray(r["trace"], np.float32) for r in rows]),
        "label": np.array([int(np.asarray(r["label"])) for r in rows], dtype=np.int64),
        "style_ok": np.array([bool(np.asarray(r["style_ok"])) for r in rows], dtype=bool),
        "ids": np.array([r["id"] for r in rows], dtype=object),
        "sources": np.array([r["source"] for r in rows], dtype=object),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)


def verify(new_records: dict) -> tuple[bool, dict]:
    tr, va, te = new_records["train"], new_records["val"], new_records["test"]

    def keyset(rows, getter):
        return {getter(r) for r in rows if getter(r) is not None}

    checks = {
        "id": lambda r: r["id"],
        "author_id": lambda r: (str(r["author_id"])
                                if r.get("author_id") is not None else None),
        "text": lambda r: str(r["text"]) if r.get("text") is not None else None,
    }
    clean = True
    report = {}
    for name, get in checks.items():
        s_tr, s_va, s_te = keyset(tr, get), keyset(va, get), keyset(te, get)
        inter = {
            "train_test": len(s_tr & s_te),
            "train_val": len(s_tr & s_va),
            "val_test": len(s_va & s_te),
        }
        report[name] = inter
        if any(inter.values()):
            clean = False
    return clean, report


def split_counts(rows: list[dict]) -> dict:
    human = sum(1 for r in rows if int(np.asarray(r["label"])) == 0)
    ai = sum(1 for r in rows if int(np.asarray(r["label"])) == 1)
    return {"records": len(rows), "human": human, "ai": ai}


# ---------------------------------------------------------------------------
# Per-seed driver
# ---------------------------------------------------------------------------

def resplit_one_seed(seed: int, rows: list[dict]) -> dict:
    groups = build_groups(rows)
    kept, dropped = drop_singletons(groups, MIN_GROUP_SIZE)
    new_records = stratify_and_split(kept, seed=seed)

    out_dir = RESPLIT_ROOT / f"seed{seed}"
    for sp, rs in new_records.items():
        write_split_npz(rs, out_dir / f"{sp}.npz")

    clean, leak_report = verify(new_records)
    meta = {
        "seed": seed,
        "protocol": "A: fixed 70/15/15 author-disjoint, seed varied",
        "source_cache": str(FEATURE_DIR),
        "use_only": True,
        "total_rows": len(rows),
        "n_groups_kept": len(kept),
        "dropped_below_min_group": dropped,
        "min_group_size": MIN_GROUP_SIZE,
        "ratios": list(TARGET_RATIOS),
        "splits": {sp: split_counts(rs) for sp, rs in new_records.items()},
        "leak_check": leak_report,
        "leak_free": clean,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    status = "LEAK-FREE" if clean else "LEAK DETECTED"
    print(f"[seed {seed}] {status}  "
          f"train={meta['splits']['train']['records']} "
          f"val={meta['splits']['val']['records']} "
          f"test={meta['splits']['test']['records']}  "
          f"(groups={len(kept)}, dropped {dropped})  -> {out_dir}")
    if not clean:
        print(f"   leak report: {json.dumps(leak_report)}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    corpus_meta = load_corpus_meta(DATASET_DIR)
    rows = load_cache_rows(FEATURE_DIR, corpus_meta)
    print(f"loaded {len(rows)} cached USE rows from {FEATURE_DIR}")
    src_counts: dict = defaultdict(int)
    miss_author = 0
    for r in rows:
        src_counts[r["source"]] += 1
        if r.get("author_id") is None:
            miss_author += 1
    print(f"source breakdown: {dict(src_counts)}   rows missing author_id: {miss_author}")

    all_clean = True
    for s in args.seed:
        meta = resplit_one_seed(s, rows)
        all_clean = all_clean and meta["leak_free"]
    print("\nALL SEEDS LEAK-FREE" if all_clean else "\nSOME SEEDS HAD LEAKAGE -- inspect logs")
    return 0 if all_clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
