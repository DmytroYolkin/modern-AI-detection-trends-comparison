"""Strict resplit of dataset_ready_final to eliminate all train/val/test leakage.

Enforces three rules atomically:

  1. **No text leakage** -- any two records sharing exact text are forced into the
     same split (handles the HC3 Wikipedia-entry duplicates that the original
     split left scattered across train/val/test).

  2. **Author / prompt disjoint** -- all records belonging to the same "author"
     stay together:
       * USE source essays + their rewrites  -> grouped by `author_id` (rewrites
         inherit their source's author)
       * HC3 records                         -> grouped by `prompt` (so the
         human and ChatGPT answers to the same question always share a split)
       * ArguGPT / RAID                      -> singletons (no author concept;
         each record is free to land anywhere, since the "author" is just the
         generating model and isn't the leakage axis we care about)

  3. **>= MIN_GROUP_SIZE records per author group** (default 2) so the TRACE
     extractor always has at least one sibling text when `trace_context="author"`
     is enabled. Author-based groups (USE author / HC3 prompt) below the
     threshold are dropped from the dataset entirely. Source-singleton groups
     (ArguGPT/RAID) are not subject to this rule.

Resplit ratio: ~70 / 15 / 15, stratified by dominant-source within each group
so source distribution stays balanced across the new splits.

Side effects:
  * Backs up old `data/dataset_ready_final/{train,val,test}.jsonl` to `.bak`.
  * Backs up old `data/features/{train,val,test}.npz`           to `.bak`.
  * Writes new `.jsonl` files at the same paths.
  * Filters the existing cached features (`data/features/*.npz`) into new
    `.npz`s matching the new splits -- NO re-extraction needed.
  * Updates `data/features/meta.json` with the new per-split stats.

Run from the repo root:
    python -m data.dataset_ready_final.scripts.pipeline.resplit_strict
or
    python data/dataset_ready_final/scripts/pipeline/resplit_strict.py
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT       = Path(__file__).resolve().parents[4]
DATASET_DIR     = REPO_ROOT / "data" / "dataset_ready_final"
FEATURE_DIR     = REPO_ROOT / "data" / "features"

SEED            = 42
TARGET_RATIOS   = (0.70, 0.15, 0.15)  # train, val, test
MIN_GROUP_SIZE  = 2                   # for author-based groups (USE / HC3)


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class UnionFind:
    """Minimal Union-Find for merging groups that share text."""
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


# ---------------------------------------------------------------------------
# Grouping logic
# ---------------------------------------------------------------------------

def group_key(record: dict, by_id: dict[str, dict]):
    """Return a hashable ('type', identifier) tuple naming this record's group.

    USE essays and their rewrites share the USE author. HC3 humans + their
    ChatGPT counterparts share the prompt. ArguGPT / RAID records are singletons.
    """
    src = record.get("source")

    if src == "use":
        return ("use_author", record.get("author_id"))

    if src == "use_rewrite":
        sid = record.get("source_text_id")
        if sid and sid in by_id:
            return ("use_author", by_id[sid].get("author_id"))
        return ("orphan_rewrite", record["id"])

    if src == "hc3":
        return ("hc3_prompt", record.get("prompt"))

    if src == "argugpt":
        return ("argugpt_singleton", record["id"])

    if src == "raid":
        return ("raid_singleton", record["id"])

    return ("unknown", record["id"])


def is_author_based(group_records: list[dict]) -> bool:
    """Does this group represent a human author or HC3 prompt (so the >=2 rule applies)?"""
    for r in group_records:
        if r.get("source") in ("use", "use_rewrite", "hc3"):
            return True
    return False


# ---------------------------------------------------------------------------
# Stage 1 -- assemble + merge groups
# ---------------------------------------------------------------------------

def build_groups(records: list[dict]) -> dict:
    by_id = {r["id"]: r for r in records}

    # initial grouping by author/prompt/singleton
    groups: dict = defaultdict(list)
    for r in records:
        groups[group_key(r, by_id)].append(r)

    # union groups that share an exact text (handles HC3 Wikipedia duplicates)
    uf = UnionFind()
    for gk in groups:
        uf.add(gk)
    text_to_gks: dict = defaultdict(set)
    for gk, rs in groups.items():
        for r in rs:
            text_to_gks[r["text"]].add(gk)
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
    kept: dict = {}
    dropped_records = 0
    for gk, rs in groups.items():
        if is_author_based(rs) and len(rs) < min_size:
            dropped_records += len(rs)
            continue
        kept[gk] = rs
    return kept, dropped_records


# ---------------------------------------------------------------------------
# Stage 2 -- stratified atomic split
# ---------------------------------------------------------------------------

def stratify_and_split(groups: dict, ratios=TARGET_RATIOS, seed=SEED) -> dict:
    rng = random.Random(seed)

    strata: dict = defaultdict(list)
    for rs in groups.values():
        # bucket by dominant source so train/val/test see proportional source mixes
        counts = defaultdict(int)
        for r in rs:
            counts[r.get("source", "unknown")] += 1
        dominant = max(counts, key=counts.get)
        strata[dominant].append(rs)

    splits = {"train": [], "val": [], "test": []}
    for stratum, gs in strata.items():
        rng.shuffle(gs)
        n = len(gs)
        n_train = int(round(n * ratios[0]))
        n_val   = int(round(n * ratios[1]))
        n_test  = n - n_train - n_val
        splits["train"].extend(gs[:n_train])
        splits["val"].extend(gs[n_train:n_train + n_val])
        splits["test"].extend(gs[n_train + n_val:n_train + n_val + n_test])

    # flatten groups → records, also fix each record's `split` field
    flat: dict = {}
    for sp, gs in splits.items():
        rs = [r for g in gs for r in g]
        for r in rs:
            r["split"] = sp
        flat[sp] = rs
    return flat


# ---------------------------------------------------------------------------
# Stage 3 -- write out + filter feature cache
# ---------------------------------------------------------------------------

def backup(path: Path) -> None:
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))


def write_new_splits(new_records: dict, dataset_dir: Path) -> None:
    for sp, rs in new_records.items():
        out = dataset_dir / f"{sp}.jsonl"
        backup(out)
        write_jsonl(out, rs)


def filter_feature_cache(new_records: dict, feature_dir: Path) -> dict[str, int]:
    """Build new .npz files by selecting cached rows whose id is in the new split."""
    # Load every existing cache so we can look up any record id.
    all_id_to_row: dict[str, dict] = {}
    for sp in ("train", "val", "test"):
        p = feature_dir / f"{sp}.npz"
        if not p.exists():
            continue
        backup(p)
        d = np.load(p, allow_pickle=True)
        for i, rid in enumerate(d["ids"]):
            rid = str(rid)
            if rid in all_id_to_row:
                continue  # already seen (shouldn't happen, but be safe)
            all_id_to_row[rid] = {k: d[k][i] for k in d.files}

    missing_counts: dict[str, int] = {}
    for sp, records in new_records.items():
        rows = []
        missing = 0
        for r in records:
            row = all_id_to_row.get(r["id"])
            if row is None:
                missing += 1
                continue
            rows.append(row)
        missing_counts[sp] = missing

        if not rows:
            print(f"  [{sp}] no rows -- skipping cache write")
            continue

        keys = list(rows[0].keys())
        stacked: dict = {}
        for k in keys:
            if k in ("ids", "sources"):
                stacked[k] = np.array([row[k] for row in rows], dtype=object)
            elif rows[0][k].shape == ():     # scalar (label, style_ok)
                stacked[k] = np.array([row[k] for row in rows])
            else:                            # vector (nela, style, trace)
                stacked[k] = np.stack([row[k] for row in rows])

        out = feature_dir / f"{sp}.npz"
        np.savez_compressed(out, **stacked)
        print(f"  [{sp}] wrote {out.name}  ({len(rows)} records, {missing} not in old cache)")

    return missing_counts


def update_meta_json(feature_dir: Path) -> None:
    meta_path = feature_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for sp in ("train", "val", "test"):
        p = feature_dir / f"{sp}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        n = len(d["ids"])
        human = int((d["label"] == 0).sum())
        ai    = int((d["label"] == 1).sum())
        style_cov = float(d["style_ok"].mean()) if n else 0.0
        meta.setdefault("splits", {})[sp] = {
            "records": n, "human": human, "ai": ai,
            "style_coverage": round(style_cov, 4),
        }
    meta["resplit_note"] = (
        f"author-disjoint (USE) + prompt-disjoint (HC3), no exact-text leakage, "
        f">={MIN_GROUP_SIZE} records per author/prompt group, seed={SEED}"
    )
    meta_path.write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Stage 4 -- verify the result
# ---------------------------------------------------------------------------

def verify(new_records: dict) -> bool:
    print("\n=== verification ===")
    tr, va, te = new_records["train"], new_records["val"], new_records["test"]

    def sset(records, getter):
        return {getter(r) for r in records if getter(r) is not None}

    all_clean = True
    checks = [
        ("id",                lambda r: r["id"]),
        ("text",              lambda r: r["text"]),
        ("author_id",         lambda r: r.get("author_id")),
        ("hc3 prompt",        lambda r: r["prompt"] if r.get("source") == "hc3" else None),
        ("source_text_id",    lambda r: r.get("source_text_id")),
    ]
    for name, get in checks:
        s_tr, s_va, s_te = sset(tr, get), sset(va, get), sset(te, get)
        intersections = {
            "train&test": len(s_tr & s_te),
            "train&val":  len(s_tr & s_va),
            "val&test":   len(s_va & s_te),
        }
        nonzero = {k: v for k, v in intersections.items() if v}
        flag = "OK" if not nonzero else "LEAK"
        if nonzero:
            all_clean = False
        print(f"  [{flag}] {name:<16} {intersections}")

    print("\n  per-author group size distribution (after resplit):")
    by_id_all = {r["id"]: r for r in tr + va + te}
    for sp, records in new_records.items():
        sizes_use = defaultdict(int)
        sizes_hc3 = defaultdict(int)
        for r in records:
            gk = group_key(r, by_id_all)
            gt = gk[0]
            if gt == "use_author" and gk[1] is not None:
                sizes_use[gk[1]] += 1
            elif gt == "hc3_prompt":
                sizes_hc3[gk[1]] += 1
        below_use = sum(1 for s in sizes_use.values() if s < MIN_GROUP_SIZE)
        below_hc3 = sum(1 for s in sizes_hc3.values() if s < MIN_GROUP_SIZE)
        print(f"    {sp:<5}  USE authors: {len(sizes_use):>4}  (<{MIN_GROUP_SIZE}: {below_use})   "
              f"HC3 prompts: {len(sizes_hc3):>4}  (<{MIN_GROUP_SIZE}: {below_hc3})")

    return all_clean


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"REPO_ROOT   = {REPO_ROOT}")
    print(f"DATASET_DIR = {DATASET_DIR}")
    print(f"FEATURE_DIR = {FEATURE_DIR}\n")

    # 1. load all records from existing splits
    all_records: list[dict] = []
    for sp in ("train", "val", "test"):
        path = DATASET_DIR / f"{sp}.jsonl"
        rs = load_jsonl(path)
        print(f"  loaded {len(rs):>5} from {path.name}")
        all_records.extend(rs)
    print(f"  total: {len(all_records)} records\n")

    # 2. group and merge by text-sharing
    groups = build_groups(all_records)
    print(f"  raw groups (after text-merge): {len(groups)}")

    # 3. drop tiny author-based groups
    kept_groups, dropped = drop_singletons(groups, MIN_GROUP_SIZE)
    print(f"  dropped {dropped} records from author-based groups with <{MIN_GROUP_SIZE} members")
    print(f"  kept groups: {len(kept_groups)}\n")

    # 4. stratified atomic split
    new_records = stratify_and_split(kept_groups)
    print(f"  new sizes:")
    for sp, rs in new_records.items():
        human = sum(1 for r in rs if r.get("label") == "human")
        ai    = sum(1 for r in rs if r.get("label") == "ai")
        print(f"    {sp:<5} {len(rs):>5}  (human {human}, ai {ai})")

    # 5. write new jsonl + filter the cached features
    print("\n  writing new splits:")
    write_new_splits(new_records, DATASET_DIR)
    print("\n  filtering feature cache:")
    filter_feature_cache(new_records, FEATURE_DIR)
    update_meta_json(FEATURE_DIR)

    # 6. verify zero leakage
    clean = verify(new_records)
    print("\n" + ("ALL CLEAN -- no leakage detected" if clean else "LEAKAGE STILL PRESENT -- inspect logs"))
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
