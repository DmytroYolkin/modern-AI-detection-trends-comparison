"""Build cached extractor-feature matrices for each dataset split.

Runs NELA + StyleDecipher + TRACE over every text in train/val/test and writes
the resulting feature triples to ``data/features/<split>.npz`` so that training
never has to touch the (slow) extractors again.

Each ``.npz`` holds:  nela (N,87)  style (N,10)  trace (N,128)
                      label (N,)  style_ok (N,)  ids (N,)  sources (N,)

Resumable
---------
The build checkpoints atomically every ``--checkpoint-every`` records. If it is
interrupted (Ctrl+C, kill, crash, power loss), just rerun the **same command**:
already-cached records are skipped and extraction continues where it stopped.
Use ``--restart`` to ignore existing caches and rebuild from scratch.

Usage
-----
    # full build, offline StyleDecipher from the dataset's rewrite clusters
    python -m training.build_dataset --splits all --styledecipher cached

    # full StyleDecipher coverage via a running Ollama server (slow -- resumable)
    python -m training.build_dataset --splits all --styledecipher ollama

    # quick smoke run on 50 records of the test split
    python -m training.build_dataset --splits test --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

# Bootstrap so `python training/build_dataset.py` works as well as `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional -- the bar is nice to have, not required
    tqdm = None

from data.preprocessing import Dataset
from training import paths
from training.extractor_pipeline import FeaturePipeline

LABEL_TO_INDEX = {"human": 0, "ai": 1}
ALL_SPLITS = ("train", "val", "test")
# Out-of-distribution evaluation splits. Each is wired in as a standalone split
# (name -> jsonl path) that is loaded only when explicitly requested via
# --splits. They are kept OUT of ALL_SPLITS so training flows never see them.
#
# - "arxiv"           : 2 574 rows -- 1 287 humans + 1 287 Claude-haiku rewrites.
# - "arxiv_humanized" : 3 861 rows -- the same 1 287 humans + 2 574 humanized AI
#                       rows (Adversarial Paraphrasing + TempParaphraser, 1 287
#                       each), produced by the pod-2 humanization pipeline.
OOD_SPLITS: dict[str, Path] = {
    "arxiv": paths.REPO_ROOT / "data" / "testing_dataset" / "arxiv_final" / "arxiv_merged.jsonl",
    "arxiv_humanized": paths.REPO_ROOT / "data" / "testing_dataset" / "arxiv_final" / "arxiv_eval_with_humanizers.jsonl",
}
# Back-compat aliases (other modules still import these by name).
ARXIV_SPLIT = "arxiv"
ARXIV_JSONL = OOD_SPLITS["arxiv"]

NELA_DIM = FeaturePipeline.NELA_DIM
STYLE_DIM = FeaturePipeline.STYLE_DIM
TRACE_DIM = FeaturePipeline.TRACE_DIM


# ===========================================================================
# Checkpoint I/O
# ===========================================================================

def _save_features(path: Path, rows: dict, samples: list) -> None:
    """Atomically write all cached feature rows to `path`.

    Rows are emitted in `samples` order (ids/sources stay row-aligned). The
    write goes to a temp file first, then `os.replace` swaps it in -- so an
    interruption mid-write never corrupts the existing checkpoint.
    """
    ordered = [s.record_id for s in samples if s.record_id in rows]

    def _stack(key: str, dim: int) -> np.ndarray:
        if not ordered:
            return np.zeros((0, dim), dtype=np.float32)
        return np.stack([rows[i][key] for i in ordered]).astype(np.float32)

    payload = dict(
        nela=_stack("nela", NELA_DIM),
        style=_stack("style", STYLE_DIM),
        trace=_stack("trace", TRACE_DIM),
        label=np.array([rows[i]["label"] for i in ordered], dtype=np.int64),
        style_ok=np.array([rows[i]["style_ok"] for i in ordered], dtype=bool),
        ids=np.array(ordered, dtype=object),
        sources=np.array([rows[i]["source"] for i in ordered], dtype=object),
    )
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def _load_rows(path: Path, keep_ids: set) -> dict:
    """Load a (possibly partial) checkpoint into a ``record_id -> row`` dict."""
    data = np.load(path, allow_pickle=True)
    has_style_ok = "style_ok" in data.files
    rows = {}
    for i, rid in enumerate(data["ids"]):
        rid = str(rid)
        if rid not in keep_ids:
            continue
        rows[rid] = {
            "nela": data["nela"][i].astype(np.float32),
            "style": data["style"][i].astype(np.float32),
            "trace": data["trace"][i].astype(np.float32),
            "label": int(data["label"][i]),
            "style_ok": bool(data["style_ok"][i]) if has_style_ok else False,
            "source": str(data["sources"][i]),
        }
    return rows


def _zero_row(sample) -> dict:
    """A safe all-zero feature row (used when a sample's extraction fails)."""
    return {
        "nela": np.zeros(NELA_DIM, dtype=np.float32),
        "style": np.zeros(STYLE_DIM, dtype=np.float32),
        "trace": np.zeros(TRACE_DIM, dtype=np.float32),
        "label": LABEL_TO_INDEX.get(sample.label, 0),
        "style_ok": False,
        "source": sample.source,
    }


def _stats(rows: dict, samples: list) -> dict:
    labels = np.array([rows[s.record_id]["label"]
                       for s in samples if s.record_id in rows], dtype=np.int64)
    style_ok = np.array([rows[s.record_id]["style_ok"]
                         for s in samples if s.record_id in rows], dtype=bool)
    return {
        "records": int(len(labels)),
        "human": int((labels == 0).sum()),
        "ai": int((labels == 1).sum()),
        "style_coverage": round(float(style_ok.mean()), 4) if len(style_ok) else 0.0,
    }


# ===========================================================================
# Per-split build
# ===========================================================================

def _human_sibling_count(sample, ds: Dataset) -> int:
    """Count same-author HUMAN texts available as TRACE context for `sample`.

    Mirrors `training.rebuild_trace_author.trace_human_siblings`:
      * same author_id
      * label == "human"
      * if sample is a rewrite, its source human text is excluded.
    """
    n = 0
    for s in ds._by_author.get(sample.author_id, []):
        if s.record_id == sample.record_id:
            continue
        if s.label != "human":
            continue
        if sample.is_rewrite and s.record_id == sample.source_text_id:
            continue
        n += 1
    return n


def build_split(name: str, ds: Dataset, pipeline: FeaturePipeline, out_path: Path,
                 *, limit: int | None = None, checkpoint_every: int = 25,
                 restart: bool = False, log_every: int = 100,
                 require_known_author: bool = False,
                 min_human_siblings: int = 0) -> dict:
    """Extract features for one split, with resumable checkpointing."""
    samples = list(ds)
    if require_known_author:
        before = len(samples)
        samples = [s for s in samples if s.has_known_author]
        dropped = before - len(samples)
        if dropped:
            print(f"  [{name}] --require-known-author dropped {dropped}/{before} samples "
                  f"with no known author (HC3 humans etc.)")
    if min_human_siblings > 0:
        before = len(samples)
        samples = [s for s in samples if _human_sibling_count(s, ds) >= min_human_siblings]
        dropped = before - len(samples)
        if dropped:
            print(f"  [{name}] --min-human-siblings {min_human_siblings} dropped "
                  f"{dropped}/{before} samples whose author had too few human texts "
                  "(non-USE AI + USE singletons)")
    if limit and limit < len(samples):
        # stride-sample so a smoke run spans the whole split (both classes)
        stride = max(1, len(samples) // limit)
        samples = samples[::stride][:limit]
    n = len(samples)
    keep_ids = {s.record_id for s in samples}

    rows: dict = {}
    if out_path.exists() and not restart:
        rows = _load_rows(out_path, keep_ids)
        if rows:
            print(f"  [{name}] resuming -- {len(rows)}/{n} records already cached")

    todo = [s for s in samples if s.record_id not in rows]
    if not todo:
        print(f"  [{name}] all {n} records already cached")
        _save_features(out_path, rows, samples)
        return _stats(rows, samples)

    print(f"  [{name}] extracting {len(todo)} of {n} records "
          f"(checkpoint every {checkpoint_every})")
    failures = 0
    processed = 0
    since_ckpt = 0
    start = time.time()

    # Live progress bar (tqdm) with elapsed/remaining/rate so two runs on
    # different hardware can be compared at a glance. Persistent milestone
    # lines every `log_every` records stay in tee'd logs for post-mortem
    # comparison even after the bar's carriage returns are gone.
    if tqdm is not None:
        bar = tqdm(
            total=len(todo), desc=f"[{name}]", unit="rec",
            mininterval=2.0, dynamic_ncols=True,
            bar_format="{l_bar}{bar:24}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}, {rate_fmt}]",
        )
        _emit = tqdm.write  # writes a line without breaking the bar
    else:
        bar = None
        _emit = print

    try:
        for sample in todo:
            try:
                feats = pipeline.extract(sample, siblings=ds.author_siblings(sample))
                rows[sample.record_id] = {
                    "nela": feats.nela, "style": feats.style, "trace": feats.trace,
                    "label": LABEL_TO_INDEX.get(sample.label, 0),
                    "style_ok": feats.style_ok, "source": sample.source,
                }
            except Exception as exc:  # a zero row is the safe fallback
                failures += 1
                _emit(f"  [{name}] {sample.record_id} failed: {exc}")
                rows[sample.record_id] = _zero_row(sample)

            processed += 1
            since_ckpt += 1
            if bar is not None:
                bar.update(1)
            if since_ckpt >= checkpoint_every:
                _save_features(out_path, rows, samples)
                since_ckpt = 0

            if processed % log_every == 0 or processed == len(todo):
                elapsed = time.time() - start
                rate = processed / max(elapsed, 1e-9)
                eta = (len(todo) - processed) / max(rate, 1e-9)
                _emit(f"  [{name}] {len(rows)}/{n} cached  "
                      f"({rate:5.2f} rec/s, elapsed {elapsed:5.0f}s, ETA {eta:5.0f}s)")
    except KeyboardInterrupt:
        if bar is not None:
            bar.close()
        _save_features(out_path, rows, samples)
        print(f"\n  [{name}] interrupted -- saved {len(rows)}/{n} records to {out_path}")
        print("  rerun the same command to resume.")
        raise SystemExit(130)
    finally:
        if bar is not None:
            bar.close()

    _save_features(out_path, rows, samples)
    if failures:
        print(f"  [{name}] completed with {failures} extraction failure(s)")
    return _stats(rows, samples)


# ===========================================================================
# CLI
# ===========================================================================

def _parse_splits(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(ALL_SPLITS)
    chosen = [s.strip() for s in value.split(",") if s.strip()]
    # Any registered OOD split is accepted alongside train/val/test.
    valid = (*ALL_SPLITS, *OOD_SPLITS)
    bad = [s for s in chosen if s not in valid]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown split(s): {bad}; pick from {valid}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", type=_parse_splits, default="all",
                        help='"all" or comma list of train,val,test (default: all)')
    parser.add_argument("--styledecipher", choices=("cached", "ollama", "off"),
                        default="cached", help="StyleDecipher rewrite source (default: cached)")
    parser.add_argument("--trace-context", choices=("single", "author"), default="single",
                        help="TRACE input: just the text, or the author's other texts too")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap records per split (smoke testing)")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="save progress every N records (default: 25)")
    parser.add_argument("--restart", action="store_true",
                        help="ignore existing caches and rebuild from scratch")
    parser.add_argument("--require-known-author", action="store_true",
                        help="skip samples whose author is unknown (e.g. HC3 humans)")
    parser.add_argument("--min-human-siblings", type=int, default=0,
                        help="drop samples whose author contributes fewer than this many "
                             "human texts (after rewrite-source exclusion). 2 keeps a USE-only "
                             "set where every TRACE embedding is genuinely author-aware.")
    parser.add_argument("--out-dir", type=Path, default=paths.FEATURE_DIR,
                        help="output directory for the .npz caches")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="override dataset_ready_final location")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Make `kill` (SIGTERM) flush a checkpoint too, not just Ctrl+C (SIGINT).
    try:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    except (ValueError, OSError, AttributeError):
        pass  # not supported on this platform / thread

    # Load every split: the rewrite clusters for StyleDecipher are built from
    # the full dataset, regardless of which splits we are extracting.
    # When only OOD split(s) are requested, we skip loading train/val/test
    # entirely -- each OOD split carries its own authors/rewrites and never
    # participates in clusters.
    requested_ood = [s for s in args.splits if s in OOD_SPLITS]
    ood_only = bool(requested_ood) and all(s in OOD_SPLITS for s in args.splits)
    print("Loading dataset splits ...")
    if ood_only:
        all_datasets = {s: Dataset.from_jsonl(OOD_SPLITS[s]) for s in requested_ood}
    else:
        all_datasets = {sp: Dataset.load(sp, args.data_dir) for sp in ALL_SPLITS}
        for s in requested_ood:
            all_datasets[s] = Dataset.from_jsonl(OOD_SPLITS[s])
    for sp, ds in all_datasets.items():
        print(f"  {sp:<5} {len(ds):>6} records")

    pipeline = FeaturePipeline.from_datasets(
        list(all_datasets.values()),
        styledecipher_mode=args.styledecipher,
        trace_context=args.trace_context,
        device=args.device,
        seed=args.seed,
    )
    print(f"\nPipeline ready  (device={pipeline.device}, "
          f"styledecipher={args.styledecipher}, trace_context={args.trace_context})\n")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    split_meta = {}
    for sp in args.splits:
        print(f"=== building '{sp}' ===")
        stats = build_split(
            sp, all_datasets[sp], pipeline, out_dir / f"{sp}.npz",
            limit=args.limit, checkpoint_every=args.checkpoint_every,
            restart=args.restart, require_known_author=args.require_known_author,
            min_human_siblings=args.min_human_siblings,
        )
        split_meta[sp] = stats
        print(f"  saved -> {out_dir / f'{sp}.npz'}  ({stats})\n")

    meta = {
        "dims": {"nela": NELA_DIM, "style": STYLE_DIM, "trace": TRACE_DIM},
        "styledecipher_mode": args.styledecipher,
        "trace_context": args.trace_context,
        "require_known_author": args.require_known_author,
        "min_human_siblings": args.min_human_siblings,
        "limit": args.limit,
        "seed": args.seed,
        "splits": split_meta,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote feature caches + meta.json to {out_dir}")


if __name__ == "__main__":
    main()
