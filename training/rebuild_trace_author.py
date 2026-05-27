"""One-off rebuild: switch the cached feature triples to TRACE author mode
and drop samples with no known author (HC3 humans).

Why a dedicated script instead of ``build_dataset.py --restart``?
----------------------------------------------------------------
The expensive part of the feature cache is the StyleDecipher pass: for any
sample outside a USE rewrite cluster, it generates rewrites via Ollama (slow,
multi-LLM, network-bound). Those rewrites are not persisted -- only the
resulting 10-dim StyleDecipher vector lives in the .npz. A ``--restart``
would re-run every Ollama call.

Since the only things changing are
  (a) the TRACE context (single -> author), and
  (b) the sample set (drop has_known_author=False),
we can keep the cached NELA + StyleDecipher columns verbatim and recompute
just TRACE for the kept samples. SBERT inference for TRACE is comparatively
cheap (a few minutes for the whole dataset on GPU).

Usage
-----
    python -m training.rebuild_trace_author
    python -m training.rebuild_trace_author --device cuda --out-dir data/features
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

# Line-buffer stdout so progress shows up live under nohup / tee / Bash background
# tasks on Windows, where the default fully-buffered stdout hides output until exit.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

# Bootstrap so `python training/rebuild_trace_author.py` works as well as `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from data.preprocessing import Dataset, TextSample
from training import paths
from training.extractor_pipeline import FeaturePipeline

ALL_SPLITS = ("train", "val", "test")
LABEL_TO_INDEX = {"human": 0, "ai": 1}

NELA_DIM = FeaturePipeline.NELA_DIM
STYLE_DIM = FeaturePipeline.STYLE_DIM
TRACE_DIM = FeaturePipeline.TRACE_DIM


def trace_human_siblings(sample: TextSample, ds: Dataset) -> list[TextSample]:
    """TRACE context for a sample under the human-only / no-source-leakage rule.

    Rule (see CLAUDE / project notes):
      * Anchor (sample) may be human or AI.
      * Context contains ONLY human texts from the same author.
      * If the anchor is an LLM rewrite, its source human text is excluded
        (otherwise the TRACE embedding would trivially reflect the very text
        the rewrite was derived from -- author-style leakage).
      * Non-USE AI samples (HC3 chatgpt, ArguGPT, RAID) have a generator-based
        author_id with no human siblings -- they get an empty context and
        fall back to single-text TRACE, which is the intended behaviour.
    """
    out: list[TextSample] = []
    for s in ds._by_author.get(sample.author_id, []):
        if s.record_id == sample.record_id:
            continue
        if s.label != "human":
            continue
        if sample.is_rewrite and s.record_id == sample.source_text_id:
            continue
        out.append(s)
    return out


def _load_cached_rows(npz_path: Path) -> dict:
    """Return ``{record_id: row}`` from an existing build_dataset .npz."""
    if not npz_path.exists():
        raise FileNotFoundError(f"missing {npz_path} -- run build_dataset.py first")
    data = np.load(npz_path, allow_pickle=True)
    has_style_ok = "style_ok" in data.files
    rows = {}
    for i, rid in enumerate(data["ids"]):
        rid = str(rid)
        rows[rid] = {
            "nela": data["nela"][i].astype(np.float32),
            "style": data["style"][i].astype(np.float32),
            "trace": data["trace"][i].astype(np.float32),
            "label": int(data["label"][i]),
            "style_ok": bool(data["style_ok"][i]) if has_style_ok else False,
            "source": str(data["sources"][i]),
        }
    return rows


def _save_features(path: Path, rows: dict, ordered_ids: list) -> None:
    """Atomically write features in ``ordered_ids`` order."""
    def _stack(key: str, dim: int) -> np.ndarray:
        if not ordered_ids:
            return np.zeros((0, dim), dtype=np.float32)
        return np.stack([rows[i][key] for i in ordered_ids]).astype(np.float32)

    payload = dict(
        nela=_stack("nela", NELA_DIM),
        style=_stack("style", STYLE_DIM),
        trace=_stack("trace", TRACE_DIM),
        label=np.array([rows[i]["label"] for i in ordered_ids], dtype=np.int64),
        style_ok=np.array([rows[i]["style_ok"] for i in ordered_ids], dtype=bool),
        ids=np.array(ordered_ids, dtype=object),
        sources=np.array([rows[i]["source"] for i in ordered_ids], dtype=object),
    )
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def rebuild_split(name: str, ds: Dataset, pipeline: FeaturePipeline,
                   in_path: Path, out_path: Path,
                   *, min_human_siblings: int = 0) -> dict:
    """Drop under-supported samples, recompute TRACE in author mode, keep nela/style.

    ``min_human_siblings`` enforces: every kept sample's author must contribute
    at least that many other human texts to use as TRACE context (after the
    rewrite-source exclusion). With ``min_human_siblings=2`` the kept set is
    USE-only and every TRACE embedding is genuinely author-aware.
    """
    cached = _load_cached_rows(in_path)
    print(f"  [{name}] loaded {len(cached)} cached rows from {in_path.name}")

    kept_ids: list[str] = []
    missing = 0
    dropped_no_author = 0
    dropped_low_siblings = 0
    drop_by_source: dict[str, int] = {}

    for sample in ds:
        if not sample.has_known_author:
            dropped_no_author += 1
            continue
        if sample.record_id not in cached:
            # Defensive: the input cache should cover every record in the split.
            missing += 1
            continue
        sib_count = len(trace_human_siblings(sample, ds))
        if sib_count < min_human_siblings:
            dropped_low_siblings += 1
            drop_by_source[sample.source] = drop_by_source.get(sample.source, 0) + 1
            continue
        kept_ids.append(sample.record_id)

    print(f"  [{name}] dropping {dropped_no_author} no-author samples")
    if min_human_siblings > 0:
        print(f"  [{name}] dropping {dropped_low_siblings} samples with <"
              f"{min_human_siblings} human siblings (per source: "
              f"{dict(sorted(drop_by_source.items(), key=lambda kv: -kv[1]))})")
    if missing:
        print(f"  [{name}] WARN: {missing} kept samples missing from input cache "
              f"(input cache may be stale)")

    print(f"  [{name}] recomputing TRACE (author mode) for {len(kept_ids)} samples ...")
    start = time.time()

    iter_ids = kept_ids
    if tqdm is not None:
        iter_ids = tqdm(
            kept_ids, desc=f"[{name}] TRACE", unit="rec",
            mininterval=2.0, dynamic_ncols=True,
            bar_format="{l_bar}{bar:24}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}, {rate_fmt}]",
        )

    failures = 0
    sib_count_total = 0
    sib_zero_count = 0
    for rid in iter_ids:
        sample = ds.get(rid)
        siblings = trace_human_siblings(sample, ds)
        sib_count_total += len(siblings)
        if not siblings:
            sib_zero_count += 1
        try:
            trace = pipeline.trace_features(sample, siblings=siblings)
            cached[rid]["trace"] = trace.astype(np.float32)
        except Exception as exc:
            failures += 1
            # Keep the old single-mode trace as a fallback -- don't zero it,
            # since that would silently drop the modality for that sample.
            print(f"  [{name}] {rid} TRACE re-extract failed ({exc}); keeping cached value")

    avg_sib = sib_count_total / max(len(kept_ids), 1)
    print(f"  [{name}] siblings stats: avg={avg_sib:.2f}, "
          f"zero-sibling-samples={sib_zero_count}/{len(kept_ids)} "
          "(those fell back to single-text TRACE)")

    _save_features(out_path, cached, kept_ids)
    elapsed = time.time() - start
    print(f"  [{name}] wrote {out_path} in {elapsed:.1f}s "
          f"({failures} TRACE failures)")

    labels = np.array([cached[i]["label"] for i in kept_ids], dtype=np.int64)
    style_ok = np.array([cached[i]["style_ok"] for i in kept_ids], dtype=bool)
    return {
        "records": int(len(kept_ids)),
        "human": int((labels == 0).sum()),
        "ai": int((labels == 1).sum()),
        "style_coverage": round(float(style_ok.mean()), 4) if len(style_ok) else 0.0,
        "dropped_no_author": dropped_no_author,
        "dropped_low_siblings": dropped_low_siblings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--in-dir", type=Path, default=paths.FEATURE_DIR,
                        help="directory holding the existing .npz cache")
    parser.add_argument("--out-dir", type=Path, default=paths.FEATURE_DIR,
                        help="directory to write the rebuilt cache (default: in-place)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="override dataset_ready_final location")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", default="all",
                        help='"all" (default) or comma list of train,val,test '
                             "(use to resume after a crash without redoing finished splits)")
    parser.add_argument("--min-human-siblings", type=int, default=2,
                        help="drop samples whose author contributes fewer than this many "
                             "human texts (after rewrite-source exclusion). Default 2 = USE-only "
                             "kept set, every TRACE embedding is genuinely author-aware.")
    args = parser.parse_args()

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset splits ...")
    all_datasets = {sp: Dataset.load(sp, args.data_dir) for sp in ALL_SPLITS}
    for sp, ds in all_datasets.items():
        print(f"  {sp:<5} {len(ds):>6} records")

    # TRACE only -- styledecipher_mode is irrelevant here; we just need a TRACE
    # extractor. Pass "off" so the StyleDecipher backend never initialises.
    pipeline = FeaturePipeline(
        styledecipher_mode="off",
        trace_context="author",
        device=args.device,
        seed=args.seed,
    )
    print(f"\nPipeline ready  (device={pipeline.device}, trace_context=author, "
          f"styledecipher=off-for-this-rebuild)\n")

    splits = list(ALL_SPLITS) if args.splits.strip().lower() == "all" else [s.strip() for s in args.splits.split(",") if s.strip()]
    bad = [s for s in splits if s not in ALL_SPLITS]
    if bad:
        raise SystemExit(f"unknown split(s): {bad}; pick from {ALL_SPLITS}")

    split_meta = {}
    for sp in splits:
        print(f"=== rebuilding '{sp}' ===")
        stats = rebuild_split(
            sp, all_datasets[sp], pipeline,
            in_path=in_dir / f"{sp}.npz",
            out_path=out_dir / f"{sp}.npz",
            min_human_siblings=args.min_human_siblings,
        )
        split_meta[sp] = stats
        print(f"  -> {stats}\n")

    # Update meta.json. Read the existing one so we keep styledecipher_mode etc.
    in_meta_path = in_dir / "meta.json"
    if in_meta_path.exists():
        meta = json.loads(in_meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}
    # When only some splits were rebuilt, merge into any pre-existing per-split stats
    # rather than replacing them.
    existing_splits = meta.get("splits", {}) or {}
    existing_splits.update(split_meta)
    meta.update({
        "dims": {"nela": NELA_DIM, "style": STYLE_DIM, "trace": TRACE_DIM},
        "trace_context": "author",
        "require_known_author": True,
        "min_human_siblings": args.min_human_siblings,
        "splits": existing_splits,
        "rebuild_note": "TRACE rebuilt in author mode; HC3 no-author humans dropped; "
                        "NELA + StyleDecipher (ollama) preserved from prior cache",
    })
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote updated meta.json to {out_dir}/meta.json")


if __name__ == "__main__":
    main()
