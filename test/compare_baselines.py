"""Score external AI-text detectors on the same test split as the in-house models.

Reads raw records from ``data/dataset_ready_final/<split>.jsonl``, runs each
selected detector over them, and writes one
``models/baseline_results/<detector>.metrics.json`` per detector. The output
schema mirrors ``models/ready_models/*.metrics.json`` so the comparison
notebook in ``models/analysis.ipynb`` can consume both sets uniformly.

Usage
-----
    # list every registered detector and what it needs to run
    python -m test.compare_baselines --list

    # run a single detector
    python -m test.compare_baselines --detectors radar

    # run several, optionally cap the number of records (smoke test)
    python -m test.compare_baselines --detectors fast_detect_gpt,binoculars --limit 50

    # run everything
    python -m test.compare_baselines --detectors all

Skips and continues on per-detector errors so one missing dep / API key does
not abort the whole sweep; the failure is recorded in the output JSON for
that detector.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# Bootstrap so `python test/compare_baselines.py` works as well as `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_dotenv(path: Path) -> int:
    """Populate os.environ from a .env file. Existing values take priority.

    Tiny hand-rolled parser; supports ``KEY=VALUE``, ``export KEY=VALUE``,
    optional surrounding quotes, blank lines and ``#`` comments. Returns the
    number of new keys imported (excluding ones already in the environment).
    """
    if not path.exists():
        return 0
    imported = 0
    pattern = re.compile(r"(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.*)")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = pattern.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip().strip('"').strip("'")
        if key not in os.environ:
            os.environ[key] = val
            imported += 1
    return imported

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from test.baselines import REGISTRY, get_detector
from test.baselines.base import BaselineDetector, DetectorResult


LABEL_TO_INT = {"human": 0, "ai": 1}
INT_TO_LABEL = {0: "human", 1: "ai"}


def _iter_records(path: Path, limit: int | None) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _strict_fpr_threshold(scores: np.ndarray, y: np.ndarray, max_fpr: float = 0.01) -> dict:
    """Lowest threshold whose human-class FPR <= max_fpr; report TPR/precision at it.

    Mirrors the §6.2 evaluation regime in METHODOLOGY.md. Returns ``{}`` if no
    such threshold exists (e.g. detector is uninformative).
    """
    order = np.argsort(-scores)  # high-score first
    fp = 0
    tp = 0
    fn = int((y == 1).sum())
    tn = int((y == 0).sum())
    best: dict | None = None
    for idx in order:
        if y[idx] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        if fpr <= max_fpr:
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            best = {
                "threshold": float(scores[idx]),
                "test_tpr": tpr,
                "test_precision": prec,
                "test_fpr": fpr,
            }
        else:
            break
    return best or {}


def _summarise(detector_name: str, results: list[DetectorResult], records: list[dict],
               wall_seconds: float, describe: dict) -> dict:
    y_true = np.array([LABEL_TO_INT[r["label"]] for r in records], dtype=int)
    y_pred = np.array([LABEL_TO_INT[r.label] for r in results], dtype=int)
    scores = np.array([r.score_ai for r in results], dtype=float)

    out: dict = {
        "detector": detector_name,
        "n_records": len(records),
        "wall_seconds": round(wall_seconds, 2),
        "config": describe,
        # per-sample scores + ids so downstream eval can rebuild ROC / score-dist
        "ids": [str(r.get("id", i)) for i, r in enumerate(records)],
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_scores": scores.tolist(),
    }
    try:
        out["test"] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "roc_auc": (
                float(roc_auc_score(y_true, scores))
                if len(set(y_true.tolist())) > 1 else None
            ),
            "per_class": _per_class_report(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
            "strict_fpr_1pct": _strict_fpr_threshold(scores, y_true, max_fpr=0.01),
        }
    except Exception as e:  # pragma: no cover -- guard against degenerate detectors
        out["error"] = f"metrics computation failed: {e!r}"

    # per-source breakdown (matches evaluate.py output).
    by_source: dict[str, list[int]] = defaultdict(list)
    for rec, pred in zip(records, y_pred):
        by_source[str(rec.get("source", "unknown"))].append(
            int(pred == LABEL_TO_INT[rec["label"]])
        )
    out["per_source_accuracy"] = {
        src: {"accuracy": float(np.mean(v)), "n": len(v)}
        for src, v in sorted(by_source.items())
    }
    return out


def _per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rep = classification_report(
        y_true, y_pred, target_names=["human", "ai"], output_dict=True, zero_division=0,
    )
    return {
        "human": {k: float(rep["human"][k]) for k in ("precision", "recall", "f1-score")},
        "ai":    {k: float(rep["ai"][k])    for k in ("precision", "recall", "f1-score")},
    }


def _print_list() -> None:
    print(f"{'name':<20} {'requires'}")
    print(f"{'-'*20} {'-'*60}")
    for name in REGISTRY:
        cls = get_detector(name)
        print(f"{name:<20} {', '.join(cls.requires) or '-'}")


def _resolve_detectors(arg: str) -> list[str]:
    if arg == "all":
        return list(REGISTRY)
    names = [n.strip() for n in arg.split(",") if n.strip()]
    for n in names:
        if n not in REGISTRY:
            raise SystemExit(f"unknown detector {n!r}. --list to see all options")
    return names


def _score_with_checkpoint(detector: BaselineDetector, records: list[dict],
                           progress_path: Path, checkpoint_every: int
                           ) -> list[DetectorResult]:
    """Score ``records`` with per-record JSONL checkpoints for crash-resume.

    Reads ``progress_path`` first to discover which record ids are already
    scored. Skips those. For the remaining records, scores each via
    ``detector.predict_batch`` and appends a JSON line per record:
    ``{"id": str, "y_true": int, "y_pred": int, "y_score": float, "raw": {}}``.
    Flushes + ``fsync``'s the sidecar every ``checkpoint_every`` records so a
    crash loses at most ``checkpoint_every - 1`` records of work.

    The returned ``DetectorResult`` list is in the original ``records`` order
    (cached results interleaved with newly-scored ones).
    """
    done_by_id: dict[str, dict] = {}
    if progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    done_by_id[str(e["id"])] = e
                except (json.JSONDecodeError, KeyError):
                    pass  # tolerate a partial last line from the crash
        print(f"  --checkpoint-every: resuming with {len(done_by_id)} record(s) "
              f"cached from {progress_path.name}")

    results: list[DetectorResult | None] = [None] * len(records)
    pending_idx: list[int] = []
    for i, r in enumerate(records):
        rid = str(r.get("id", i))
        cached = done_by_id.get(rid)
        if cached is not None:
            results[i] = DetectorResult(
                score_ai=float(cached["y_score"]),
                label="ai" if int(cached["y_pred"]) == 1 else "human",
                raw=cached.get("raw", {}) or {},
            )
        else:
            pending_idx.append(i)

    if not pending_idx:
        print(f"  --checkpoint-every: all {len(records)} records already cached; no scoring needed")
        return [r for r in results if r is not None]  # type: ignore[list-item]

    print(f"  --checkpoint-every: {len(pending_idx)} record(s) still to score "
          f"(fsync every {checkpoint_every})")
    pending_records = [records[i] for i in pending_idx]
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    written_since_flush = 0
    with progress_path.open("a", encoding="utf-8") as f:
        for idx, result in zip(
            pending_idx,
            detector.predict_batch(rec["text"] for rec in pending_records),
        ):
            r = records[idx]
            results[idx] = result
            entry = {
                "id": str(r.get("id", idx)),
                "y_true": LABEL_TO_INT[r["label"]],
                "y_pred": LABEL_TO_INT[result.label],
                "y_score": float(result.score_ai),
                "raw": result.raw,
            }
            f.write(json.dumps(entry, default=str) + "\n")
            written_since_flush += 1
            if written_since_flush >= checkpoint_every:
                f.flush()
                try:
                    import os as _os
                    _os.fsync(f.fileno())
                except OSError:
                    pass
                written_since_flush = 0
        f.flush()
        try:
            import os as _os
            _os.fsync(f.fileno())
        except OSError:
            pass

    return [r for r in results if r is not None]  # type: ignore[list-item]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detectors", default="all",
                        help="comma-separated names or 'all' (default). See --list.")
    parser.add_argument("--list", action="store_true",
                        help="list every registered detector and exit")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--data-root", type=Path, default=Path("data/dataset_ready_final"))
    parser.add_argument("--input-jsonl", type=Path, default=None,
                        help="explicit JSONL file to score; overrides --split + --data-root. "
                             "Output filenames are prefixed with the input stem so multiple "
                             "input sets do not overwrite each other.")
    parser.add_argument("--output", type=Path, default=Path("models/baseline_results"))
    parser.add_argument("--limit", type=int, default=None,
                        help="only score the first N records (smoke test)")
    parser.add_argument("--detector-config", type=Path, default=None,
                        help="JSON file mapping detector name -> kwargs passed to its "
                             "constructor. Shape: {detector_name: {kwarg: value, ...}}. "
                             "Unknown detector names are silently ignored; a missing "
                             "file or empty config leaves the wrappers at their defaults.")
    parser.add_argument("--skip-if-exists", action="store_true",
                        help="resume mode: if a detector's output JSON already exists "
                             "at the target path AND parses to a successful run (has a "
                             "'test' block, no 'error' field), skip that detector. "
                             "Lets a pod-level rerun continue from where it stopped "
                             "without redoing already-completed detectors.")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="When > 0, write each record's per-record prediction "
                             "to a JSONL sidecar (<output>.progress.jsonl) as it is "
                             "produced, with an fsync every N records. Allows mid-run "
                             "resume after a crash: a re-launch with the same args "
                             "reads the sidecar, skips already-scored ids, and "
                             "continues. The sidecar is deleted after the final "
                             "metrics.json is written successfully. Required for slow "
                             "Ollama-bound detectors like raidar on flaky hardware.")
    args = parser.parse_args()

    if args.list:
        _print_list()
        return

    project_root = Path(__file__).resolve().parent.parent
    n_env = _load_dotenv(project_root / ".env")
    if n_env:
        print(f"Loaded {n_env} variable(s) from .env")

    # --input-jsonl overrides --split + --data-root; output files are then
    # prefixed with the input stem so multiple input sets don't collide.
    if args.input_jsonl is not None:
        split_path = args.input_jsonl
        out_prefix = f"{split_path.stem}__"
    else:
        split_path = args.data_root / f"{args.split}.jsonl"
        out_prefix = ""
    if not split_path.exists():
        raise SystemExit(f"split file not found: {split_path}")
    args.output.mkdir(parents=True, exist_ok=True)

    records = list(_iter_records(split_path, args.limit))
    print(f"Loaded {len(records)} records from {split_path}")

    # --detector-config: per-detector kwargs JSON. Unrecognised detector names
    # are ignored (allowing a single config file to cover the whole registry
    # even when only a subset is being run).
    detector_config: dict[str, dict] = {}
    if args.detector_config is not None:
        if not args.detector_config.exists():
            raise SystemExit(f"--detector-config file not found: {args.detector_config}")
        try:
            detector_config = json.loads(args.detector_config.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise SystemExit(f"--detector-config invalid JSON: {e}")
        if not isinstance(detector_config, dict):
            raise SystemExit("--detector-config root must be a JSON object "
                             "({detector_name: {kwarg: value, ...}})")
        print(f"Loaded detector config for: {sorted(detector_config.keys())}")

    for name in _resolve_detectors(args.detectors):
        out_path = args.output / f"{out_prefix}{name}.metrics.json"
        print(f"\n=== {name} ===")
        if args.skip_if_exists and out_path.exists():
            try:
                prev = json.loads(out_path.read_text(encoding="utf-8"))
                if "test" in prev and "error" not in prev:
                    print(f"  --skip-if-exists: {out_path.name} already complete; skipping")
                    continue
                print(f"  --skip-if-exists: existing {out_path.name} is incomplete; re-running")
            except Exception:
                print(f"  --skip-if-exists: existing {out_path.name} is unreadable; re-running")
        # ``out_path`` is e.g. ``...raidar.metrics.json``. We want
        # ``...raidar.progress.jsonl``, stripping the ``.metrics`` component.
        # ``Path.with_suffix`` only replaces the *last* suffix, so naively
        # using it produces ``...raidar.metrics.progress.jsonl`` (bug fixed
        # 2026-05-29 after RAIDAR clean was already in flight under the old
        # name; old in-flight files at the buggy path are still readable
        # because resume here goes through the same path-builder).
        progress_path = out_path.parent / out_path.name.replace(
            ".metrics.json", ".progress.jsonl",
        )
        try:
            cls = get_detector(name)
            overrides = detector_config.get(name, {}) or {}
            if overrides:
                print(f"  applying config overrides: {overrides}")
            detector: BaselineDetector = cls(**overrides)
            detector.load()
            start = time.time()
            if args.checkpoint_every and args.checkpoint_every > 0:
                results = _score_with_checkpoint(
                    detector, records, progress_path, args.checkpoint_every,
                )
            else:
                results = list(detector.predict_batch(r["text"] for r in records))
            wall = time.time() - start
            summary = _summarise(name, results, records, wall, detector.describe())
            detector.close()
        except Exception as e:
            traceback.print_exc()
            summary = {
                "detector": name,
                "n_records": len(records),
                "error": f"{type(e).__name__}: {e}",
            }

        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        # Clean up the progress sidecar once the final metrics.json is on disk,
        # but only when the run produced a real "test" block. If it errored
        # mid-way, keep the sidecar so the next launch can still resume.
        if (args.checkpoint_every and args.checkpoint_every > 0
                and progress_path.exists() and "test" in summary):
            try:
                progress_path.unlink()
            except OSError:
                pass
        if "test" in summary:
            t = summary["test"]
            print(f"  acc={t['accuracy']:.4f}  macro-F1={t['macro_f1']:.4f}  "
                  f"AUC={t['roc_auc']!s}  wall={summary['wall_seconds']}s")
        else:
            print(f"  FAILED -- see {out_path}")
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
