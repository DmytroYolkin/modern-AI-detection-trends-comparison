"""Apples-to-apples arxiv evaluation across all detectors.

Runs every in-house checkpoint in ``--in-house-models`` over the cached arxiv
feature .npz (clean and/or humanized), collects the matching baseline JSONs
written by ``test/compare_baselines.py``, computes a uniform metric block per
(detector, eval-set) pair, and writes summary CSVs / per-source tables /
confusion-matrix PNGs / ROC curves / score-distribution panels plus a
human-readable ``REPORT.md`` that ties it all together.

All inputs except ``--features-clean`` and ``--in-house-models`` are optional:
the report gracefully degrades, labelling missing pieces "not available".

Usage
-----
    python -m test.evaluate_arxiv \
        --features-clean      data/features/arxiv.npz \
        --features-humanized  data/features/arxiv_humanized.npz \
        --records-clean       data/testing_dataset/arxiv_final/arxiv_merged.jsonl \
        --records-humanized   data/testing_dataset/arxiv_final/arxiv_eval_with_humanizers.jsonl \
        --baselines-clean     models/baseline_results/arxiv_clean \
        --baselines-humanized models/baseline_results/arxiv_humanized \
        --in-house-models     models/ready_models \
        --out                 test/results/arxiv_eval/

The strict-FPR-at-1% computation mirrors
``test/compare_baselines.py::_strict_fpr_threshold`` (METHODOLOGY.md §6.2).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# Bootstrap so ``python test/evaluate_arxiv.py`` works as well as ``-m``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from training import paths
from training.classical import ClassicalClassifier, flatten_features
from training.feature_dataset import FeatureNormalizer, FusionFeatureDataset
from training.model import FusionClassifier


# -- colour palette (consistent with dataset-overview / analysis notebooks) ---
COLOR_HUMAN = "#1f77b4"      # blue
COLOR_AI = "#d62728"         # red
COLOR_HUMANIZED = "#ff7f0e"  # orange
DPI = 150
FIGSIZE = (8, 6)
FIGSIZE_THUMB = (3, 3)


def _load_dotenv(path: Path) -> int:
    """Lightweight .env loader, mirrors ``test/compare_baselines.py``."""
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


# -- IO ---------------------------------------------------------------------

def _iter_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_records(path: Path | None) -> list[dict] | None:
    if path is None or not path.exists():
        return None
    return list(_iter_records(path))


# -- in-house model discovery & prediction ----------------------------------

def _list_in_house(models_dir: Path) -> list[Path]:
    """All ``.pt`` (neural) and ``.joblib`` (classical) checkpoints in dir."""
    if not models_dir.exists():
        return []
    return sorted(
        list(models_dir.glob("*.pt")) + list(models_dir.glob("*.joblib"))
    )


def _detector_name_from_path(p: Path) -> str:
    """Stable detector name for a checkpoint, matches the analysis notebook."""
    if p.suffix == ".pt":
        return p.stem                       # e.g. "fusion_gating"
    # classical -- strip the "clf_" prefix so we get "xgboost", "nela_only" ...
    stem = p.stem
    if stem.startswith("clf_"):
        stem = stem[len("clf_"):]
    return f"classical_{stem}"


def _predict_neural(model, ds, device, batch_size: int = 256):
    import torch

    model.eval().to(device)
    preds, probs = [], []
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            end = min(start + batch_size, len(ds))
            nela = torch.from_numpy(ds.nela[start:end]).to(device)
            style = torch.from_numpy(ds.style[start:end]).to(device)
            trace = torch.from_numpy(ds.trace[start:end]).to(device)
            logits = model(nela, style, trace)
            preds.append(logits.argmax(dim=-1).cpu().numpy())
            probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def _predict_classical(clf, ds):
    X = flatten_features(ds.nela, ds.style, ds.trace)
    return clf.predict(X), clf.predict_proba(X)[:, 1]


def _run_in_house(model_path: Path, npz_path: Path, device: str) -> dict:
    """Predict over the cached features; return a results dict (preds, probs,
    label_names). Raises on failure -- caller catches and skips."""
    is_neural = model_path.suffix == ".pt"
    if is_neural:
        model, payload = FusionClassifier.load(model_path, map_location=device)
    else:
        model, payload = ClassicalClassifier.load(model_path)

    label_names = payload.get("label_names", ["human", "ai"])
    normalizer = FeatureNormalizer.from_state_dict(payload.get("normalizer"))

    ds = FusionFeatureDataset(npz_path)
    if normalizer is not None:
        ds.apply_normalizer(normalizer)

    y_true = ds.labels
    if is_neural:
        y_pred, y_score = _predict_neural(model, ds, device)
    else:
        y_pred, y_score = _predict_classical(model, ds)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
        "sources": ds.sources,
        "label_names": label_names,
    }


# -- baseline JSON ingestion ------------------------------------------------

def _load_baseline_dir(baseline_dir: Path | None) -> dict[str, dict]:
    """Return ``{detector_name: parsed_summary_json}``.

    Filenames follow ``test/compare_baselines.py`` -- either ``<name>.metrics.json``
    or ``<prefix>__<name>.metrics.json`` (multiple input sets to one dir).
    """
    out: dict[str, dict] = {}
    if baseline_dir is None or not baseline_dir.exists():
        return out
    for path in sorted(baseline_dir.glob("*.metrics.json")):
        stem = path.stem.replace(".metrics", "")
        # ``foo__bar`` -> baseline detector is ``bar``; the prefix encodes
        # which input file produced it, which we ignore here.
        if "__" in stem:
            stem = stem.split("__", 1)[1]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        out[f"baseline_{stem}"] = data
    return out


def _merge_humanized_with_clean_humans(
    humanized: dict[str, dict],
    clean: dict[str, dict],
    records_humanized: list[dict] | None,
) -> dict[str, dict]:
    """Reconstruct full-3861-row baseline payloads for the humanized eval-set.

    The humanized baseline sweep runs on ``arxiv_humanized_ai_only.jsonl`` (2574
    AI rows: Adv-P + Temp, no humans) to skip re-scoring the 1287 human texts
    that are byte-identical to the clean set. This function patches in the
    human-row scores from the clean run for each detector and returns a fresh
    ``humanized`` dict whose payloads carry the full ``ids / y_true / y_pred /
    y_scores / per_source_accuracy`` arrays at the humanized eval's canonical
    record order (taken from ``records_humanized``).

    If ``clean`` is empty / unsupplied, returns ``humanized`` unchanged after
    emitting a warning per detector. If the row counts mismatch (clean and
    humanized AI-only were run on different dataset versions), raises a
    ``SystemExit`` with a clear message.
    """
    if not humanized:
        return humanized
    # Build a canonical id order from --records-humanized so the merged vector
    # lines up with the rest of the evaluation (per-source breakdown, etc.).
    canonical_ids: list[str] | None = None
    if records_humanized:
        canonical_ids = [str(r.get("id", i)) for i, r in enumerate(records_humanized)]
    out: dict[str, dict] = {}
    for name, hum_payload in humanized.items():
        clean_payload = clean.get(name)
        hum_ids = hum_payload.get("ids")
        hum_y_true = hum_payload.get("y_true")
        hum_y_pred = hum_payload.get("y_pred")
        hum_y_scores = hum_payload.get("y_scores")
        # Only merge when both sides have the per-sample arrays.
        if (clean_payload is None
                or hum_ids is None or hum_y_true is None
                or hum_y_pred is None or hum_y_scores is None):
            if clean_payload is None:
                print(f"[merge] WARNING baseline {name}: no matching clean baseline "
                      "found; using AI-only humanized rows for the humanized "
                      "metrics. This produces an asymmetric metric block (only "
                      "the AI class is represented).")
            out[name] = hum_payload
            continue
        clean_ids = clean_payload.get("ids")
        clean_y_true = clean_payload.get("y_true")
        clean_y_pred = clean_payload.get("y_pred")
        clean_y_scores = clean_payload.get("y_scores")
        if (clean_ids is None or clean_y_true is None
                or clean_y_pred is None or clean_y_scores is None):
            print(f"[merge] WARNING baseline {name}: clean payload missing "
                  "per-sample arrays; using AI-only humanized rows only.")
            out[name] = hum_payload
            continue

        # Defensive check: the AI-only humanized run and the clean run must both
        # have been scored on the same 2574-row dataset (humans + AI in clean;
        # AI only in humanized AI-only). The AI-only humanized file has the
        # same number of rows as the clean file by design.
        clean_n = len(clean_ids)
        hum_n = len(hum_ids)
        if clean_n != hum_n:
            raise SystemExit(
                f"baseline {name}: clean has {clean_n} rows but humanized "
                f"AI-only has {hum_n}; can't merge -- were they run on the "
                "same dataset version?"
            )
        # Pull humans (y_true == 0) out of the clean payload.
        human_rows = [
            (clean_ids[i], int(clean_y_true[i]), int(clean_y_pred[i]),
             float(clean_y_scores[i]))
            for i in range(clean_n)
            if int(clean_y_true[i]) == 0
        ]
        # All humanized-AI-only rows are AI (label==1); keep them as-is.
        ai_rows = [
            (str(hum_ids[i]), int(hum_y_true[i]), int(hum_y_pred[i]),
             float(hum_y_scores[i]))
            for i in range(hum_n)
        ]
        merged = human_rows + ai_rows
        # Reorder to the canonical humanized eval-set order when available.
        if canonical_ids is not None:
            by_id = {r[0]: r for r in merged}
            try:
                merged = [by_id[cid] for cid in canonical_ids]
            except KeyError as e:
                missing = e.args[0]
                print(f"[merge] WARNING baseline {name}: id {missing!r} from "
                      "--records-humanized not found in merged baseline rows; "
                      "falling back to (humans-then-AI) order.")
                merged = human_rows + ai_rows

        m_ids   = [r[0] for r in merged]
        m_true  = [r[1] for r in merged]
        m_pred  = [r[2] for r in merged]
        m_score = [r[3] for r in merged]

        # Recompute the aggregate metric block on the merged vector so the
        # downstream `_baseline_to_metrics` ingestion picks up real values
        # rather than the AI-only payload.
        y_true_arr = np.asarray(m_true, dtype=int)
        y_pred_arr = np.asarray(m_pred, dtype=int)
        y_score_arr = np.asarray(m_score, dtype=float)
        try:
            roc = float(roc_auc_score(y_true_arr, y_score_arr)) \
                if len(set(m_true)) > 1 else None
        except Exception:
            roc = None
        merged_test = {
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "macro_f1": float(f1_score(y_true_arr, y_pred_arr,
                                       average="macro", zero_division=0)),
            "roc_auc": roc,
            "per_class": _per_class_report(y_true_arr, y_pred_arr),
            "confusion_matrix": confusion_matrix(
                y_true_arr, y_pred_arr, labels=[0, 1]).tolist(),
            "strict_fpr_1pct": _strict_fpr_threshold(y_score_arr, y_true_arr, 0.01),
        }
        # per-source map: build from the humanized records JSONL when available,
        # otherwise fall back to the union of clean + humanized records.
        per_source: dict[str, dict] = {}
        if records_humanized:
            by_id_rec = {str(r.get("id", i)): r
                         for i, r in enumerate(records_humanized)}
            from collections import defaultdict as _dd
            bs: dict[str, list[int]] = _dd(list)
            for rid, t, p, _s in zip(m_ids, m_true, m_pred, m_score):
                rec = by_id_rec.get(rid)
                src = str(rec.get("source", "unknown")) if rec else "unknown"
                bs[src].append(int(t == p))
            per_source = {
                src: {"accuracy": float(np.mean(v)), "n": len(v)}
                for src, v in sorted(bs.items())
            }

        merged_payload = dict(hum_payload)
        merged_payload["n_records"] = len(merged)
        merged_payload["ids"] = m_ids
        merged_payload["y_true"] = m_true
        merged_payload["y_pred"] = m_pred
        merged_payload["y_scores"] = m_score
        merged_payload["test"] = merged_test
        if per_source:
            merged_payload["per_source_accuracy"] = per_source
        out[name] = merged_payload
        print(f"[merge] baseline {name}: merged {len(human_rows)} clean-human + "
              f"{len(ai_rows)} humanized-AI -> {len(merged)} rows")
    return out


# -- metric computation -----------------------------------------------------

def _strict_fpr_threshold(scores: np.ndarray, y: np.ndarray, max_fpr: float = 0.01) -> dict:
    """Mirror of ``test/compare_baselines.py::_strict_fpr_threshold``."""
    order = np.argsort(-scores)
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


def _per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rep = classification_report(
        y_true, y_pred, target_names=["human", "ai"], output_dict=True, zero_division=0,
    )
    return {
        "human": {k: float(rep["human"][k]) for k in ("precision", "recall", "f1-score")},
        "ai":    {k: float(rep["ai"][k])    for k in ("precision", "recall", "f1-score")},
    }


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None,
                     sources: np.ndarray | list[str] | None) -> dict:
    out: dict = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": _per_class_report(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_score is not None and len(set(y_true.tolist())) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["roc_auc"] = None
        out["strict_fpr_1pct"] = _strict_fpr_threshold(np.asarray(y_score),
                                                       np.asarray(y_true), 0.01)
    else:
        out["roc_auc"] = None
        out["strict_fpr_1pct"] = {}

    if sources is not None and len(sources) == len(y_true):
        by_source: dict[str, list[int]] = defaultdict(list)
        for src, t, p in zip(sources, y_true, y_pred):
            by_source[str(src)].append(int(t == p))
        out["per_source_accuracy"] = {
            src: {"accuracy": float(np.mean(v)), "n": len(v)}
            for src, v in sorted(by_source.items())
        }
    return out


def _baseline_to_metrics(payload: dict) -> dict | None:
    """Reshape a baseline JSON into the same shape as ``_compute_metrics``.

    Returns ``None`` when the baseline run errored (no ``test`` block).
    """
    test = payload.get("test")
    if not test:
        return None
    out = {
        "n": payload.get("n_records", 0),
        "accuracy": test.get("accuracy"),
        "macro_f1": test.get("macro_f1"),
        "roc_auc": test.get("roc_auc"),
        "per_class": test.get("per_class", {}),
        "confusion_matrix": test.get("confusion_matrix", [[0, 0], [0, 0]]),
        "strict_fpr_1pct": test.get("strict_fpr_1pct", {}),
        "per_source_accuracy": payload.get("per_source_accuracy", {}),
    }
    return out


# -- plotting ---------------------------------------------------------------

def _setup_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    return plt, sns


def _plot_confusion(cm: list[list[int]], title: str, out_path: Path) -> None:
    plt, sns = _setup_plotting()
    cm_arr = np.asarray(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = cm_arr / row_sums

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.heatmap(
        norm, annot=cm_arr.astype(int), fmt="d", cmap="Blues", cbar=True,
        xticklabels=["human", "ai"], yticklabels=["human", "ai"],
        vmin=0.0, vmax=1.0, ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def _plot_roc_panel(detectors: list[tuple[str, np.ndarray, np.ndarray]],
                    title: str, out_path: Path) -> None:
    """detectors: list of ``(name, y_true, y_score)`` -- skips degenerate ones."""
    plt, _ = _setup_plotting()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # Sort by AUC descending so the strongest curve sits at the top of legend.
    rows = []
    for name, y, s in detectors:
        if s is None or len(set(y.tolist())) < 2:
            continue
        try:
            fpr, tpr, _ = roc_curve(y, s)
            auc = roc_auc_score(y, s)
            rows.append((auc, name, fpr, tpr))
        except Exception:
            continue
    rows.sort(key=lambda r: -r[0])
    for auc, name, fpr, tpr in rows:
        if name.startswith("baseline_"):
            linestyle, family = ":", "baseline"
        elif name.startswith("classical_"):
            linestyle, family = "--", "classical"
        else:
            linestyle, family = "-", "fusion"
        ax.plot(fpr, tpr, linestyle=linestyle,
                label=f"[{family}] {name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title(title)
    if rows:
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def _plot_score_distribution(name: str,
                             clean: tuple[np.ndarray, np.ndarray] | None,
                             humanized: tuple[np.ndarray, np.ndarray] | None,
                             out_path: Path) -> None:
    """clean / humanized: ``(y_true, y_score)`` or None."""
    plt, sns = _setup_plotting()
    panels = [(t, d) for t, d in [("clean", clean), ("humanized", humanized)] if d is not None]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (label, (y, s)) in zip(axes, panels):
        if s is None:
            ax.set_visible(False)
            continue
        human_scores = s[np.asarray(y) == 0]
        ai_scores = s[np.asarray(y) == 1]
        if len(human_scores):
            ax.hist(human_scores, bins=30, alpha=0.6, color=COLOR_HUMAN, label="human")
        if len(ai_scores):
            ax.hist(ai_scores, bins=30, alpha=0.6, color=COLOR_AI, label="ai")
        ax.set_title(f"{name} -- {label}")
        ax.set_xlabel("score_ai")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def _plot_tpr_drop_per_humanizer(records: list[dict],
                                 humanized_by_detector: dict[str, dict],
                                 out_path: Path) -> None:
    """One grouped bar per detector: AI-class accuracy on each source.

    ``humanized_by_detector[name] -> metrics dict`` (must have
    ``per_source_accuracy`` populated).
    """
    plt, sns = _setup_plotting()
    if not humanized_by_detector:
        return
    sources = sorted({
        s for d in humanized_by_detector.values()
        for s in d.get("per_source_accuracy", {}).keys()
    })
    if not sources:
        return
    detectors = sorted(humanized_by_detector.keys())
    n_src = len(sources)
    n_det = len(detectors)
    x = np.arange(n_det)
    width = 0.8 / max(n_src, 1)

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * n_det + 2), 6))
    palette = sns.color_palette("tab10", n_colors=n_src)
    for i, src in enumerate(sources):
        vals = []
        for d in detectors:
            entry = humanized_by_detector[d].get("per_source_accuracy", {}).get(src, {})
            vals.append(entry.get("accuracy", np.nan))
        ax.bar(x + (i - n_src / 2 + 0.5) * width, vals, width=width,
               label=src, color=palette[i])
    ax.set_xticks(x)
    ax.set_xticklabels(detectors, rotation=45, ha="right")
    ax.set_ylabel("per-source accuracy (humanized eval-set)")
    ax.set_title("Per-humanizer attack success (higher = detector resisted attack)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


# -- summary CSV / markdown writers -----------------------------------------

def _write_summary_csv(per_detector: dict[str, dict], out_path: Path) -> None:
    """One row per detector with the headline metrics."""
    fields = ["detector", "n", "accuracy", "macro_f1", "roc_auc",
              "human_f1", "ai_f1", "strict_fpr_tpr"]
    rows = []
    for name in sorted(per_detector.keys()):
        m = per_detector[name]
        if m is None:
            rows.append({"detector": name, "n": 0, "accuracy": None,
                         "macro_f1": None, "roc_auc": None,
                         "human_f1": None, "ai_f1": None, "strict_fpr_tpr": None})
            continue
        rows.append({
            "detector": name,
            "n": m.get("n"),
            "accuracy": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"),
            "roc_auc": m.get("roc_auc"),
            "human_f1": m.get("per_class", {}).get("human", {}).get("f1-score"),
            "ai_f1":    m.get("per_class", {}).get("ai",    {}).get("f1-score"),
            "strict_fpr_tpr": m.get("strict_fpr_1pct", {}).get("test_tpr"),
        })
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_per_source_csv(per_detector: dict[str, dict], out_path: Path) -> None:
    """One row per detector; columns are sources seen anywhere."""
    sources = sorted({
        s for m in per_detector.values() if m
        for s in m.get("per_source_accuracy", {}).keys()
    })
    fields = ["detector", *sources]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name in sorted(per_detector.keys()):
            m = per_detector[name] or {}
            row = {"detector": name}
            for s in sources:
                e = m.get("per_source_accuracy", {}).get(s)
                row[s] = e["accuracy"] if e else None
            w.writerow(row)


def _write_combined_markdown(clean: dict[str, dict], humanized: dict[str, dict],
                             out_path: Path) -> None:
    """Side-by-side clean | humanized | delta, sorted by clean macroF1."""
    names = sorted(set(clean.keys()) | set(humanized.keys()))
    def _f1(d, n):
        m = d.get(n)
        return m.get("macro_f1") if m else None

    rows = []
    for n in names:
        c = _f1(clean, n)
        h = _f1(humanized, n)
        delta = (h - c) if (c is not None and h is not None) else None
        rows.append((n, c, h, delta))
    # Sort by clean macroF1 desc; missing values fall to the bottom.
    rows.sort(key=lambda r: (-(r[1] if r[1] is not None else -1)))

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else "-"

    lines = [
        "| detector | clean macroF1 | humanized macroF1 | Δ |",
        "|---|---:|---:|---:|",
    ]
    for n, c, h, d in rows:
        lines.append(f"| {n} | {_fmt(c)} | {_fmt(h)} | {_fmt(d)} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -- REPORT.md -------------------------------------------------------------

def _report_md(out_dir: Path,
               clean: dict[str, dict],
               humanized: dict[str, dict],
               records_clean: list[dict] | None,
               records_humanized: list[dict] | None,
               available_inputs: dict,
               cmd_str: str) -> None:
    md: list[str] = []
    md.append("# arxiv evaluation report")
    md.append("")
    md.append(f"Generated by `test/evaluate_arxiv.py`.")
    md.append("")

    # --- 1. Dataset description --------------------------------------------
    md.append("## 1. Dataset description")
    md.append("")
    if records_clean is not None:
        labels = [r.get("label") for r in records_clean]
        srcs = sorted({r.get("source", "?") for r in records_clean})
        md.append(f"- **Clean** (`{available_inputs.get('records_clean','?')}`): "
                  f"{len(records_clean)} rows, "
                  f"{labels.count('human')} human / {labels.count('ai')} ai, "
                  f"sources: {', '.join(srcs)}.")
    elif available_inputs.get("features_clean"):
        md.append(f"- **Clean**: features `{available_inputs['features_clean']}` "
                  f"available, but records JSONL not provided.")
    else:
        md.append("- **Clean**: not available.")

    if records_humanized is not None:
        labels = [r.get("label") for r in records_humanized]
        srcs = sorted({r.get("source", "?") for r in records_humanized})
        md.append(f"- **Humanized** (`{available_inputs.get('records_humanized','?')}`): "
                  f"{len(records_humanized)} rows, "
                  f"{labels.count('human')} human / {labels.count('ai')} ai, "
                  f"sources: {', '.join(srcs)}.")
    elif available_inputs.get("features_humanized"):
        md.append(f"- **Humanized**: features `{available_inputs['features_humanized']}` "
                  f"available, but records JSONL not provided.")
    else:
        md.append("- **Humanized**: not available.")
    md.append("")

    # --- 2. Leaderboard ----------------------------------------------------
    md.append("## 2. Leaderboard (top 10 by clean macroF1)")
    md.append("")
    if not clean:
        md.append("_Clean evaluation set not available -- leaderboard skipped._")
    else:
        rows = []
        for n, m in clean.items():
            if not m:
                continue
            h = humanized.get(n)
            c_f1 = m.get("macro_f1")
            h_f1 = h.get("macro_f1") if h else None
            delta = (h_f1 - c_f1) if (h_f1 is not None and c_f1 is not None) else None
            rows.append((n, c_f1, h_f1, delta,
                         m.get("accuracy"), m.get("roc_auc"),
                         m.get("strict_fpr_1pct", {}).get("test_tpr")))
        rows.sort(key=lambda r: -(r[1] if r[1] is not None else -1))
        rows = rows[:10]
        md.append("| # | detector | clean macroF1 | clean acc | clean AUC | "
                  "strict-FPR@1% TPR | humanized macroF1 | Δ |")
        md.append("|---:|---|---:|---:|---:|---:|---:|---:|")
        def _f(v): return f"{v:.4f}" if isinstance(v, float) else "-"
        for i, (n, cF1, hF1, d, acc, auc, tpr) in enumerate(rows, 1):
            md.append(f"| {i} | {n} | {_f(cF1)} | {_f(acc)} | {_f(auc)} | "
                      f"{_f(tpr)} | {_f(hF1)} | {_f(d)} |")
    md.append("")

    # --- 3. Per-humanizer attack success -----------------------------------
    md.append("## 3. Per-humanizer attack success")
    md.append("")
    if not humanized:
        md.append("_Humanized evaluation set not available._")
    else:
        md.append("Per-source accuracy on the humanized set. Lower accuracy on "
                  "an `arxiv_humanized_*` source = stronger attack against that "
                  "detector. See `per_source_humanized.csv` for the full table.")
        md.append("")
        md.append("![per-humanizer attack success](tpr_drop_per_humanizer.png)")
    md.append("")

    # --- 4. Confusion matrices --------------------------------------------
    md.append("## 4. Confusion matrices")
    md.append("")
    names = sorted(set(clean.keys()) | set(humanized.keys()))
    if not names:
        md.append("_No detectors produced metrics._")
    else:
        md.append("| detector | clean | humanized |")
        md.append("|---|---|---|")
        for n in names:
            cell_c = f"![clean]({Path('confusion_matrices') / f'clean__{n}.png'})" \
                if n in clean else "_n/a_"
            cell_h = f"![humanized]({Path('confusion_matrices') / f'humanized__{n}.png'})" \
                if n in humanized else "_n/a_"
            md.append(f"| {n} | {cell_c} | {cell_h} |")
    md.append("")

    # --- 5. ROC curves -----------------------------------------------------
    md.append("## 5. ROC curves")
    md.append("")
    if clean:
        md.append("**Clean**")
        md.append("")
        md.append("![ROC clean](roc_curves/clean.png)")
        md.append("")
    if humanized:
        md.append("**Humanized**")
        md.append("")
        md.append("![ROC humanized](roc_curves/humanized.png)")
        md.append("")

    # --- 6. Score-distribution panels --------------------------------------
    md.append("## 6. Score distributions")
    md.append("")
    if not names:
        md.append("_No detectors produced metrics._")
    else:
        for n in names:
            md.append(f"### {n}")
            md.append(f"![{n}](score_distributions/{n}.png)")
            md.append("")
    md.append("")

    # --- 7. Anomalies ------------------------------------------------------
    md.append("## 7. Anomalies / automatic flags")
    md.append("")
    flags: list[str] = []
    for n, m in clean.items():
        if not m:
            continue
        h = humanized.get(n)
        c_f1 = m.get("macro_f1")
        if h and isinstance(c_f1, float) and isinstance(h.get("macro_f1"), float):
            drop = c_f1 - h["macro_f1"]
            if drop * 100 > 20:
                flags.append(f"- **{n}**: macroF1 drops {drop * 100:.1f} pp "
                             f"from clean ({c_f1:.4f}) to humanized "
                             f"({h['macro_f1']:.4f}) -- vulnerable to humanization.")
        if isinstance(m.get("roc_auc"), float) and m["roc_auc"] < 0.6:
            flags.append(f"- **{n}**: clean ROC-AUC = {m['roc_auc']:.3f} "
                         f"(<0.6) -- detector broken or wrong score direction.")
        if not m.get("strict_fpr_1pct"):
            flags.append(f"- **{n}**: strict-FPR-at-1% threshold could not be set "
                         f"(scores not informative enough or eval set degenerate).")
    if flags:
        md.extend(flags)
    else:
        md.append("_No automatic flags raised._")
    md.append("")

    # --- 8. Methodology footnote -------------------------------------------
    md.append("## 8. Methodology footnote")
    md.append("")
    md.append("- **accuracy** -- fraction of records whose predicted label matches the gold.")
    md.append("- **macroF1** -- unweighted mean of per-class F1 (human, ai).")
    md.append("- **ROC-AUC** -- area under the ROC of the ai-class score; "
              "`null` when one class is missing from the eval set.")
    md.append("- **strict-FPR@1% TPR** -- TPR at the lowest threshold that keeps "
              "human-class false-positive rate ≤ 1% (METHODOLOGY.md §6.2). "
              "Empty when no such threshold exists.")
    md.append("- **per-source accuracy** -- accuracy restricted to records whose "
              "`source` field equals the given corpus; reveals per-humanizer attack success.")
    md.append("")
    md.append("Reproducible command:")
    md.append("```")
    md.append(cmd_str)
    md.append("```")
    md.append("")

    (out_dir / "REPORT.md").write_text("\n".join(md), encoding="utf-8")


# -- driver -----------------------------------------------------------------

def _collect_results(features_path: Path | None,
                     records_path: Path | None,
                     baselines_dir: Path | None,
                     in_house_dir: Path,
                     device: str,
                     log_prefix: str,
                     baselines_override: dict[str, dict] | None = None,
                     ) -> tuple[dict[str, dict], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Return ``(per_detector_metrics, per_detector_scores)`` for one eval set.

    ``per_detector_scores[name] = (y_true, y_score)`` -- used for ROC + score
    distributions. Baselines without raw scores are absent from this dict.

    When ``baselines_override`` is provided, it is used in lieu of reading
    ``baselines_dir``. This is how the humanized-eval driver injects the
    merged (clean-humans + humanized-AI) payloads.
    """
    metrics: dict[str, dict] = {}
    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # in-house --------------------------------------------------------------
    if features_path is not None and features_path.exists():
        print(f"[{log_prefix}] in-house: features = {features_path}")
        for ckpt in _list_in_house(in_house_dir):
            name = _detector_name_from_path(ckpt)
            try:
                r = _run_in_house(ckpt, features_path, device)
            except Exception as e:
                print(f"[{log_prefix}] FAILED {name}: {type(e).__name__}: {e}")
                traceback.print_exc()
                metrics[name] = None
                continue
            # Override sources from --records-* when provided so the per-source
            # breakdown matches the records JSONL (cache may not carry it).
            sources = r["sources"]
            m = _compute_metrics(r["y_true"], r["y_pred"], r["y_score"], sources)
            metrics[name] = m
            raw[name] = (np.asarray(r["y_true"]), np.asarray(r["y_score"]))
            f1 = m.get("macro_f1")
            print(f"[{log_prefix}]   {name:<35} acc={m['accuracy']:.4f}  macroF1={f1:.4f}")
    else:
        if features_path is not None:
            print(f"[{log_prefix}] features {features_path} does not exist -- skipping in-house run.")
        else:
            print(f"[{log_prefix}] no features path provided -- skipping in-house run.")

    # baselines -------------------------------------------------------------
    if baselines_override is not None:
        baselines = baselines_override
        print(f"[{log_prefix}] using {len(baselines)} caller-supplied baseline payload(s)")
    else:
        baselines = _load_baseline_dir(baselines_dir)
        if baselines:
            print(f"[{log_prefix}] loaded {len(baselines)} baseline JSON(s) from {baselines_dir}")
    for name, payload in baselines.items():
        m = _baseline_to_metrics(payload)
        metrics[name] = m
        # newer `compare_baselines.py` runs persist per-sample arrays; when
        # present, lift them into `raw` so the baseline joins ROC + score-dist.
        y_true_b = payload.get("y_true")
        y_scores_b = payload.get("y_scores")
        if y_true_b is not None and y_scores_b is not None \
                and len(y_true_b) == len(y_scores_b):
            raw[name] = (np.asarray(y_true_b, dtype=int),
                         np.asarray(y_scores_b, dtype=float))
    return metrics, raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-clean", type=Path, required=True)
    parser.add_argument("--features-humanized", type=Path, default=None)
    parser.add_argument("--records-clean", type=Path, default=None)
    parser.add_argument("--records-humanized", type=Path, default=None)
    parser.add_argument("--baselines-clean", type=Path, default=None)
    parser.add_argument("--baselines-humanized", type=Path, default=None)
    parser.add_argument("--in-house-models", type=Path,
                        default=paths.READY_MODELS_DIR)
    parser.add_argument("--out", type=Path,
                        default=Path("test/results/arxiv_eval"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    _load_dotenv(project_root / ".env")

    device = paths.resolve_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "confusion_matrices").mkdir(exist_ok=True)
    (args.out / "roc_curves").mkdir(exist_ok=True)
    (args.out / "score_distributions").mkdir(exist_ok=True)

    if not args.features_clean.exists():
        raise SystemExit(f"--features-clean not found: {args.features_clean}")

    # records (used only for the dataset-description section + per-source map)
    records_clean = _load_records(args.records_clean)
    records_humanized = _load_records(args.records_humanized)

    # === collect ===========================================================
    clean_metrics, clean_scores = _collect_results(
        args.features_clean, args.records_clean, args.baselines_clean,
        args.in_house_models, device, "clean",
    )

    # For the humanized baseline runs we score only the 2574 AI rows
    # (Adv-P + Temp); the 1287 human rows are byte-identical to the clean set
    # and reused from there. Merge the per-sample arrays here, before computing
    # the humanized metric block, so the humanized leaderboard reflects the
    # full 3861-row evaluation.
    humanized_baselines_override: dict[str, dict] | None = None
    if args.baselines_humanized is not None:
        humanized_raw = _load_baseline_dir(args.baselines_humanized)
        clean_raw = _load_baseline_dir(args.baselines_clean) if args.baselines_clean else {}
        if humanized_raw and clean_raw:
            humanized_baselines_override = _merge_humanized_with_clean_humans(
                humanized_raw, clean_raw, records_humanized,
            )
        elif humanized_raw and not clean_raw:
            print("[merge] WARNING --baselines-clean not supplied; the humanized "
                  "baseline metrics will be computed on the AI-only rows only "
                  "(asymmetric -- only the AI class is represented).")
            humanized_baselines_override = humanized_raw

    humanized_metrics, humanized_scores = _collect_results(
        args.features_humanized, args.records_humanized, args.baselines_humanized,
        args.in_house_models, device, "humanized",
        baselines_override=humanized_baselines_override,
    )

    # === per-detector summary CSV/markdown ================================
    if clean_metrics:
        _write_summary_csv(clean_metrics, args.out / "summary_clean.csv")
        _write_per_source_csv(clean_metrics, args.out / "per_source_clean.csv")
    if humanized_metrics:
        _write_summary_csv(humanized_metrics, args.out / "summary_humanized.csv")
        _write_per_source_csv(humanized_metrics, args.out / "per_source_humanized.csv")
    _write_combined_markdown(clean_metrics, humanized_metrics,
                             args.out / "summary_combined.md")

    # === plots: confusion matrices =========================================
    for name, m in clean_metrics.items():
        if m and m.get("confusion_matrix"):
            try:
                _plot_confusion(m["confusion_matrix"],
                                f"{name} -- clean",
                                args.out / "confusion_matrices" / f"clean__{name}.png")
            except Exception as e:
                print(f"[plot] confusion matrix failed for {name} (clean): {e}")
    for name, m in humanized_metrics.items():
        if m and m.get("confusion_matrix"):
            try:
                _plot_confusion(m["confusion_matrix"],
                                f"{name} -- humanized",
                                args.out / "confusion_matrices" / f"humanized__{name}.png")
            except Exception as e:
                print(f"[plot] confusion matrix failed for {name} (humanized): {e}")

    # === plots: ROC panels =================================================
    if clean_scores:
        det_list = [(n, y, s) for n, (y, s) in clean_scores.items()]
        try:
            _plot_roc_panel(det_list, "ROC -- clean", args.out / "roc_curves" / "clean.png")
        except Exception as e:
            print(f"[plot] ROC clean failed: {e}")
    if humanized_scores:
        det_list = [(n, y, s) for n, (y, s) in humanized_scores.items()]
        try:
            _plot_roc_panel(det_list, "ROC -- humanized", args.out / "roc_curves" / "humanized.png")
        except Exception as e:
            print(f"[plot] ROC humanized failed: {e}")

    # === plots: score distributions ========================================
    names = sorted(set(clean_scores.keys()) | set(humanized_scores.keys()))
    for n in names:
        try:
            _plot_score_distribution(
                n,
                clean_scores.get(n),
                humanized_scores.get(n),
                args.out / "score_distributions" / f"{n}.png",
            )
        except Exception as e:
            print(f"[plot] score distribution failed for {n}: {e}")

    # === plot: per-humanizer attack success ===============================
    if humanized_metrics:
        try:
            _plot_tpr_drop_per_humanizer(
                records_humanized or [], humanized_metrics,
                args.out / "tpr_drop_per_humanizer.png",
            )
        except Exception as e:
            print(f"[plot] tpr_drop_per_humanizer failed: {e}")

    # === REPORT.md =========================================================
    cmd_str = " ".join([
        "python -m test.evaluate_arxiv",
        f"--features-clean {args.features_clean}",
        f"--features-humanized {args.features_humanized}" if args.features_humanized else "",
        f"--records-clean {args.records_clean}" if args.records_clean else "",
        f"--records-humanized {args.records_humanized}" if args.records_humanized else "",
        f"--baselines-clean {args.baselines_clean}" if args.baselines_clean else "",
        f"--baselines-humanized {args.baselines_humanized}" if args.baselines_humanized else "",
        f"--in-house-models {args.in_house_models}",
        f"--out {args.out}",
    ]).replace("  ", " ").strip()

    _report_md(
        args.out,
        clean=clean_metrics,
        humanized=humanized_metrics,
        records_clean=records_clean,
        records_humanized=records_humanized,
        available_inputs={
            "features_clean": str(args.features_clean) if args.features_clean else None,
            "features_humanized": str(args.features_humanized) if args.features_humanized else None,
            "records_clean": str(args.records_clean) if args.records_clean else None,
            "records_humanized": str(args.records_humanized) if args.records_humanized else None,
            "baselines_clean": str(args.baselines_clean) if args.baselines_clean else None,
            "baselines_humanized": str(args.baselines_humanized) if args.baselines_humanized else None,
        },
        cmd_str=cmd_str,
    )

    print(f"\nWrote report to {args.out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
