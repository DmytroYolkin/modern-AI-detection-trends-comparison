"""Emit ``models/ready_models/STEP1_RETRAIN_SUMMARY.md`` after the step-1
resplit + retrain (fusion x4, classical x7).

Reads:
  * ``models/ready_models/*.metrics.json``   -- new (post-retrain) metrics
  * ``models/ready_models_step1_pre_retrain_backup_2026-05-27/*.metrics.json``
                                            -- prior metrics for delta column
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NEW = REPO_ROOT / "models" / "ready_models"
OLD = REPO_ROOT / "models" / "ready_models_step1_pre_retrain_backup_2026-05-27"

FUSION = ["concat", "mlp", "attention", "gating"]
CLASSICAL = ["xgboost", "random_forest", "logreg", "svm", "mlp",
             "hist_gbm", "gradient_boosting"]


def _load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _val_block(metrics: dict, kind: str) -> dict:
    """Normalize fusion vs classical metric shape into one dict."""
    if metrics is None:
        return {}
    if kind == "fusion":
        # train.py writes {"metrics": {...}, "history": [...]}
        m = metrics.get("metrics", metrics)
        val = m.get("val", {})
        per = m.get("val_per_class", {})
        return {
            "val_accuracy": val.get("accuracy"),
            "val_macro_f1": val.get("macro_f1"),
            "val_human_f1": per.get("human", {}).get("f1"),
            "val_ai_f1": per.get("ai", {}).get("f1"),
            "best_epoch": m.get("best_epoch"),
            "train_seconds": m.get("train_seconds"),
        }
    else:
        # train_classical.py writes flat
        val = metrics.get("val", {})
        per = val.get("per_class", {})
        return {
            "val_accuracy": val.get("accuracy"),
            "val_macro_f1": val.get("macro_f1"),
            "val_human_f1": per.get("human", {}).get("f1"),
            "val_ai_f1": per.get("ai", {}).get("f1"),
            "best_epoch": None,
            "train_seconds": metrics.get("fit_seconds"),
        }


def _fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main() -> None:
    rows = []
    deltas = []

    for m in FUSION:
        new = _val_block(_load_metrics(NEW / f"fusion_{m}.metrics.json"), "fusion")
        old = _val_block(_load_metrics(OLD / f"fusion_{m}.metrics.json"), "fusion")
        rows.append((f"fusion_{m}", new))
        deltas.append((f"fusion_{m}", old.get("val_macro_f1"), new.get("val_macro_f1")))

    for m in CLASSICAL:
        new = _val_block(_load_metrics(NEW / f"clf_{m}.metrics.json"), "classical")
        old = _val_block(_load_metrics(OLD / f"clf_{m}.metrics.json"), "classical")
        rows.append((f"clf_{m}", new))
        deltas.append((f"clf_{m}", old.get("val_macro_f1"), new.get("val_macro_f1")))

    lines = []
    lines.append("# Step-1 Retrain Summary (2026-05-27)")
    lines.append("")
    lines.append("Author-disjoint 90/10 resplit (pooled prior train+val+test, "
                 "USE-only, post `--require-known-author --min-human-siblings 2`, "
                 "n=5547). All 11 models retrained on the new caches "
                 "(`data/features/{train,val}.npz`).")
    lines.append("")
    lines.append("Test set for these numbers is the **new val split** "
                 "(589 records, author-disjoint from train). The held-out arXiv "
                 "test set has not been built yet (separate step).")
    lines.append("")

    # ---- prior vs new val macro-F1 ---------------------------------------
    lines.append("## Prior vs new val macro-F1")
    lines.append("")
    lines.append("| model | prior val_macro_f1 | new val_macro_f1 | delta |")
    lines.append("|---|---|---|---|")
    for name, old_f1, new_f1 in deltas:
        if old_f1 is None or new_f1 is None:
            delta_str = "-"
        else:
            d = new_f1 - old_f1
            delta_str = f"{d:+.4f}"
        lines.append(f"| {name} | {_fmt(old_f1)} | {_fmt(new_f1)} | {delta_str} |")
    lines.append("")
    lines.append("Note: the prior models were validated on a *different* "
                 "(smaller, source-disjoint) val split of 667 records; the new "
                 "val split is 589 author-disjoint records drawn from the pool "
                 "of prior train+val+test. Direct numeric comparison is "
                 "indicative, not strict.")
    lines.append("")

    # ---- full per-model table --------------------------------------------
    lines.append("## Per-model val metrics (new splits)")
    lines.append("")
    lines.append("| model | val_accuracy | val_macro_f1 | val_human_f1 | "
                 "val_ai_f1 | best_epoch | train_seconds |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, m in rows:
        lines.append(
            f"| {name} | {_fmt(m.get('val_accuracy'))} | "
            f"{_fmt(m.get('val_macro_f1'))} | "
            f"{_fmt(m.get('val_human_f1'))} | "
            f"{_fmt(m.get('val_ai_f1'))} | "
            f"{_fmt(m.get('best_epoch'), nd=0)} | "
            f"{_fmt(m.get('train_seconds'), nd=2)} |"
        )
    lines.append("")

    out_path = NEW / "STEP1_RETRAIN_SUMMARY.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote: {out_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
