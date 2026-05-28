"""Train a classical classifier on the cached extractor features.

The non-neural counterpart of `train.py`: same feature cache, same evaluation,
but the model is XGBoost / random forest / logistic regression / SVM / a
gradient-boosting variant instead of the neural fusion network.

Checkpoints (`.joblib`) land in `models/test_models/` by default.

Usage
-----
    python -m training.train_classical --classifier xgboost
    python -m training.train_classical --classifier all          # sweep every backend
    python -m training.train_classical --classifier random_forest --n-estimators 800
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Bootstrap so `python training/train_classical.py` works as well as `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from training import paths
from training.classical import (
    BACKENDS,
    ClassicalClassifier,
    block_importances,
    flatten_features,
    select_blocks,
)
from training.feature_dataset import FeatureNormalizer, FusionFeatureDataset

VALID_BLOCKS = ("nela", "style", "trace")

LABEL_NAMES = ("human", "ai")


def _metrics(y_true, y_pred) -> dict:
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {
            LABEL_NAMES[c]: {"precision": float(prec[c]), "recall": float(rec[c]),
                             "f1": float(f1[c])}
            for c in (0, 1)
        },
    }


def train_one(backend: str, X_train, y_train, X_val, y_val, dims: dict,
              normalizer, out_path: Path, *, seed: int, class_weighting: bool,
              overrides: dict) -> dict:
    """Fit one classical backend, evaluate on val, save the checkpoint."""
    print(f"\n[{backend}] fitting on {len(X_train)} records ...")
    clf = ClassicalClassifier(backend, seed=seed, class_weighting=class_weighting,
                              **overrides)
    start = time.time()
    clf.fit(X_train, y_train)
    fit_seconds = time.time() - start

    val = _metrics(y_val, clf.predict(X_val))
    metrics = {"backend": backend, "fit_seconds": round(fit_seconds, 2), "val": val}

    # per-extractor contribution -- the comparison this repo is about
    importances = clf.feature_importances()
    if importances is not None:
        blocks = block_importances(importances, dims)
        metrics["val"]["extractor_importance"] = blocks
        block_str = "  ".join(f"{k}={v:.1%}" for k, v in blocks.items())
        print(f"  extractor importance:  {block_str}")

    clf.save(out_path, dims=dims, normalizer=normalizer, metrics=metrics,
             label_names=LABEL_NAMES)
    out_path.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  val acc={val['accuracy']:.4f}  macroF1={val['macro_f1']:.4f}  "
          f"({fit_seconds:.1f}s)  -> saved {out_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", default="xgboost",
                        choices=list(BACKENDS) + ["all"],
                        help='backend, or "all" to sweep every available backend')
    parser.add_argument("--feature-dir", type=Path, default=paths.FEATURE_DIR,
                        help="directory holding train/val/test .npz caches")
    parser.add_argument("--out-dir", type=Path, default=paths.TEST_MODELS_DIR,
                        help="where to save checkpoints (default: models/test_models)")
    parser.add_argument("--name", default=None,
                        help="checkpoint base name (default: clf_<backend>)")
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--no-class-weighting", action="store_true",
                        help="disable balanced sample weighting")
    parser.add_argument("--no-normalize", action="store_true",
                        help="disable feature standardisation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-blocks", default=None,
        help='Comma-separated subset of {nela,style,trace}. When set, the '
             'classifier is trained only on those feature blocks and saved '
             'with reduced "dims" -- e.g. --feature-blocks nela trains a '
             'NELA-only model (87-dim input) for per-modality ablation.',
    )
    args = parser.parse_args()

    train_npz = args.feature_dir / "train.npz"
    val_npz = args.feature_dir / "val.npz"
    for p in (train_npz, val_npz):
        if not p.exists():
            raise SystemExit(
                f"Missing feature cache: {p}\n"
                f"Run:  python -m training.build_dataset --splits all")

    train_ds = FusionFeatureDataset(train_npz)
    val_ds = FusionFeatureDataset(val_npz)
    dims = train_ds.dims

    normalizer = None
    if not args.no_normalize:
        normalizer = FeatureNormalizer.fit(train_ds.raw_arrays())
        train_ds.apply_normalizer(normalizer)
        val_ds.apply_normalizer(normalizer)

    if args.feature_blocks:
        requested = tuple(b.strip() for b in args.feature_blocks.split(",") if b.strip())
        bad = [b for b in requested if b not in VALID_BLOCKS]
        if bad:
            raise SystemExit(f"unknown feature block(s): {bad}; pick from {VALID_BLOCKS}")
        X_train = select_blocks(train_ds.nela, train_ds.style, train_ds.trace, requested)
        X_val = select_blocks(val_ds.nela, val_ds.style, val_ds.trace, requested)
        dims = {b: dims[b] for b in VALID_BLOCKS if b in requested}
        print(f"Restricted to feature blocks: {requested}")
    else:
        X_train = flatten_features(train_ds.nela, train_ds.style, train_ds.trace)
        X_val = flatten_features(val_ds.nela, val_ds.style, val_ds.trace)
    y_train = train_ds.labels
    y_val = val_ds.labels

    print(f"Feature dims: {dims}  ->  flattened input width = {X_train.shape[1]}")
    print(f"StyleDecipher coverage -- train {train_ds.style_coverage():.1%}, "
          f"val {val_ds.style_coverage():.1%}")

    if args.classifier == "all":
        backends = ClassicalClassifier.available_backends()
        unavailable = [b for b in BACKENDS if b not in backends]
        if unavailable:
            print(f"Skipping unavailable backends: {unavailable}")
    else:
        backends = [args.classifier]

    overrides = {"n_estimators": args.n_estimators, "max_depth": args.max_depth,
                 "learning_rate": args.learning_rate}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for backend in backends:
        base = args.name or f"clf_{backend}"
        if args.classifier == "all" and args.name:
            base = f"{args.name}_{backend}"
        if args.feature_blocks and args.name is None:
            base = f"{base}_{'_'.join(b.strip() for b in args.feature_blocks.split(','))}"
        out_path = args.out_dir / f"{base}.joblib"
        try:
            metrics = train_one(
                backend, X_train, y_train, X_val, y_val, dims, normalizer, out_path,
                seed=args.seed, class_weighting=not args.no_class_weighting,
                overrides=overrides,
            )
            summary.append((backend, metrics, out_path))
        except Exception as exc:
            print(f"[{backend}] FAILED: {exc}")

    print("\n=== summary ===")
    for backend, metrics, out_path in sorted(summary, key=lambda x: -x[1]["val"]["macro_f1"]):
        v = metrics["val"]
        print(f"  {backend:<18} val_acc={v['accuracy']:.4f}  "
              f"val_macroF1={v['macro_f1']:.4f}  ({out_path.name})")
    if len(summary) > 1:
        best = max(summary, key=lambda x: x[1]["val"]["macro_f1"])
        print(f"\nBest classifier: {best[0]}  ->  {best[2]}")


if __name__ == "__main__":
    main()
