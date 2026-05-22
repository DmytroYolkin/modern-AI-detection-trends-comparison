"""Evaluate a trained checkpoint on the test split.

Handles both model families produced by the training framework:

  * neural fusion model      -- ``.pt``     (training/train.py)
  * classical classifier     -- ``.joblib`` (training/train_classical.py)

Loads the checkpoint (with its embedded feature normalizer), runs it over the
cached test features, and reports accuracy, per-class precision/recall/F1, the
confusion matrix, and a per-source-corpus accuracy breakdown.

Usage
-----
    python -m test.evaluate --model models/test_models/fusion_gating.pt
    python -m test.evaluate --model models/test_models/clf_xgboost.joblib
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Bootstrap so `python test/evaluate.py` works as well as `-m test.evaluate`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from training import paths
from training.classical import ClassicalClassifier, block_importances, flatten_features
from training.feature_dataset import FeatureNormalizer, FusionFeatureDataset
from training.model import FusionClassifier


def _predict_neural(model, ds, device, batch_size=256):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True,
                        help="path to a .pt (neural) or .joblib (classical) checkpoint")
    parser.add_argument("--features", type=Path, default=paths.FEATURE_DIR / "test.npz",
                        help="cached feature .npz to evaluate on (default: test split)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Checkpoint not found: {args.model}")
    if not args.features.exists():
        raise SystemExit(
            f"Feature cache not found: {args.features}\n"
            f"Run:  python -m training.build_dataset --splits test")

    is_neural = args.model.suffix == ".pt"
    device = paths.resolve_device(args.device)

    if is_neural:
        model, payload = FusionClassifier.load(args.model, map_location=device)
        descr = f"neural fusion ({payload['hparams']['fusion_method']})"
    else:
        model, payload = ClassicalClassifier.load(args.model)
        descr = f"classical ({payload['backend']})"

    label_names = payload.get("label_names", ["human", "ai"])
    normalizer = FeatureNormalizer.from_state_dict(payload.get("normalizer"))

    ds = FusionFeatureDataset(args.features)
    if normalizer is not None:
        ds.apply_normalizer(normalizer)

    y_true = ds.labels
    if is_neural:
        y_pred, _ = _predict_neural(model, ds, device)
    else:
        y_pred, _ = _predict_classical(model, ds)

    print(f"Model        : {args.model}")
    print(f"Type         : {descr}")
    print(f"Features     : {args.features}  ({len(ds)} records)")
    print(f"StyleDecipher coverage: {ds.style_coverage():.1%}\n")

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}\n")

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=label_names,
                                digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("Confusion matrix (rows=true, cols=pred):")
    print("            " + "".join(f"{n:>10}" for n in label_names))
    for i, name in enumerate(label_names):
        print(f"  {name:<10}" + "".join(f"{int(v):>10}" for v in cm[i]))

    # per-extractor importance (classical tree/linear models only)
    if not is_neural:
        importances = model.feature_importances()
        if importances is not None:
            blocks = block_importances(importances, payload["dims"])
            print("\nExtractor contribution (feature importance):")
            for name, frac in blocks.items():
                print(f"  {name:<8} {frac:.1%}")

    # per-source-corpus accuracy
    by_source: dict[str, list[int]] = defaultdict(list)
    for src, t, p in zip(ds.sources, y_true, y_pred):
        by_source[str(src)].append(int(t == p))
    print("\nPer-source accuracy:")
    for src in sorted(by_source):
        hits = by_source[src]
        print(f"  {src:<14} {np.mean(hits):.4f}  (n={len(hits)})")


if __name__ == "__main__":
    main()
