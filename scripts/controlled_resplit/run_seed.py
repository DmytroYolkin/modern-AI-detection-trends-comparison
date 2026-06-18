"""Stage 2 -- per-seed retrain + held-out TEST evaluation for all 14 models.

For one seed this:
  1. loads the seed's re-split cache from ``data/features_resplit/seed<k>/``;
  2. fits a FeatureNormalizer on that seed's TRAIN (so val/test see the same
     transform the model was trained under);
  3. trains all 14 models on TRAIN, selecting on VAL where the trainer does:
       * 7 classical backends  (xgboost, random_forest, logreg, svm, mlp,
                                 hist_gbm, gradient_boosting)
       * 4 neural fusion heads (concat, mlp, attention, gating)
       * 3 single-modality RFs (nela / style / trace)  -- reduced-dim inputs
     Checkpoints land in ``models/ready_models_resplit/seed<k>/``.
  4. produces probability scores on VAL and TEST, then for each model records:
       * acc@0.5, macro_f1@0.5  (argmax / 0.5 threshold on P(ai))
       * roc_auc                (on TEST, P(ai) vs label)
       * strict_fpr_threshold   (calibrated on VAL: smallest P(ai) cut whose
                                 VAL human false-positive-rate is <= 1%)
       * test_tpr_at_strict_fpr (TEST recall on AI at that VAL-calibrated cut)
       * test_fpr_at_strict_fpr (realised TEST human FPR at that cut)
       * per-record TEST y_true / p_ai / pred@0.5  (for McNemar + bootstrap)

Positive class = AI = label 1. "Strict FPR <= 1%" = at most 1% of true humans
flagged as AI.

This trains from scratch (no reuse of committed checkpoints). The committed
``data/features/`` cache, its backups, and ``models/ready_models/`` are never
touched.

Usage (from repo root):
    python -m scripts.controlled_resplit.run_seed --seed 0
    python -m scripts.controlled_resplit.run_seed --seed 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# training.paths bootstraps sys.path for the bare extractor/fusion imports.
from training import paths  # noqa: F401
from training.classical import (
    ClassicalClassifier,
    flatten_features,
    select_blocks,
)
from training.config import FUSION_METHODS, TrainConfig
from training.feature_dataset import FeatureNormalizer, FusionFeatureDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
RESPLIT_ROOT = REPO_ROOT / "data" / "features_resplit"
CKPT_ROOT = REPO_ROOT / "models" / "ready_models_resplit"
RESULTS_DIR = CKPT_ROOT / "results"

CLASSICAL_BACKENDS = ("xgboost", "random_forest", "logreg", "svm", "mlp",
                      "hist_gbm", "gradient_boosting")
SINGLE_MODALITY = ("nela", "style", "trace")  # each a random_forest on one block
STRICT_FPR = 0.01
LABEL_NAMES = ("human", "ai")


# ===========================================================================
# Metric helpers
# ===========================================================================

def _basic_metrics(y_true: np.ndarray, p_ai: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    pred = (p_ai >= 0.5).astype(int)
    out = {
        "acc": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }
    # roc_auc undefined if a split is single-class; guard it
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, p_ai))
    else:
        out["roc_auc"] = float("nan")
    return out


def _strict_fpr_threshold(y_val: np.ndarray, p_val: np.ndarray,
                          max_fpr: float = STRICT_FPR) -> float:
    """Smallest P(ai) cut whose VAL human FPR is <= max_fpr.

    Among true humans (y==0), find the threshold so that the fraction with
    P(ai) >= threshold is <= max_fpr. Lower threshold -> more AI flagged ->
    higher TPR, so we want the *lowest* threshold still satisfying the FPR cap.
    """
    human_scores = np.sort(p_val[y_val == 0])
    if human_scores.size == 0:
        return 0.5
    n = human_scores.size
    # allowed number of human false positives
    allowed = int(np.floor(max_fpr * n))
    if allowed >= n:
        return float(human_scores[0])
    # we may flag at most `allowed` humans -> threshold just above the
    # (n-allowed-1)-th highest human score. Use the (n-allowed)-th smallest
    # human score as the cut (>= it flags exactly `allowed` humans or fewer).
    idx = n - allowed - 1
    thr = float(human_scores[idx])
    # nudge above ties so we don't exceed the budget
    eps = 1e-12
    return thr + eps


def _tpr_fpr_at(y_true: np.ndarray, p_ai: np.ndarray, thr: float) -> tuple[float, float]:
    pred = (p_ai >= thr).astype(int)
    ai = y_true == 1
    hu = y_true == 0
    tpr = float(pred[ai].mean()) if ai.any() else float("nan")
    fpr = float(pred[hu].mean()) if hu.any() else float("nan")
    return tpr, fpr


# ===========================================================================
# Neural fusion training (compact reimplementation of training.train.train_one)
# ===========================================================================

def _train_fusion(method: str, train_ds, val_ds, normalizer, out_path: Path,
                  seed: int) -> dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from training.model import FusionClassifier
    from training.train import run_epoch

    config = TrainConfig(fusion_method=method, seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = paths.resolve_device(config.device)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    dims = train_ds.dims
    model = FusionClassifier(
        dims["nela"], dims["style"], dims["trace"],
        fusion_method=config.fusion_method,
        hidden_dim=config.hidden_dim,
        head_hidden_dim=config.head_hidden_dim,
        dropout=config.dropout,
        num_classes=config.num_classes,
        modality_dropout_nela=config.modality_dropout_nela,
        modality_dropout_style=config.modality_dropout_style,
        modality_dropout_trace=config.modality_dropout_trace,
    ).to(device)

    counts = train_ds.class_counts()
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (config.num_classes * max(counts.get(c, 0), 1))
         for c in range(config.num_classes)],
        dtype=torch.float32, device=device,
    ) if config.class_weighting else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=config.lr_patience)

    best_f1, best_state, epochs_since = -1.0, None, 0
    for epoch in range(1, config.epochs + 1):
        run_epoch(model, train_loader, criterion, device, optimizer)
        va, _, _ = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va["macro_f1"])
        if va["macro_f1"] > best_f1:
            best_f1 = va["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since = 0
        else:
            epochs_since += 1
        if epochs_since >= config.early_stopping_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.save(out_path, normalizer=normalizer,
               metrics={"val_macro_f1": best_f1, "fusion_method": method},
               train_config=config.to_dict(), label_names=LABEL_NAMES)
    return {"model": model, "device": device}


def _fusion_proba(model, ds, device) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        nela = torch.from_numpy(np.asarray(ds.nela, np.float32)).to(device)
        style = torch.from_numpy(np.asarray(ds.style, np.float32)).to(device)
        trace = torch.from_numpy(np.asarray(ds.trace, np.float32)).to(device)
        probs = model.predict_proba(nela, style, trace).cpu().numpy()
    return probs[:, 1]  # P(ai)


# ===========================================================================
# One model -> evaluated record
# ===========================================================================

def _evaluate(name: str, kind: str, y_val, p_val, y_test, p_test) -> dict:
    base = _basic_metrics(y_test, p_test)
    thr = _strict_fpr_threshold(y_val, p_val, STRICT_FPR)
    tpr, fpr = _tpr_fpr_at(y_test, p_test, thr)
    val_tpr, val_fpr = _tpr_fpr_at(y_val, p_val, thr)
    return {
        "model": name,
        "kind": kind,
        "test_acc": base["acc"],
        "test_macro_f1": base["macro_f1"],
        "test_roc_auc": base["roc_auc"],
        "strict_fpr_threshold": thr,
        "val_tpr_at_strict_fpr": val_tpr,
        "val_fpr_at_strict_fpr": val_fpr,
        "test_tpr_at_strict_fpr": tpr,
        "test_fpr_at_strict_fpr": fpr,
        "test_y_true": y_test.astype(int).tolist(),
        "test_p_ai": [float(x) for x in p_test],
        "test_pred_05": (p_test >= 0.5).astype(int).tolist(),
    }


# ===========================================================================
# Per-seed driver
# ===========================================================================

def run_one_seed(seed: int) -> dict:
    feat_dir = RESPLIT_ROOT / f"seed{seed}"
    for sp in ("train", "val", "test"):
        if not (feat_dir / f"{sp}.npz").exists():
            raise SystemExit(f"missing {feat_dir / (sp + '.npz')}; run resplit_seed --seed {seed} first")

    ckpt_dir = CKPT_ROOT / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # raw datasets (for fitting the normalizer + classical block slicing)
    train_ds = FusionFeatureDataset(feat_dir / "train.npz")
    val_ds = FusionFeatureDataset(feat_dir / "val.npz")
    test_ds = FusionFeatureDataset(feat_dir / "test.npz")
    normalizer = FeatureNormalizer.fit(train_ds.raw_arrays())
    for ds in (train_ds, val_ds, test_ds):
        ds.apply_normalizer(normalizer)

    y_val = np.asarray(val_ds.labels)
    y_test = np.asarray(test_ds.labels)

    results = []
    t0 = time.time()

    # ---- 7 classical backends (full 225-d) -------------------------------
    Xtr = flatten_features(train_ds.nela, train_ds.style, train_ds.trace)
    Xva = flatten_features(val_ds.nela, val_ds.style, val_ds.trace)
    Xte = flatten_features(test_ds.nela, test_ds.style, test_ds.trace)
    ytr = np.asarray(train_ds.labels)
    dims = train_ds.dims
    for backend in CLASSICAL_BACKENDS:
        try:
            clf = ClassicalClassifier(backend, seed=seed, class_weighting=True)
            clf.fit(Xtr, ytr)
            clf.save(ckpt_dir / f"clf_{backend}.joblib", dims=dims,
                     normalizer=normalizer, metrics={}, label_names=LABEL_NAMES)
            p_val = clf.predict_proba(Xva)[:, 1]
            p_test = clf.predict_proba(Xte)[:, 1]
            results.append(_evaluate(f"clf_{backend}", "classical",
                                     y_val, p_val, y_test, p_test))
            print(f"  [seed {seed}] clf_{backend:<18} "
                  f"f1={results[-1]['test_macro_f1']:.4f} auc={results[-1]['test_roc_auc']:.4f} "
                  f"tpr@fpr1={results[-1]['test_tpr_at_strict_fpr']:.4f}")
        except Exception as exc:
            print(f"  [seed {seed}] clf_{backend} FAILED: {exc}")

    # ---- 3 single-modality random forests --------------------------------
    block_arrays = {
        "train": {"nela": train_ds.nela, "style": train_ds.style, "trace": train_ds.trace},
        "val": {"nela": val_ds.nela, "style": val_ds.style, "trace": val_ds.trace},
        "test": {"nela": test_ds.nela, "style": test_ds.style, "trace": test_ds.trace},
    }
    for block in SINGLE_MODALITY:
        try:
            Xtr_b = select_blocks(train_ds.nela, train_ds.style, train_ds.trace, (block,))
            Xva_b = select_blocks(val_ds.nela, val_ds.style, val_ds.trace, (block,))
            Xte_b = select_blocks(test_ds.nela, test_ds.style, test_ds.trace, (block,))
            clf = ClassicalClassifier("random_forest", seed=seed, class_weighting=True)
            clf.fit(Xtr_b, ytr)
            clf.save(ckpt_dir / f"clf_random_forest_{block}.joblib",
                     dims={block: dims[block]}, normalizer=normalizer,
                     metrics={}, label_names=LABEL_NAMES)
            p_val = clf.predict_proba(Xva_b)[:, 1]
            p_test = clf.predict_proba(Xte_b)[:, 1]
            results.append(_evaluate(f"rf_{block}", "single_modality",
                                     y_val, p_val, y_test, p_test))
            print(f"  [seed {seed}] rf_{block:<20} "
                  f"f1={results[-1]['test_macro_f1']:.4f} auc={results[-1]['test_roc_auc']:.4f} "
                  f"tpr@fpr1={results[-1]['test_tpr_at_strict_fpr']:.4f}")
        except Exception as exc:
            print(f"  [seed {seed}] rf_{block} FAILED: {exc}")

    # ---- 4 neural fusion heads -------------------------------------------
    for method in FUSION_METHODS:
        try:
            bundle = _train_fusion(method, train_ds, val_ds, normalizer,
                                   ckpt_dir / f"fusion_{method}.pt", seed)
            p_val = _fusion_proba(bundle["model"], val_ds, bundle["device"])
            p_test = _fusion_proba(bundle["model"], test_ds, bundle["device"])
            results.append(_evaluate(f"fusion_{method}", "neural",
                                     y_val, p_val, y_test, p_test))
            print(f"  [seed {seed}] fusion_{method:<16} "
                  f"f1={results[-1]['test_macro_f1']:.4f} auc={results[-1]['test_roc_auc']:.4f} "
                  f"tpr@fpr1={results[-1]['test_tpr_at_strict_fpr']:.4f}")
        except Exception as exc:
            print(f"  [seed {seed}] fusion_{method} FAILED: {exc}")

    elapsed = round(time.time() - t0, 1)
    out = {
        "seed": seed,
        "feature_dir": str(feat_dir),
        "strict_fpr": STRICT_FPR,
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "elapsed_seconds": elapsed,
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"seed{seed}_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[seed {seed}] {len(results)} models in {elapsed}s -> {out_path}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    for s in args.seed:
        run_one_seed(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
