"""Train the fusion-based human-vs-AI text classifier.

Consumes the cached feature matrices written by ``build_dataset.py`` and trains
a `FusionClassifier`. Checkpoints land in ``models/test_models/`` by default
(experimental, gitignored); copy a validated one into ``models/ready_models/``.

Usage
-----
    # train once with the default (gating) fusion strategy
    python -m training.train

    # sweep every fusion strategy and compare
    python -m training.train --fusion-method all --epochs 60

    # custom run
    python -m training.train --fusion-method attention --lr 5e-4 --name attn_v2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Bootstrap so `python training/train.py` works as well as `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from training import paths
from training.config import FUSION_METHODS, TrainConfig
from training.feature_dataset import FeatureNormalizer, FusionFeatureDataset
from training.model import FusionClassifier

LABEL_NAMES = ("human", "ai")


# ===========================================================================
# Epoch runner
# ===========================================================================

def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[dict, np.ndarray, np.ndarray]:
    """Run one pass over `loader`. Trains when `optimizer` is given, else evals."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, seen = 0.0, 0
    preds, trues = [], []
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for nela, style, trace, label in loader:
            nela = nela.to(device)
            style = style.to(device)
            trace = trace.to(device)
            label = label.to(device)

            logits = model(nela, style, trace)
            loss = criterion(logits, label)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            total_loss += loss.item() * label.size(0)
            seen += label.size(0)
            preds.append(logits.argmax(dim=-1).detach().cpu())
            trues.append(label.detach().cpu())

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(trues).numpy()
    metrics = {
        "loss": total_loss / max(seen, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return metrics, y_true, y_pred


# ===========================================================================
# Single training run
# ===========================================================================

def train_one(config: TrainConfig, train_ds: FusionFeatureDataset,
               val_ds: FusionFeatureDataset, normalizer: FeatureNormalizer,
               out_path: Path) -> dict:
    """Train one model end to end; save the best checkpoint; return metrics."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
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

    # class-weighted loss for the imbalanced human/ai split
    counts = train_ds.class_counts()
    total = sum(counts.values())
    if config.class_weighting:
        weights = torch.tensor(
            [total / (config.num_classes * max(counts.get(c, 0), 1))
             for c in range(config.num_classes)],
            dtype=torch.float32, device=device,
        )
    else:
        weights = None
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=config.lr_patience)

    print(f"\n[{config.fusion_method}] device={device}  "
          f"train={len(train_ds)} val={len(val_ds)}  class_counts={counts}")

    best_f1 = -1.0
    best_epoch = -1
    best_state = None
    best_val_metrics: dict = {}
    epochs_since_best = 0
    history = []
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        tr, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        va, _, _ = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va["macro_f1"])
        history.append({"epoch": epoch, "train": tr, "val": va})

        marker = ""
        if va["macro_f1"] > best_f1:
            best_f1 = va["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = va
            epochs_since_best = 0
            marker = "  <- best"
        else:
            epochs_since_best += 1

        print(f"  epoch {epoch:>3}/{config.epochs}  "
              f"train_loss={tr['loss']:.4f} acc={tr['accuracy']:.3f}  |  "
              f"val_loss={va['loss']:.4f} acc={va['accuracy']:.3f} "
              f"macroF1={va['macro_f1']:.4f}{marker}")

        if epochs_since_best >= config.early_stopping_patience:
            print(f"  early stop: no val-F1 gain for {config.early_stopping_patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # detailed best-epoch validation report
    _, y_true, y_pred = run_epoch(model, val_loader, criterion, device)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    per_class = {
        LABEL_NAMES[c]: {"precision": float(prec[c]), "recall": float(rec[c]), "f1": float(f1[c])}
        for c in (0, 1)
    }
    elapsed = time.time() - start
    metrics = {
        "fusion_method": config.fusion_method,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_seconds": round(elapsed, 1),
        "val": best_val_metrics,
        "val_per_class": per_class,
    }

    model.save(out_path, normalizer=normalizer, metrics=metrics,
               train_config=config.to_dict(), label_names=LABEL_NAMES)
    out_path.with_suffix(".metrics.json").write_text(
        json.dumps({"metrics": metrics, "history": history}, indent=2), encoding="utf-8")
    print(f"  best epoch {best_epoch}  val macroF1={best_f1:.4f}  -> saved {out_path}")
    return metrics


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fusion-method", default="gating",
                        choices=list(FUSION_METHODS) + ["all"],
                        help='fusion strategy, or "all" to sweep every strategy')
    parser.add_argument("--feature-dir", type=Path, default=paths.FEATURE_DIR,
                        help="directory holding train/val/test .npz caches")
    parser.add_argument("--out-dir", type=Path, default=paths.TEST_MODELS_DIR,
                        help="where to save checkpoints (default: models/test_models)")
    parser.add_argument("--name", default=None,
                        help="checkpoint base name (default: fusion_<method>)")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--hidden-dim", type=int, default=TrainConfig.hidden_dim)
    parser.add_argument("--head-hidden-dim", type=int, default=TrainConfig.head_hidden_dim)
    parser.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    parser.add_argument("--no-class-weighting", action="store_true",
                        help="disable inverse-frequency class weighting")
    parser.add_argument("--no-normalize", action="store_true",
                        help="disable feature standardisation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--modality-dropout-nela",  type=float,
                        default=TrainConfig.modality_dropout_nela,
                        help="per-sample whole-block dropout probability for NELA "
                             "(training-time only; default 0.5)")
    parser.add_argument("--modality-dropout-style", type=float,
                        default=TrainConfig.modality_dropout_style,
                        help="per-sample whole-block dropout probability for StyleDecipher "
                             "(training-time only; default 0.2)")
    parser.add_argument("--modality-dropout-trace", type=float,
                        default=TrainConfig.modality_dropout_trace,
                        help="per-sample whole-block dropout probability for TRACE "
                             "(training-time only; default 0.2)")
    args = parser.parse_args()

    train_npz = args.feature_dir / "train.npz"
    val_npz = args.feature_dir / "val.npz"
    for p in (train_npz, val_npz):
        if not p.exists():
            raise SystemExit(
                f"Missing feature cache: {p}\n"
                f"Run:  python -m training.build_dataset --splits all")

    # load caches; fit the normalizer on the (un-normalised) train split
    train_ds = FusionFeatureDataset(train_npz)
    val_ds = FusionFeatureDataset(val_npz)
    normalizer = None
    if not args.no_normalize:
        normalizer = FeatureNormalizer.fit(train_ds.raw_arrays())
        train_ds.apply_normalizer(normalizer)
        val_ds.apply_normalizer(normalizer)

    print(f"Feature dims: {train_ds.dims}")
    print(f"StyleDecipher coverage -- train {train_ds.style_coverage():.1%}, "
          f"val {val_ds.style_coverage():.1%}")

    methods = list(FUSION_METHODS) if args.fusion_method == "all" else [args.fusion_method]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for method in methods:
        config = TrainConfig(
            fusion_method=method,
            hidden_dim=args.hidden_dim,
            head_hidden_dim=args.head_hidden_dim,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            class_weighting=not args.no_class_weighting,
            normalize_features=not args.no_normalize,
            seed=args.seed,
            device=args.device,
            modality_dropout_nela=args.modality_dropout_nela,
            modality_dropout_style=args.modality_dropout_style,
            modality_dropout_trace=args.modality_dropout_trace,
        )
        base_name = args.name or f"fusion_{method}"
        # when sweeping, keep each method's checkpoint distinct
        if args.fusion_method == "all" and args.name:
            base_name = f"{args.name}_{method}"
        out_path = args.out_dir / f"{base_name}.pt"
        metrics = train_one(config, train_ds, val_ds, normalizer, out_path)
        summary.append((method, metrics, out_path))

    print("\n=== summary ===")
    for method, metrics, out_path in sorted(summary, key=lambda x: -x[1]["val"]["macro_f1"]):
        v = metrics["val"]
        print(f"  {method:<10} val_acc={v['accuracy']:.4f}  "
              f"val_macroF1={v['macro_f1']:.4f}  ({out_path.name})")
    if len(summary) > 1:
        best = max(summary, key=lambda x: x[1]["val"]["macro_f1"])
        print(f"\nBest fusion strategy: {best[0]}  ->  {best[2]}")


if __name__ == "__main__":
    main()
