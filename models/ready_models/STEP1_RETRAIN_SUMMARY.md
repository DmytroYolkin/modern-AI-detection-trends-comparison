# Step-1 Retrain Summary (2026-05-27)

Author-disjoint 90/10 resplit (pooled prior train+val+test, USE-only, post `--require-known-author --min-human-siblings 2`, n=5547). All 11 models retrained on the new caches (`data/features/{train,val}.npz`).

Test set for these numbers is the **new val split** (589 records, author-disjoint from train). The held-out arXiv test set has not been built yet (separate step).

## Prior vs new val macro-F1

| model | prior val_macro_f1 | new val_macro_f1 | delta |
|---|---|---|---|
| fusion_concat | 1.0000 | 0.9970 | -0.0030 |
| fusion_mlp | 1.0000 | 0.9970 | -0.0030 |
| fusion_attention | 1.0000 | 1.0000 | +0.0000 |
| fusion_gating | 0.9974 | 0.9970 | -0.0004 |
| clf_xgboost | 0.9948 | 0.9940 | -0.0008 |
| clf_random_forest | 0.9895 | 0.9940 | +0.0045 |
| clf_logreg | 0.9974 | 0.9941 | -0.0033 |
| clf_svm | 1.0000 | 0.9970 | -0.0030 |
| clf_mlp | 0.9921 | 0.9970 | +0.0049 |
| clf_hist_gbm | 0.9948 | 0.9911 | -0.0037 |
| clf_gradient_boosting | 0.9921 | 0.9911 | -0.0011 |

Note: the prior models were validated on a *different* (smaller, source-disjoint) val split of 667 records; the new val split is 589 author-disjoint records drawn from the pool of prior train+val+test. Direct numeric comparison is indicative, not strict.

## Per-model val metrics (new splits)

| model | val_accuracy | val_macro_f1 | val_human_f1 | val_ai_f1 | best_epoch | train_seconds |
|---|---|---|---|---|---|---|
| fusion_concat | 0.9983 | 0.9970 | 0.9951 | 0.9990 | 16 | 9.60 |
| fusion_mlp | 0.9983 | 0.9970 | 0.9951 | 0.9990 | 16 | 9.50 |
| fusion_attention | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 11 | 9.30 |
| fusion_gating | 0.9983 | 0.9970 | 0.9951 | 0.9990 | 4 | 4.80 |
| clf_xgboost | 0.9966 | 0.9940 | 0.9901 | 0.9980 | - | 3.52 |
| clf_random_forest | 0.9966 | 0.9940 | 0.9901 | 0.9980 | - | 1.50 |
| clf_logreg | 0.9966 | 0.9941 | 0.9902 | 0.9979 | - | 0.06 |
| clf_svm | 0.9983 | 0.9970 | 0.9951 | 0.9990 | - | 2.55 |
| clf_mlp | 0.9983 | 0.9970 | 0.9951 | 0.9990 | - | 2.40 |
| clf_hist_gbm | 0.9949 | 0.9911 | 0.9852 | 0.9969 | - | 4.65 |
| clf_gradient_boosting | 0.9949 | 0.9911 | 0.9852 | 0.9969 | - | 125.97 |
