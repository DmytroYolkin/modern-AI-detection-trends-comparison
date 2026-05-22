# Training framework

End-to-end pipeline for training the multi-extractor human-vs-AI text detector.

## Pipeline

```
data/dataset_ready_final/*.jsonl
        │
        │  build_dataset.py   (NELA + StyleDecipher + TRACE extractors)
        ▼
data/features/{train,val,test}.npz        ← cached feature matrices
        │
        ├─ train.py            → neural    : FusionClassifier (fusion + head)   .pt
        └─ train_classical.py  → classical : XGBoost / RF / LogReg / SVM / GBM  .joblib
        │
        ▼
models/test_models/<checkpoint>   (+ .metrics.json)
        │
        │  test/evaluate.py    (handles both .pt and .joblib)
        ▼
test-split accuracy / F1 / confusion matrix
```

Two interchangeable classifier tracks consume the *same* feature cache:

- **neural** — `MultiFeatureFusion` projects each stream to a hidden space and
  fuses them, then a head predicts the class.
- **classical** — the 3 streams are concatenated into one 225-dim vector
  (87 NELA + 10 StyleDecipher + 128 TRACE) and fed straight to a classical
  classifier, which does its own feature combination.

Extractors are slow, so `build_dataset.py` runs them **once** and caches the
`(nela=87, style=10, trace=128)` feature triple per record. `train.py` then
trains purely on the cache — fast and repeatable.

## Files

| File                   | Role |
|------------------------|------|
| `build_dataset.py`     | Run the 3 extractors over each split → `data/features/*.npz` |
| `extractor_pipeline.py`| `FeaturePipeline` — unified NELA / StyleDecipher / TRACE wrapper |
| `feature_dataset.py`   | `FusionFeatureDataset` (torch) + `FeatureNormalizer` |
| `model.py`             | `FusionClassifier` — neural fusion backbone + classification head |
| `classical.py`         | `ClassicalClassifier` — XGBoost / RF / LogReg / SVM / GBM wrapper |
| `config.py`            | `TrainConfig` hyper-parameters (neural) |
| `train.py`             | Training loop / CLI — neural model |
| `train_classical.py`   | Training CLI — classical classifiers |

`fusion/combination_all.py::MultiFeatureFusion` is reused unchanged as the
neural feature backbone; `model.py` only adds the classifier head it lacked.

## Usage

```bash
# 1. cache features for all splits (offline StyleDecipher from dataset rewrites)
python -m training.build_dataset --splits all --styledecipher cached

#    quick smoke run first if you like:
python -m training.build_dataset --splits test --limit 50

# 2a. train the NEURAL model — one fusion strategy, or sweep all four
python -m training.train --fusion-method gating
python -m training.train --fusion-method all --epochs 60

# 2b. or train a CLASSICAL classifier — one backend, or sweep all of them
python -m training.train_classical --classifier xgboost
python -m training.train_classical --classifier all

# 3. evaluate any checkpoint on the test split (.pt or .joblib — auto-detected)
python -m test.evaluate --model models/test_models/fusion_gating.pt
python -m test.evaluate --model models/test_models/clf_xgboost.joblib
```

## Classifier choices

Neural fusion (`train.py --fusion-method`): `concat`, `mlp`, `attention`,
`gating`.

Classical (`train_classical.py --classifier`): `xgboost`, `random_forest`,
`logreg`, `svm`, `hist_gbm`, `gradient_boosting`. Tree/linear backends also
report **per-extractor feature importance** (how much NELA vs StyleDecipher vs
TRACE drives the prediction). `xgboost` needs the `xgboost` package; missing
backends are skipped automatically under `--classifier all`.

## StyleDecipher modes (`build_dataset.py --styledecipher`)

- `cached` *(default)* — compare each text against the LLM rewrites already in
  `rewritten_texts.jsonl`. Fully offline. Coverage is limited to USE essays and
  their rewrites — see `style_coverage` in `data/features/meta.json`.
- `ollama` — generate fresh rewrites per text via Ollama. Full coverage, slow,
  needs a running Ollama server.
- `off` — skip StyleDecipher (zero vector); trains effectively on NELA + TRACE.

## Resumable builds

`build_dataset.py` checkpoints atomically every `--checkpoint-every` records
(default 25). If a build is interrupted — Ctrl+C, `kill`, crash, power loss —
just **rerun the exact same command**: cached records are skipped and
extraction continues from where it stopped. This matters most for
`--styledecipher ollama`, which can run for days. Use `--restart` to discard
existing caches and rebuild from scratch.

## Models folder

- `models/ready_models/` — validated, **committed** checkpoints.
- `models/test_models/`  — experimental checkpoints, **gitignored** (default
  output of `train.py`). Promote a good run by copying it into `ready_models/`.

## Tests

```bash
python -m pytest test/                       # fast unit tests
RUN_EXTRACTOR_TESTS=1 python -m pytest test/ # also exercise the real extractors
```
