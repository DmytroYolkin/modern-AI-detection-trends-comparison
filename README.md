# Modern AI Detection Trends Comparison

This repository compares modern AI-driven approaches for detecting human-vs-AI
text. It combines three feature extractors — **NELA** (linguistic/credibility
features), **StyleDecipher** (style-shift vs. LLM rewrites), and **TRACE**
(contrastive author fingerprint) — and trains both **neural fusion** and
**classical** classifiers on top of them, so the contribution of each extractor
can be measured side by side.

## Installation

Install the required Python packages:

```bash
pip install pandas numpy nltk nela_features
pip install sentence-transformers scikit-learn python-Levenshtein
pip install torch transformers
pip install xgboost joblib pytest tqdm
pip install ollama
```

`xgboost` is optional — without it the `xgboost` classifier backend is skipped
automatically. `ollama` is only needed for the optional StyleDecipher `ollama`
mode below.

The first run downloads NLTK data and ~400 MB of transformer weights
(sentence-transformer / TRACE models). For the optional StyleDecipher `ollama`
mode, also pull the rewrite models:

```bash
ollama pull qwen2
ollama pull llama3
ollama pull mistral
ollama pull gemma
ollama pull phi3
```

## Project Structure

- `data/` — the finalized human-vs-AI dataset (`dataset_ready_final/`), the
  `preprocessing/` loader package, and the scripts that built the dataset.
  See [data/README.md](data/README.md).
- `extractors/` — the three feature extractors: `NELA`, `StyleDecipher`,
  `TRACE`.
- `fusion/` — neural fusion strategies (`MultiFeatureFusion`) that combine the
  extractor outputs.
- `training/` — end-to-end training pipeline: feature caching, neural training,
  classical training. See [training/README.md](training/README.md).
- `test/` — the evaluation CLI (`evaluate.py`) and the unit/integration tests.
- `models/` — `ready_models/` (committed checkpoints) and `test_models/`
  (gitignored experimental checkpoints).

## Pipeline at a Glance

```
data/dataset_ready_final/*.jsonl
        │  build_dataset.py   (NELA + StyleDecipher + TRACE extractors)
        ▼
data/features/{train,val,test}.npz        ← cached feature matrices
        │
        ├─ train.py            → neural    : FusionClassifier        .pt
        └─ train_classical.py  → classical : XGBoost / RF / LogReg / …  .joblib
        │
        ▼
models/test_models/<checkpoint>
        │  test/evaluate.py
        ▼
test-split accuracy / F1 / confusion matrix
```

Extractors are slow, so `build_dataset.py` runs them **once** and caches the
`(nela=87, style=10, trace=128)` feature triple per record. Training then runs
purely on the cache — fast and repeatable.

## Usage

All commands are run from the repository root. Each step uses Python's `-m`
module syntax.

### Step 1 — Build the feature cache

Run the three extractors over the train/val/test splits and cache the result to
`data/features/*.npz`:

```bash
# full build (offline StyleDecipher from the dataset's LLM rewrites)
python -m training.build_dataset --splits all --styledecipher cached

# quick smoke run first, if you like — 50 records of the test split
python -m training.build_dataset --splits test --limit 50
```

The build is **resumable** — if it is interrupted (Ctrl+C, kill, crash), just
rerun the exact same command and it continues from the last checkpoint. Use
`--restart` to discard caches and rebuild from scratch.

StyleDecipher modes (`--styledecipher`):

- `cached` *(default)* — compare each text against the LLM rewrites already in
  the dataset. Fully offline; coverage limited to USE essays and their rewrites.
- `ollama` *(use this one)* — generate fresh rewrites per text via a running Ollama server. Full
  coverage, but slow.
- `off` — skip StyleDecipher (zero vector); trains on NELA + TRACE only.

### Step 2 — Train

Both tracks consume the same feature cache. Checkpoints land in
`models/test_models/` by default (gitignored).

**Neural fusion model** — pick one fusion strategy, or sweep all four
(`concat`, `mlp`, `attention`, `gating`):

```bash
python -m training.train --fusion-method gating
python -m training.train --fusion-method all --epochs 60
```

**Classical classifier** — pick one backend, or sweep every available one
(`xgboost`, `random_forest`, `logreg`, `svm`, `hist_gbm`, `gradient_boosting`):

```bash
python -m training.train_classical --classifier xgboost
python -m training.train_classical --classifier all
```

Classical tree/linear backends additionally report **per-extractor feature
importance** — how much NELA vs. StyleDecipher vs. TRACE drives the prediction.

Each run also writes a `<checkpoint>.metrics.json` with validation metrics and
(for neural runs) the full training history.

### Step 3 — Evaluate on the test split

`evaluate.py` auto-detects the checkpoint type (`.pt` neural or `.joblib`
classical) and reports accuracy, per-class precision/recall/F1, the confusion
matrix, and a per-source-corpus accuracy breakdown:

```bash
python -m test.evaluate --model models/test_models/fusion_gating.pt
python -m test.evaluate --model models/test_models/clf_xgboost.joblib
```

By default it evaluates on `data/features/test.npz`; pass `--features` to
evaluate on a different cached split.

## Tests

```bash
python -m pytest test/                          # fast unit tests
RUN_EXTRACTOR_TESTS=1 python -m pytest test/    # also exercise the real extractors
```

Unit tests (classical track, fusion model, feature dataset) use synthetic data
and always run. The extractor smoke tests download large model weights, so they
only run when `RUN_EXTRACTOR_TESTS=1` is set.

## More Documentation

- [data/README.md](data/README.md) — dataset composition, sources, record
  schema, and the `preprocessing/` loader API.
- [training/README.md](training/README.md) — training framework internals,
  per-file roles, and classifier choices.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request
with your changes.
