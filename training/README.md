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
        └─ train_classical.py  → classical : XGBoost / RF / LogReg / SVM / MLP / GBM  .joblib
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
| `classical.py`         | `ClassicalClassifier` — XGBoost / RF / LogReg / SVM / MLP / GBM wrapper |
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
`logreg`, `svm`, `mlp`, `hist_gbm`, `gradient_boosting`. Tree and linear
backends report **per-extractor feature importance** (how much NELA vs
StyleDecipher vs TRACE drives the prediction); `svm` and `mlp` do not.
`xgboost` needs the `xgboost` package; missing backends are skipped
automatically under `--classifier all`.

`mlp` is `sklearn`'s `MLPClassifier` — a plain multi-layer perceptron on the
flat 225-dim feature vector. It differs from the *neural* track's `mlp` fusion
method, which combines the three streams inside `MultiFeatureFusion` before a
head. The classical `mlp` has no `sample_weight`, so class balance is handled
by oversampling the minority class.

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

A resume only re-runs records that were *never* cached. Records that were
cached with **failed features** — a zero vector from a crashed extractor or an
unreachable Ollama server — are *not* retried. After every build, verify
coverage (see [If the build fails](#if-the-build-fails)) before trusting the
cache.

## Building on a cloud GPU (RunPod RTX 5090)

The `--styledecipher ollama` build is GPU-heavy — roughly 18–24 h on an
RTX 5090, much longer on smaller cards. To run it on a rented GPU:

1. **Pod** — RunPod RTX 5090 (32 GB), a CUDA 12.8+ PyTorch template, and a
   persistent volume mounted at `/workspace`. Clone the repo *into the volume*
   so a pod restart resumes the build instead of restarting it.

2. **Ollama** — install it, then export these *before* starting the server so
   all five rewrite models stay resident in the 32 GB of VRAM (this is what
   keeps the 5090 fast — no model swapping):

   ```bash
   export OLLAMA_MODELS=/workspace/ollama-models
   export OLLAMA_KEEP_ALIVE=-1          # never evict a loaded model
   export OLLAMA_MAX_LOADED_MODELS=5    # keep all 5 resident

   # supervised: if the server ever dies, it restarts itself within ~2 s.
   # A dead Ollama silently caches empty StyleDecipher features (see below).
   ( while true; do ollama serve >> /workspace/ollama.log 2>&1; sleep 2; done ) &

   for m in llama3 mistral gemma phi3 qwen2; do ollama pull $m; done
   ```

3. **Deps** — install the packages from the root `README.md`; point the
   Hugging Face cache at the volume too (`export HF_HOME=/workspace/hf-cache`).

4. **Build** — run inside `tmux` so it survives SSH disconnects. Smoke-test on
   50 records first, then launch the full build under an auto-resume loop so a
   transient crash restarts itself. The smoke-test records are valid `ollama`
   features and get reused — no `--restart` needed:

   ```bash
   # smoke test — confirms the pipeline works end to end
   python -m training.build_dataset --splits test --limit 50 --styledecipher ollama

   # full build — reruns itself (resuming from the cache) until it exits clean
   set -o pipefail
   until python -m training.build_dataset --splits all --styledecipher ollama \
         2>&1 | tee -a /workspace/build.log; do
     echo "build exited non-zero — resuming in 10 s ..." | tee -a /workspace/build.log
     sleep 10
   done
   ```

5. **Save** — the feature cache is small (a few hundred MB). Pull it off the pod
   before terminating:

   ```bash
   tar czf features.tar.gz data/features
   runpodctl send features.tar.gz       # then `runpodctl receive <code>` locally
   ```

Then **terminate the pod** to stop billing. Training (`train.py` /
`train_classical.py`) is light enough to run locally on the downloaded `.npz`.

### If the build fails

`build_dataset.py` is resumable, so most failures recover by simply **rerunning
the same command** — cached records are skipped and extraction continues. The
auto-resume loop in step 4 does this for you. What each failure needs:

| Symptom | Cause | What to do |
|---|---|---|
| SSH dropped, `tmux` session still alive | disconnect only | `tmux attach -t build` — the build never stopped |
| Pod stopped / restarted | spot reclaim, host crash | reconnect, re-export the step 2 env vars, restart Ollama + the build loop |
| Build process exited non-zero | OOM, transient extractor error | the step 4 auto-resume loop reruns it; or rerun the command yourself |
| `build.log` shows `ollama rewrite failed for …` | Ollama server was down | the supervisor restarts Ollama in ~2 s — but see the caveat below |

**The one case a plain resume does not fix.** Records processed *while Ollama
was unreachable* get cached with an **empty StyleDecipher vector**
(`style_ok = False`), and a resume will **not** retry them. Always verify
coverage once the build finishes:

```bash
python -c "import json; m=json.load(open('data/features/meta.json')); print(m['splits'])"
```

In `--styledecipher ollama` mode every split's `style_coverage` should be
≈ 1.0. If a split is materially lower (or `build.log` ends with
`completed with N extraction failure(s)`), rebuild **that split only** from
scratch — `--restart` clears its cache so every record is re-extracted against
a healthy Ollama:

```bash
python -m training.build_dataset --splits train --styledecipher ollama --restart
```

## Models folder

- `models/ready_models/` — validated, **committed** checkpoints.
- `models/test_models/`  — experimental checkpoints, **gitignored** (default
  output of `train.py`). Promote a good run by copying it into `ready_models/`.

## Tests

```bash
python -m pytest test/                       # fast unit tests
RUN_EXTRACTOR_TESTS=1 python -m pytest test/ # also exercise the real extractors
```
