# RunPod feature extraction for the arxiv OOD test set

This doc walks through running the full NELA + StyleDecipher (Ollama, 5 LLMs)
+ TRACE feature extraction on `data/testing_dataset/arxiv_final/arxiv_merged.jsonl`
(2 574 records) on a RunPod GPU pod. The output is a single
`data/features/arxiv.npz` that you copy back to your local repo.

The arxiv split is wired into `training/build_dataset.py` as a standalone split
(`--splits arxiv`); it is **not** part of `ALL_SPLITS` and therefore never
appears in any training flow.

---

## 1. Pod configuration

- **Template**: RunPod *PyTorch 2.x + CUDA 12* (any recent variant)
- **GPU**: ≥ 16 GB VRAM (RTX 4090 / A5000 / L4 all sufficient — 5 Ollama models
  fit comfortably; only one is loaded at a time)
- **Container disk**: ≥ 60 GB (Ollama model blobs alone total ~25 GB; SBERT,
  TRACE Roberta, and intermediate caches push it higher)
- **Exposed ports**: TCP `11434` (Ollama HTTP)
- **Environment**:
  - `OLLAMA_HOST=0.0.0.0:11434`
  - (optional) `HF_TOKEN=...` to avoid the unauthenticated-HF rate-limit
    warning when downloading `AnnaWegmann/Style-Embedding` and
    `all-mpnet-base-v2`

---

## 2. Pod setup (run once, top-to-bottom)

The whole flow uses **tmux** so the long-running Ollama server and the
extraction job survive SSH disconnects / browser-tab closes. tmux is
pre-installed on the RunPod PyTorch template; if not, `apt-get update && apt-get install -y tmux`.

```bash
# --- 2.1 repo -------------------------------------------------------------
cd /workspace
git clone <git-remote> modern-AI-detection-trends-comparison
cd modern-AI-detection-trends-comparison

# --- 2.2 python deps ------------------------------------------------------
pip install --upgrade pip
pip install -r requirements.txt

# pre-fetch the two tokenisation resources the extractors require at runtime
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
python -m spacy download en_core_web_sm

# --- 2.3 ollama install ---------------------------------------------------
curl -fsSL https://ollama.com/install.sh | sh
```

### 2.4 Start the Ollama server in a tmux session

Keep the server in its own tmux session so model loads/swaps + crash logs are
inspectable without blocking your shell.

```bash
tmux new-session -d -s ollama \
    "OLLAMA_HOST=0.0.0.0:11434 \
     OLLAMA_KEEP_ALIVE=24h \
     OLLAMA_NUM_PARALLEL=1 \
     OLLAMA_MAX_LOADED_MODELS=5 \
     ollama serve 2>&1 | tee /tmp/ollama.log"

sleep 5
curl -s http://127.0.0.1:11434/api/tags    # smoke-test: should return JSON
```

The env vars matter: `KEEP_ALIVE=24h` stops Ollama from unloading models
between requests, and `MAX_LOADED_MODELS=5` keeps all 5 resident (needs ≥ 32 GB
VRAM — drop to 2 on a 24 GB card like the RTX 4090).

### 2.5 Pull the five StyleDecipher models

```bash
# These names match `MODELS` in extractors/styledecipher_extractor.py.
for m in llama3 mistral gemma phi3 qwen2; do
    ollama pull "$m"
done
ollama list      # confirm all 5 are present
```

Expected `ollama list` output: five rows, one per model name above. If any are
missing, re-run that `ollama pull` — it resumes.

### 2.6 tmux cheat sheet

| Action | Keys / command |
|---|---|
| Detach from current session (leave it running) | `Ctrl-b` then `d` |
| List sessions | `tmux ls` |
| Reattach to the ollama server | `tmux attach -t ollama` |
| Reattach to the extraction job | `tmux attach -t extract` |
| Scroll back in a session | `Ctrl-b` then `[` (use arrows / PgUp; press `q` to exit scroll mode) |
| Kill a session by name | `tmux kill-session -t <name>` |

---

## 3. The single extraction command (in tmux)

Run the extraction in its **own** tmux session — separate from the ollama
server — so closing the browser / losing the SSH connection doesn't kill the
job.

```bash
tmux new-session -d -s extract \
    "cd /workspace/modern-AI-detection-trends-comparison && \
     python -m training.build_dataset \
        --splits arxiv \
        --styledecipher ollama \
        --trace-context author \
        --device cuda \
        --checkpoint-every 25 \
        2>&1 | tee /tmp/extract.log"
```

Attach to watch the tqdm progress bar:

```bash
tmux attach -t extract
# detach again with: Ctrl-b  d
```

What the command does:

- Loads only `data/testing_dataset/arxiv_final/arxiv_merged.jsonl` (no
  train/val/test loaded — arxiv is short-circuited).
- Writes incrementally to `data/features/arxiv.npz`; checkpoints atomically
  every 25 records.
- TRACE uses same-author siblings *within the arxiv set* (99 authors, every
  paper has at least one sibling).
- StyleDecipher rewrites are generated live by Ollama (5 LLMs × 1 rewrite per
  record).

**Resumable**: if the pod is interrupted or you kill the tmux session, just
re-run the exact same `tmux new-session …` block. Already-cached records are
skipped and extraction resumes from the last checkpoint.

### 3.1 Monitor without attaching

If you don't want to attach the full tmux session, tail the log directly:

```bash
tail -f /tmp/extract.log         # live progress
grep -c '^  \[arxiv\]' /tmp/extract.log   # rough count of milestone lines

# or peek at the cache as it grows
python - <<'PY'
import numpy as np
d = np.load('data/features/arxiv.npz', allow_pickle=True)
print('cached so far:', d['ids'].shape[0], '/ 2574')
PY
```

### 3.2 When the job finishes

The extract tmux session exits on its own when `build_dataset.py` returns.
Check it ran cleanly:

```bash
tmux ls                                 # should no longer list "extract"
tail -20 /tmp/extract.log               # should end with "Wrote feature caches + meta.json to ..."
python - <<'PY'
import numpy as np
d = np.load('data/features/arxiv.npz', allow_pickle=True)
print('final rows:', d['ids'].shape[0])           # expect 2574
print('shapes:', d['nela'].shape, d['style'].shape, d['trace'].shape)
print('style coverage:', d['style_ok'].mean())    # expect ~1.0
PY
```

Then stop the ollama tmux session to free the GPU before you `runpodctl send`:

```bash
tmux kill-session -t ollama
```

---

## 4. Expected duration

Per-record cost is dominated by the 5 Ollama generations. Local CPU smoke run
measured ~5.5 s/rec for **NELA + TRACE only** (no Ollama); on a 16 GB GPU with
Ollama generation on a paragraph-length text, expect each of the 5 LLM calls
to take roughly 3–8 s, so the full per-record cost lands around **30–60 s**.

Total estimate for 2 574 records:

| Per-rec time | Total wall-clock |
|--------------|------------------|
| 30 s         | ~21 h            |
| 45 s         | ~32 h            |
| 60 s         | ~43 h            |

**Plan for ~24–36 h on a single GPU.** If it overshoots, the checkpoint just
keeps going — pause / restart costs are negligible because the run is
resumable.

Assumptions: average arxiv abstract ~150 words; one rewrite per model;
SBERT/TRACE inference is sub-second per record on a modern GPU and is not the
bottleneck.

---

## 5. Output — what to download back

A single file:

```
data/features/arxiv.npz
```

Pull it back to your local machine via either:

```bash
# option A: runpodctl (works from the pod terminal)
runpodctl send data/features/arxiv.npz

# option B: scp from your laptop (replace <pod-host> + <port>)
scp -P <port> root@<pod-host>:/workspace/modern-AI-detection-trends-comparison/data/features/arxiv.npz \
    ./data/features/arxiv.npz
```

(`data/features/meta.json` will also be rewritten on the pod; you can ignore
it — local `meta.json` should not be overwritten with arxiv info per the
project workflow.)

---

## 6. Common pitfalls + quick checks

| Pitfall | Quick check | Fix |
|---------|-------------|-----|
| NLTK `punkt` resource missing | `python -c "import nltk; nltk.data.find('tokenizers/punkt')"` | `python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"` |
| spaCy `en_core_web_sm` missing | `python -c "import spacy; spacy.load('en_core_web_sm')"` | `python -m spacy download en_core_web_sm` |
| Ollama server not running | `curl -s http://127.0.0.1:11434/api/tags` (should JSON-list models) | re-launch the `tmux new-session -d -s ollama …` block in §2.4 |
| Only some models pulled | `ollama list` (expect 5 rows) | `ollama pull <missing-name>` |
| GPU not visible to PyTorch | `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` | re-select a CUDA template / re-attach GPU |
| `data/features/arxiv.npz` not appearing | `tail -f /tmp/extract.log` — look for the tqdm bar lines; file is written atomically every 25 records | wait for the first checkpoint; `ls -la data/features/` |
| `tmux` not installed on this template | `which tmux` | `apt-get update && apt-get install -y tmux` |
| `extract` tmux session vanished from `tmux ls` | `tail -50 /tmp/extract.log` — look for a stack trace or "Wrote feature caches …" | if it errored, fix the root cause and re-launch the §3 tmux command (resumes from checkpoint); if it completed, you're done |
| Ollama keeps unloading models between requests (slow) | `tmux attach -t ollama` — look for repeated "loading model" lines | confirm `OLLAMA_KEEP_ALIVE=24h` and `OLLAMA_MAX_LOADED_MODELS=5` are set in the §2.4 tmux command; if GPU < 32 GB, drop to `MAX_LOADED_MODELS=2` |
| Lost the SSH connection mid-run | both tmux sessions keep running on the pod — no action needed | `ssh` back in, `tmux ls`, `tmux attach -t extract` to confirm progress |
