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

# --- 2.3 ollama server ----------------------------------------------------
curl -fsSL https://ollama.com/install.sh | sh
export OLLAMA_HOST=0.0.0.0:11434
nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 5
curl -s http://127.0.0.1:11434/api/tags   # smoke-test the server

# --- 2.4 pull the five StyleDecipher models -------------------------------
# These names match `MODELS` in extractors/styledecipher_extractor.py.
# Ollama resolves each bare name to its current `:latest` tag.
for m in llama3 mistral gemma phi3 qwen2; do
    ollama pull "$m"
done
ollama list      # confirm all 5 are present
```

Expected `ollama list` output: five rows, one per model name above. If any are
missing, re-run that `ollama pull` — it resumes.

---

## 3. The single extraction command

```bash
python -m training.build_dataset \
    --splits arxiv \
    --styledecipher ollama \
    --trace-context author \
    --device cuda \
    --checkpoint-every 25
```

What this does:

- Loads only `data/testing_dataset/arxiv_final/arxiv_merged.jsonl` (no
  train/val/test loaded — arxiv is short-circuited).
- Writes incrementally to `data/features/arxiv.npz`; checkpoints atomically
  every 25 records.
- TRACE uses same-author siblings *within the arxiv set* (99 authors, every
  paper has at least one sibling).
- StyleDecipher rewrites are generated live by Ollama (5 LLMs × 1 rewrite per
  record).

**Resumable**: if the pod is interrupted, just re-run the exact same command.
Already-cached records are skipped and extraction resumes.

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
| Ollama server not running | `curl -s http://127.0.0.1:11434/api/tags` (should JSON-list models) | `nohup ollama serve > /tmp/ollama.log 2>&1 &` |
| Only some models pulled | `ollama list` (expect 5 rows) | `ollama pull <missing-name>` |
| GPU not visible to PyTorch | `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` | re-select a CUDA template / re-attach GPU |
| `data/features/arxiv.npz` not appearing | check pod stdout for the tqdm bar lines; the file is written atomically every 25 records | wait for the first checkpoint; `ls -la data/features/` |
