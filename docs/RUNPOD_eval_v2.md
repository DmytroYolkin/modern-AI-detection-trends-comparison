# RunPod eval v2 — 4 paper-grade baselines on arxiv (clean + humanized)

Successor to the original [RUNPOD_humanizers_and_baselines.md](RUNPOD_humanizers_and_baselines.md). The humanizers and the in-house feature extractions are already done (pods 1, 2 v1, 3); R-Detect and RAIDAR are running locally on the laptop (see [INTERIM_RESULTS_LOCAL.md](INTERIM_RESULTS_LOCAL.md) — wait, that file got renamed to [models/interim_results.ipynb](../models/interim_results.ipynb)).

What this pod runs: the **4 remaining baseline detectors** at **paper-grade configuration**, on both clean arxiv and humanized arxiv-AI-only.

| detector | paper-grade config | per-rec | total wall-clock (5090) |
|---|---|---:|---:|
| `fast_detect_gpt` | GPT-Neo-2.7B + GPT-J-6B | ~1 s | ~1.5 h |
| `binoculars` | Falcon-7B + Falcon-7B-instruct, **4-bit** | ~1.5 s | ~2 h |
| `detect_gpt` | t5-large + GPT-Neo-2.7B, **N=10 perturbations** | ~3 s | ~4 h |
| `radar` | TrustSafeAI/RADAR-Vicuna-7B | ~0.5 s | ~0.75 h |
| **total** | | | **~8 h, ~$6.50 on 5090 community** |

`detect_gpt` is set to N=10 (not paper's N=100) so the whole run fits a 15 h budget with margin. Reproducing the paper's N=100 number on a longer pod session is a separate, optional exercise.

---

## 1. Pod configuration

- **Template**: RunPod *PyTorch 2.x + CUDA 12*
- **GPU**: **5090 (32 GB) recommended.** A6000 (48 GB) also works (slightly slower per record but more VRAM headroom). 4090 (24 GB) is **tight** — Falcon-7B 4-bit fits but GPT-J-6B + GPT-Neo-2.7B co-resident may push it over; you'd need to load each detector then close before the next.
- **Container disk**: ≥ 60 GB (the HF cache is redirected to `/workspace` per §1a so the small root partition isn't the bottleneck; but the `/workspace` volume still needs the room — GPT-J-6B is ~12 GB, Falcon-7B-pair ~28 GB on disk even though 4-bit at load time, RADAR ~14 GB, GPT-Neo-2.7B ~6 GB, t5-large ~3 GB).
- **Exposed ports**: none needed (no Ollama in this pod — RAIDAR is local).
- **Environment**:
  - `HF_TOKEN=...` — recommended (avoids the unauthenticated-HF rate-limit warning on big downloads)
  - `HF_HUB_ENABLE_HF_TRANSFER=1` — faster downloads

---

## 2. Setup (run once, top-to-bottom)

### 2.1 Clone repo

```bash
cd /workspace
git clone https://github.com/DmytroYolkin/modern-AI-detection-trends-comparison.git
cd modern-AI-detection-trends-comparison

# sanity-check the two input JSONLs are there (committed) and the lean config too
ls -la data/testing_dataset/arxiv_final/arxiv_merged.jsonl
ls -la data/testing_dataset/arxiv_final/arxiv_humanized_ai_only.jsonl
ls -la scripts/baselines_paper_faithful_lean.json
ls -la scripts/pod_eval_v2.sh
```

### 2.2 Python deps

```bash
pip install --upgrade pip
pip install pandas numpy tqdm
pip install torch "transformers<5.5.0" accelerate sentencepiece
pip install scikit-learn requests
pip install bitsandbytes        # for Binoculars 4-bit loading
pip install hf_transfer         # faster HF downloads
```

(Note: no `nltk`/`spaCy`/`ollama` here — those were only needed by pod 1's feature extraction and the RAIDAR/R-Detect baselines, which are not on this pod.)

### 2.3 HuggingFace login (required for gated Falcon-7B)

```bash
hf auth login   # paste HF_TOKEN; needed for tiiuae/falcon-7b and falcon-7b-instruct
```

### 2.4 Redirect HF cache to `/workspace`

This is the **single biggest pod-2-v1 lesson** — Falcon-7B + GPT-J-6B + RADAR-Vicuna-7B together filled the small `/root` partition and made every subsequent download fail. Set this **before** any model download:

```bash
echo 'export HF_HOME=/workspace/.hf_cache' >> ~/.bashrc
echo 'export HF_HUB_CACHE=/workspace/.hf_cache/hub' >> ~/.bashrc
source ~/.bashrc
mkdir -p "$HF_HUB_CACHE"
```

The wrapper script in §3 also exports these, so even if you skip the `.bashrc` step the run itself is safe — but the export to `.bashrc` is convenient for any manual checks you do.

---

## 3. Single fire-and-forget command

```bash
tmux new-session -d -s eval_v2 \
    "bash scripts/pod_eval_v2.sh 2>&1 | tee /tmp/eval_v2.log"

tmux attach -t eval_v2   # Ctrl-b d to detach
```

Monitor without attaching:
```bash
tail -f /tmp/eval_v2.log
```

The wrapper does both stages (clean then humanized AI-only) with `--skip-if-exists`, so if it dies mid-way for any reason, the same command resumes from the last completed detector. **It exits non-zero if any detector JSON contains an `"error"` field** — so unlike pod-2 v1, "ALL DONE" actually means done.

Expected wall-clock end-to-end: ~8 h on 5090, slightly more on A6000, considerably more on 4090.

---

## 4. Download results back

When `/tmp/eval_v2.log` ends with `ALL DONE (real this time ...)` and exit code 0:

```bash
# Tar each results directory (runpodctl wants single files)
tar -czf /workspace/bl_arxiv_clean_v2.tar.gz models/baseline_results/arxiv_clean/
tar -czf /workspace/bl_arxiv_humanized_v2.tar.gz models/baseline_results/arxiv_humanized/

# Send each; paste the codes back to your local session (HOME= avoids the
# `/root/.runpod` write that fails on the small partition -- pod-2 v1 lesson).
HOME=/workspace runpodctl send /workspace/bl_arxiv_clean_v2.tar.gz
HOME=/workspace runpodctl send /workspace/bl_arxiv_humanized_v2.tar.gz
```

On local:
```bash
./runpodctl.exe receive <code1>
./runpodctl.exe receive <code2>

# Untar into the matching local dirs (will merge with the small-config JSONs
# already there from the laptop runs; paper-grade JSONs have the same
# filenames so they OVERWRITE the small-config ones -- which is what you want).
tar -xzf bl_arxiv_clean_v2.tar.gz
tar -xzf bl_arxiv_humanized_v2.tar.gz
```

---

## 5. After the run — assembling the final comparison

Local notebook [models/interim_results.ipynb](../models/interim_results.ipynb) Sections E and F will, when re-executed, automatically pick up:

- the paper-grade `fast_detect_gpt`, `binoculars`, `detect_gpt`, `radar` JSONs (this pod)
- the local `r_detect` JSONs (laptop, ~2.7 h)
- the local `raidar` JSONs (laptop overnight, ~13 h)
- the 5 in-house models (this laptop, instant once the full humanized .npz lands)

That's **5 in-house + 5 baselines × 2 eval sets** — the full headline comparison table.

---

## 6. Common pitfalls

| Pitfall | Quick check | Fix |
|---|---|---|
| Falcon-7B download 401 unauthenticated | `hf auth whoami` | accept the gated-model terms on HF; re-login |
| `bitsandbytes` import error | `python -c "import bitsandbytes"` | `pip install bitsandbytes` (and `pip install -U bitsandbytes` if cuda 12.x mismatch) |
| `/root` full mid-run | `df -h /` | the script's HF_HUB_CACHE export should prevent this. If it still happens, `rm -rf /root/.cache/huggingface` (the offending downloads are now in /workspace) and re-run with `--skip-if-exists` |
| GPT-J-6B OOM | `nvidia-smi` while it's loading | upgrade to A6000 (48 GB), OR add Fast-DetectGPT 8-bit override to the lean config |
| Pipeline exits 3 with detector errors | `grep '"error"' models/baseline_results/*/arxiv_*__*.metrics.json` | inspect the error message; usually a missing dep or VRAM issue. Fix and re-run the same tmux command (resumes) |
