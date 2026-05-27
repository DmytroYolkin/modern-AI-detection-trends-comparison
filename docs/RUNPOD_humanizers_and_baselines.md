# RunPod: arxiv humanizers + baseline detector sweep

This is the runbook for the **second** RunPod pod (the first one runs
the NELA + StyleDecipher + TRACE feature extraction in
[RUNPOD_arxiv_extraction.md](RUNPOD_arxiv_extraction.md)). On this pod
we (a) paraphrase the 1 287 Claude-haiku-rewritten arxiv abstracts with
two SOTA humanizers, then (b) score both the clean and the humanized
arxiv evaluation sets against the six external baseline detectors.

What you run here is independent of the feature-extraction pod — you can
do this in parallel with that pod, or after it finishes, or skip the
extraction entirely if you already have `arxiv.npz` cached.

---

## 1. Pod configuration

- **Template**: RunPod *PyTorch 2.x + CUDA 12* (any recent variant)
- **GPU**: **RTX 4090 (24 GB)**, **RTX 5090 (32 GB)**, or **A6000 (48 GB)**.
  - On a 24 GB card (4090), each detector is loaded then closed before
    the next — only one detector's models occupy VRAM at a time, so
    24 GB is sufficient if the per-detector configs stay within that budget.
    The `scripts/baselines_paper_faithful.json` was tuned for 24 GB:
    Binoculars uses Falcon-7B in 4-bit (~8 GB), DetectGPT uses t5-large
    + GPT-Neo-2.7B (~10 GB), Fast-DetectGPT uses GPT-Neo-2.7B + GPT-J-6B
    (~18 GB — tight but fits).
  - LLaMA-3-8B-Instruct for Adv-P + OpenAI-RoBERTa-large guidance ~18 GB
    co-resident → fine on 24 GB.
  - TempParaphraser is also ~8B → ~16 GB; humanizers must be serialised
    on 24 GB (see §3a). 32 GB or 48 GB cards can run them in parallel.
  - **Recommendation**: 24 GB minimum, 32 GB comfortable.
- **Container disk**: ≥ 60 GB.
  - LLaMA-3-8B-Instruct ~16 GB + TempParaphraser ~16 GB + paper-faithful
    baseline weights (GPT-J-6B ~12 GB, Falcon-7B-pair ~28 GB w/ 4-bit
    loading reducing on-disk to ~28 GB still — quantization is at load
    time), RADAR-Vicuna-7B ~14 GB, Ollama llama3 ~5 GB, + cache headroom.
- **Exposed ports**: TCP **11434** (Ollama, needed by the RAIDAR baseline).
- **Environment**:
  - `HF_TOKEN=...`  -- **required** for `meta-llama/Meta-Llama-3-8B-Instruct`
    (it is a gated model on HuggingFace).
  - `HF_HUB_ENABLE_HF_TRANSFER=1` (faster downloads).
  - `OLLAMA_HOST=0.0.0.0:11434` (needed only for the RAIDAR baseline,
    see §2.4 below).

---

## 2. Setup

The whole flow uses **tmux** so the long humanizer jobs survive SSH
disconnects. tmux is pre-installed on the RunPod PyTorch template;
otherwise `apt-get update && apt-get install -y tmux`.

### 2.1 Clone the repo

```bash
cd /workspace
git clone https://github.com/DmytroYolkin/modern-AI-detection-trends-comparison.git
cd modern-AI-detection-trends-comparison

# sanity-check the input file the humanizers depend on
ls -la data/testing_dataset/arxiv_final/arxiv_merged.jsonl
grep -c '"source": "arxiv_rewrite"' data/testing_dataset/arxiv_final/arxiv_merged.jsonl
# expect: 1287
```

### 2.2 Python deps

```bash
pip install --upgrade pip
pip install pandas numpy nltk tqdm
pip install torch "transformers<5.5.0" accelerate sentencepiece
pip install scikit-learn clean-text regex
pip install hf_transfer            # faster HF downloads
# detector-specific extras (only needed for §5)
pip install requests bitsandbytes  # for Binoculars / RADAR / R-Detect

python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
# spaCy is NOT required on pod #2 -- NELA's spaCy dependency is only used
# during feature extraction, which runs on pod #1. Humanizers + baselines
# do not touch spaCy.
```

### 2.3 HuggingFace login

```bash
hf auth login   # paste HF_TOKEN; required for gated Meta-LLaMA-3
```

(The legacy `huggingface-cli login` is deprecated as of late 2025; the new
binary is `hf`. Both ship with the same `huggingface_hub` pip install, so
no extra install needed.)

### 2.4 Ollama install + start (needed only by RAIDAR baseline at §5)

RAIDAR rewrites every candidate text via an LLM and then measures edit
distance to the rewrite — that LLM is `llama3` served by Ollama. The
other 5 baselines don't need Ollama.

```bash
curl -fsSL https://ollama.com/install.sh | sh

# start Ollama in its own tmux session so it survives SSH disconnects
tmux new-session -d -s ollama \
    "OLLAMA_HOST=0.0.0.0:11434 \
     OLLAMA_KEEP_ALIVE=24h \
     ollama serve 2>&1 | tee /tmp/ollama.log"

sleep 5
curl -s http://127.0.0.1:11434/api/tags    # JSON response = server up

# pull the single model RAIDAR uses
ollama pull llama3
ollama list                                # should list llama3
```

This is fire-and-forget: leave the `ollama` tmux session running through
the entire pod #2 lifetime. The RAIDAR baseline will connect on
`localhost:11434` when it's invoked at §5.

---

## 3. ONE-COMMAND PATH — fire-and-forget the whole pod (recommended)

After §1–§2 setup is done, **a single command runs everything**: both
humanizers (serial), the merge step, baselines on the clean set, and
baselines on the humanized AI-only set. It is fully resumable — if the
pod is interrupted or you kill the tmux session, just rerun the exact
same command and it picks up where it stopped:

- Humanizers skip records whose `source_text_id` already appears in their
  output JSONL.
- The merge step is idempotent.
- `compare_baselines.py --skip-if-exists` skips detectors whose successful
  output JSON already exists.

*(If the pipeline is currently running but a humanizer failed and it incorrectly moved on, kill it with `tmux kill-session -t pipeline` and then re-run the command below. It will safely skip what's already done.)*

```bash
tmux new-session -d -s pipeline \
    "cd /workspace/modern-AI-detection-trends-comparison && \
     bash scripts/pod2_full_pipeline.sh \
     2>&1 | tee /tmp/pipeline.log"

# attach to watch progress; Ctrl-b d to detach
tmux attach -t pipeline
```

Monitor without attaching:
```bash
tail -f /tmp/pipeline.log
```

The wrapper logs each of its 5 stages with timestamps and continues past
per-detector failures (a single OOM doesn't abort the rest). When it
finishes, the last log line lists the artifacts to send back to local.

**Expected wall-clock on RTX 4090 (24 GB): ~19–21 h**. Resume costs are
near-zero — re-running the command after an interruption only redoes the
single record / detector that was in flight when it died.

The remaining subsections (§3a, §3b, §4, §5) document the **same work
broken into manual steps** — useful for debugging, smoke-testing, or
running a subset. If you used §3's one-command path, jump directly to
§6 (Download results back).

---

## 3a. Run the humanizers in tmux (manual, advanced)

The two driver scripts handle the upstream clone + resumable JSONL writing
themselves — you only need to invoke them.

### 3a. Serial (24 GB GPU, recommended)

Run them one after the other so only one big LM is GPU-resident at a time:

```bash
tmux new-session -d -s humanize-adv \
    "cd /workspace/modern-AI-detection-trends-comparison && \
     python scripts/humanize_arxiv_adversarial.py --device cuda \
     2>&1 | tee /tmp/adv.log"

# attach to watch progress; Ctrl-b d to detach
tmux attach -t humanize-adv

# when adv is done:
tmux new-session -d -s humanize-temp \
    "cd /workspace/modern-AI-detection-trends-comparison && \
     python scripts/humanize_arxiv_temppara.py --device cuda \
     2>&1 | tee /tmp/temp.log"
```

### 3b. Parallel (≥ 40 GB GPU only)

If you provisioned an A6000 (48 GB) or two GPUs, you can run both at
once. **Do not do this on a 5090 (32 GB)** — both 8 B models in fp16 will
OOM as soon as the baselines pile on at §5.

```bash
tmux new-session -d -s humanize-adv \
    "cd /workspace/modern-AI-detection-trends-comparison && \
     python scripts/humanize_arxiv_adversarial.py --device cuda \
     2>&1 | tee /tmp/adv.log"

tmux new-session -d -s humanize-temp \
    "cd /workspace/modern-AI-detection-trends-comparison && \
     python scripts/humanize_arxiv_temppara.py --device cuda \
     2>&1 | tee /tmp/temp.log"
```

### Both humanizers are **resumable**

If the pod is interrupted (kill, OOM, etc.) just re-run the same `tmux
new-session …` command — rows whose `source_text_id` already appears in
the output JSONL are skipped.

### Expected duration

Estimated on a single RTX 5090 (32 GB), LLaMA-3-8B-Instruct in fp16,
arxiv abstracts averaging ~150 words:

| Humanizer                | Per-record cost | Total (1287 records) |
|--------------------------|-----------------|----------------------|
| Adversarial Paraphrasing | ~6–10 s/rec     | **~2.5–3.5 h**       |
| TempParaphraser          | ~3–5 s/rec      | **~1.5–2.0 h**       |

Adv-P is slower because of the per-token beam re-scoring with the
guidance detector. Serial total: **plan for ~4–6 h on a 5090** to
finish both humanizers.

### Monitor

```bash
tail -f /tmp/adv.log
tail -f /tmp/temp.log

# count rows already written:
wc -l data/testing_dataset/arxiv_final/arxiv_humanized_adv.jsonl
wc -l data/testing_dataset/arxiv_final/arxiv_humanized_temp.jsonl
```

Each file should end at 1287 lines.

---

## 4. Merge into the evaluation set

After **both** humanizers finish:

```bash
python scripts/merge_humanized_evalset.py
```

The script prints a summary; you should see:

```
total rows:           3861
label distribution:   {'human': 1287, 'ai': 2574}
per-source counts:
  arxiv                       1287
  arxiv_humanized_adv         1287
  arxiv_humanized_temp        1287
authors:              ~99
author-coverage check OK (>= 2 per author)
```

If one of the humanizers stalled and you want to merge what you have,
pass `--allow-missing-humanizer`.

---

## 5. Baseline detectors on the two evaluation sets

The extended `test.compare_baselines` takes `--input-jsonl <path>`; output
filenames are auto-prefixed with the input stem so the two runs don't
overwrite each other. `--detector-config` points at a JSON file mapping
`{detector_name: {kwarg: value, ...}}`; we use
`scripts/baselines_paper_faithful.json` to swap the wrapper defaults for the
checkpoints each paper actually used (Falcon-7B pair for Binoculars, GPT-Neo-2.7B
+ T5-large + 100 perturbations for DetectGPT, GPT-Neo-2.7B + GPT-J-6B for
Fast-DetectGPT). RADAR / R-Detect / RAIDAR defaults are already paper-faithful
(RADAR-Vicuna-7B, vendored R-Detect, llama3 via Ollama) and are left alone.

**Human-row reuse.** The 1287 human (`source == "arxiv"`) rows in
`arxiv_eval_with_humanizers.jsonl` are **byte-identical** to the 1287 human
rows in `arxiv_merged.jsonl` (the merge script just concatenates them through).
Re-scoring them with the slow baselines (DetectGPT at 100 perturbations
especially) would waste ~33% of pod-#2 time on identical inputs. So the
humanized baseline sweep runs on **`arxiv_humanized_ai_only.jsonl`** (2574 AI
rows: Adv-P + Temp, no humans), and `test/evaluate_arxiv.py` later merges
those 2574 AI scores with the 1287 human scores from the clean run to
reconstruct the 3861-row metric block for the humanized evaluation.

```bash
# clean arxiv (humans + Claude-haiku rewrites, 2574 rows) -- unchanged
python -m test.compare_baselines \
    --detectors all \
    --input-jsonl data/testing_dataset/arxiv_final/arxiv_merged.jsonl \
    --output models/baseline_results/arxiv_clean/ \
    --detector-config scripts/baselines_paper_faithful.json

# humanized arxiv AI ONLY (2574 rows; humans reused from clean run)
python -m test.compare_baselines \
    --detectors all \
    --input-jsonl data/testing_dataset/arxiv_final/arxiv_humanized_ai_only.jsonl \
    --output models/baseline_results/arxiv_humanized/ \
    --detector-config scripts/baselines_paper_faithful.json
```

`--detectors all` runs the six registered baselines:

  1. `fast_detect_gpt`
  2. `detect_gpt`
  3. `binoculars`
  4. `r_detect`
  5. `radar`
  6. `raidar`

Each detector writes one
`models/baseline_results/<set>/<input-stem>__<detector>.metrics.json`.
A per-detector failure (missing dep, OOM, API key) is logged into the
JSON for that detector and the sweep continues.

Expected wall-clock: ~3 h on the clean set and ~6 h on the humanized
AI-only set (2574 rows on each side), depending on which detectors load
the biggest LMs. Run this in tmux too:

```bash
tmux new-session -d -s baselines-clean \
    "cd /workspace/modern-AI-detection-trends-comparison && \
     python -m test.compare_baselines --detectors all \
       --input-jsonl data/testing_dataset/arxiv_final/arxiv_merged.jsonl \
       --output models/baseline_results/arxiv_clean/ \
       --detector-config scripts/baselines_paper_faithful.json \
     2>&1 | tee /tmp/bl_clean.log"
```

---

## 6. Download results back

```bash
# the two humanizer outputs
runpodctl send data/testing_dataset/arxiv_final/arxiv_humanized_adv.jsonl
runpodctl send data/testing_dataset/arxiv_final/arxiv_humanized_temp.jsonl
runpodctl send data/testing_dataset/arxiv_final/arxiv_eval_with_humanizers.jsonl

# the two baseline-result directories (tar them first; runpodctl wants files)
tar -czf bl_arxiv_clean.tar.gz models/baseline_results/arxiv_clean/
tar -czf bl_arxiv_humanized.tar.gz models/baseline_results/arxiv_humanized/
runpodctl send bl_arxiv_clean.tar.gz
runpodctl send bl_arxiv_humanized.tar.gz
```

On your local machine:

```powershell
.\runpodctl.exe receive <code>
```

Then drop the JSONLs back into
`data/testing_dataset/arxiv_final/` and untar the result tarballs into
`models/baseline_results/`.

---

## 7. Estimated cost & duration

Assuming RTX 5090 at ~$0.79/h on RunPod community-cloud (Q4-2025
price; verify on the dashboard):

| Stage                              | Wall-clock | Cost (5090) |
|------------------------------------|------------|-------------|
| Setup + HF login + downloads       | ~0.5 h     | ~$0.40      |
| Adv-P humanizer (1287 records)     | ~3 h       | ~$2.40      |
| TempParaphraser humanizer (1287)   | ~1.8 h     | ~$1.40      |
| Baselines on `arxiv_clean` (6×)    | ~3 h       | ~$2.40      |
| Baselines on `arxiv_humanized`     | ~5 h       | ~$4.00      |
| Buffer (debug / re-runs)           | ~1 h       | ~$0.80      |
| **Total**                          | **~14 h**  | **~$12**    |

On a 48 GB A6000 (~$0.49/h) running both humanizers in parallel:
~9–10 h wall-clock, ~$5. Trade-off: bigger card = more $/h but less
calendar time and less risk of a baseline OOM-ing because it can't find
free VRAM.

Be honest with yourself about overhead: the first time you do this on a
new pod, expect ~50 % more wall-clock than the table above. Subsequent
runs (after `huggingface_hub` cache is primed and the upstream repos
are cloned) match the estimates.

---

## 8. Common pitfalls + quick checks

| Pitfall | Quick check | Fix |
|---------|-------------|-----|
| LLaMA-3 download fails with 401 | `huggingface-cli whoami` | accept the gated-model terms on HF; re-login |
| OOM mid-Adv-P run | `nvidia-smi` while it's running | drop `--beam-size` from 4 to 2; or upgrade to A6000 |
| TempParaphraser upstream repo not found | `ls scripts/_workdirs/tempparaphraser/` | re-run; the driver clones on first call |
| Driver complains it can't import Adv-P's class | inspect `scripts/_workdirs/adversarial_paraphrasing/` and grep for the right class name | update the `candidates` list in `scripts/humanize_arxiv_adversarial.py` |
| baseline detector exits with `ImportError` | `python -m test.compare_baselines --list` | install the listed package; the sweep continues regardless |
| `tmux` session vanished mid-run | `tail -200 /tmp/<job>.log` for stack trace | resume with the same `tmux new-session …` command |
