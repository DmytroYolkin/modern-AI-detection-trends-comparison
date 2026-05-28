# RunPod feature extraction for the humanized arxiv eval set

This runs the full NELA + StyleDecipher (Ollama, 5 LLMs) + TRACE feature
extraction on **`data/testing_dataset/arxiv_final/arxiv_eval_with_humanizers.jsonl`**
(3 861 records = 1 287 humans + 1 287 Adversarial-Paraphrasing rows + 1 287
TempParaphraser rows). The output is `data/features/arxiv_humanized.npz`, which
the evaluation harness will consume alongside the existing `arxiv.npz` (clean
arxiv) and the in-domain `test.npz`.

The split is wired into `training/build_dataset.py` as `--splits arxiv_humanized`
via the `OOD_SPLITS` registry; it is **not** part of `ALL_SPLITS` and therefore
never appears in any training flow.

**Pod context.** This runs on a **fresh, clean pod** — no state from the prior
extraction or humanizer pods. Set up from zero following the steps below.

**Speed expectation.** With the parallel-rewrites change in
`extractors/styledecipher_extractor.py` (5 Ollama calls per record now issued
concurrently across one thread per model) and an Ollama server configured to
serve all 5 models simultaneously (`OLLAMA_NUM_PARALLEL=5`,
`OLLAMA_MAX_LOADED_MODELS=5`), per-record wall-clock drops from ~30–60 s
(sequential) to ~8–12 s (parallel — bounded by the slowest of the 5 models).
On a 5090 (32 GB, fits all 5 models in VRAM with no swap) the full 3 861-row
extraction lands at **~9–13 h** — comfortably under the 10 h target on a hot
cache after warmup. On a 4090 (24 GB) only 2 models stay resident, so the
parallel speedup degrades to ~2× and you're back to ~16 h. **Use a 5090 or
larger.**

This is a near-clone of [RUNPOD_arxiv_extraction.md](RUNPOD_arxiv_extraction.md)
— follow that runbook for the *general* shape (clone, install, tmux), but
**apply the §1 tweaks below instead** of the corresponding sections in the
older doc.

---

## 1. Setup — follow the arxiv runbook, with two tweaks

Do every step in §1–§2 of [RUNPOD_arxiv_extraction.md](RUNPOD_arxiv_extraction.md)
(pod template, deps, Ollama install, model pulls, tmux sessions) but apply the
two additions below in the relevant places. They exist because pod 2 hit
"No space left on device" on the small 50 GB root filesystem after the
combined HF + Ollama caches filled `/root` — both default there.

### 1a. Redirect the HF Hub cache to `/workspace` (the 60 GB volume)

Add this to `~/.bashrc` (and `source ~/.bashrc`) **before** any
`pip install`, model download, or extractor run:

```bash
echo 'export HF_HOME=/workspace/.hf_cache' >> ~/.bashrc
echo 'export HF_HUB_CACHE=/workspace/.hf_cache/hub' >> ~/.bashrc
source ~/.bashrc
mkdir -p "$HF_HUB_CACHE"
```

The SBERT (`all-mpnet-base-v2`, ~400 MB) and Wegmann encoder
(`AnnaWegmann/Style-Embedding`, ~500 MB) downloads land here on first
extraction. Modest in absolute terms, but `/root` is small enough that
combined with the Ollama blobs (§1b) it crashes.

### 1b. Redirect Ollama's model blobs to `/workspace` + enable parallel serving

The five StyleDecipher models (`llama3, mistral, gemma, phi3, qwen2`) total
~25 GB of blobs and default to `/root/.ollama`. Move them to `/workspace`
by setting `OLLAMA_MODELS`. The other two flags are what realises the
in-extractor parallelism: `NUM_PARALLEL=5` lets Ollama serve 5 concurrent
requests, `MAX_LOADED_MODELS=5` keeps all 5 model weights GPU-resident so
none of those concurrent requests pay a load-time penalty.

```bash
mkdir -p /workspace/.ollama
tmux new-session -d -s ollama \
    "OLLAMA_HOST=0.0.0.0:11434 \
     OLLAMA_KEEP_ALIVE=24h \
     OLLAMA_NUM_PARALLEL=5 \
     OLLAMA_MAX_LOADED_MODELS=5 \
     OLLAMA_MODELS=/workspace/.ollama \
     ollama serve 2>&1 | tee /tmp/ollama.log"

sleep 5
curl -s http://127.0.0.1:11434/api/tags    # smoke-test: should return JSON
```

Then `ollama pull` the five models — they land under `/workspace/.ollama/`
instead of `/root/.ollama/`.

(If you already pulled them to `/root` on this pod before reading this,
either move them with `mv /root/.ollama /workspace/.ollama` and re-launch
the server, or just live with the disk usage and free space elsewhere.)

**Parallel-rewrites sanity check.** After the Ollama server is up and all 5
models pulled, smoke-test the parallel pathway in
`extractors/styledecipher_extractor.py` end-to-end:

```bash
python - <<'PY'
import time
from extractors.styledecipher_extractor import generate_rewrites_multi_llm
t = time.time()
out = generate_rewrites_multi_llm(
    "The transformer architecture revolutionised natural language processing "
    "by enabling parallel computation across input tokens."
)
elapsed = time.time() - t
print(f"got {len(out)} unique rewrites in {elapsed:.1f}s")
# Expected:
#   parallel (this pod, correctly configured): ~6-12s
#   serial   (old behaviour or NUM_PARALLEL=1): ~20-40s
PY
```

If the elapsed time is > 20 s, `OLLAMA_NUM_PARALLEL` is wrong or the GPU
isn't big enough to keep all 5 models resident — fix before launching the
full extraction (you'll burn ~$25 of pod time learning the same lesson
the hard way otherwise).

---

## 2. The single extraction command (in tmux)

```bash
tmux new-session -d -s extract \
    "cd /workspace/modern-AI-detection-trends-comparison && \
     python -m training.build_dataset \
        --splits arxiv_humanized \
        --styledecipher ollama \
        --trace-context author \
        --device cuda \
        --checkpoint-every 25 \
        2>&1 | tee /tmp/extract.log"
```

Attach to watch progress:

```bash
tmux attach -t extract       # Ctrl-b d to detach
```

What this does (differences from the clean-arxiv extraction):

- Loads only `data/testing_dataset/arxiv_final/arxiv_eval_with_humanizers.jsonl`
  (3 861 rows: 1 287 humans + 2 574 humanized AI), no train/val/test.
- Writes incrementally to `data/features/arxiv_humanized.npz`; checkpoints
  atomically every 25 records.
- TRACE uses same-author siblings *within this jsonl* — the 1 287 humans
  carry the same `author_id`s as in the clean arxiv set (99 authors), and
  the 2 574 humanized rewrites inherit `author_id` from their source human.
- StyleDecipher generates 5 fresh rewrites per record via Ollama.

**Resumable**: same as the clean run — re-launch the exact same `tmux
new-session` block to continue from the last checkpoint. Already-cached
records are skipped.

### 2.1 Optional: redundant-human reuse (manual, ~30 min saved)

The 1 287 human rows in `arxiv_eval_with_humanizers.jsonl` are
**byte-identical** to the 1 287 humans in `arxiv_merged.jsonl` (the merge
script just concatenates through). If you want to skip re-extracting them,
do that as a post-processing step on your **local** machine after pulling
both `arxiv.npz` and `arxiv_humanized.npz` back — it's not worth the
extra wiring complexity on the pod.

The evaluation harness can also align the two caches by `id` at read time;
the redundant features won't hurt anything, they just take ~30 min of GPU
wall-clock extra. Not worth doing the surgery on the pod.

### 2.2 Monitor without attaching

```bash
tail -f /tmp/extract.log
grep -c '^  \[arxiv_humanized\]' /tmp/extract.log   # rough milestone count

python - <<'PY'
import numpy as np
d = np.load('data/features/arxiv_humanized.npz', allow_pickle=True)
print('cached so far:', d['ids'].shape[0], '/ 3861')
PY
```

### 2.3 When the job finishes

```bash
tmux ls                                 # should no longer list "extract"
tail -20 /tmp/extract.log               # last line: "Wrote feature caches ..."
python - <<'PY'
import numpy as np
d = np.load('data/features/arxiv_humanized.npz', allow_pickle=True)
print('final rows :', d['ids'].shape[0])           # expect 3861
print('shapes    :', d['nela'].shape, d['style'].shape, d['trace'].shape)
print('style cov :', d['style_ok'].mean())         # expect ~1.0
import collections
print('per-source:', dict(collections.Counter(d['sources'].tolist())))
# expect: arxiv (~1287), arxiv_humanized_adv (~1287), arxiv_humanized_temp (~1287)
PY
```

Then stop the ollama tmux session to free the GPU before `runpodctl send`:

```bash
tmux kill-session -t ollama
```

---

## 3. Expected duration

The previous (serial) baseline on pod 1 was ~30 s/rec dominated by the 5
Ollama generations. With the parallel-rewrites change in
[`extractors/styledecipher_extractor.py`](../extractors/styledecipher_extractor.py)
(one thread per LLM, all 5 issued concurrently) and an Ollama server
configured per §1b, the per-record cost drops to roughly **8–12 s** — bounded
by the slowest of the 5 models per record rather than by their sum.

| GPU | All 5 models resident? | Per-rec | Total (3 861 records) | Cost (community) |
|---|---|---|---|---|
| **5090 (32 GB)** | yes | ~8–12 s | **~9–13 h** | ~$8–10 (5090 @ ~$0.79/h) |
| 4090 (24 GB) | only 2 resident → swap | ~15–20 s | ~16–21 h | ~$13–17 |
| A6000 (48 GB) | yes (room to spare) | ~8–12 s | ~9–13 h | ~$5–7 (A6000 @ ~$0.49/h) |

**Use a 5090 (recommended) or A6000.** The 4090 is technically usable but
sacrifices most of the parallel-rewrites speedup and the cost saving
disappears in extra wall-clock.

If the per-record cost on your run is significantly higher than the table
suggests, the most likely cause is `OLLAMA_NUM_PARALLEL` not being set to 5,
or the GPU running out of VRAM and Ollama swapping model weights between
requests. Both are visible in the Ollama tmux log (`tmux attach -t ollama`,
look for repeated "loading model" lines).

---

## 4. Download the result back

A single small (~2 MB compressed) file:

```bash
# on the pod
HOME=/workspace runpodctl send data/features/arxiv_humanized.npz
```

(The `HOME=/workspace` is a pod-2 lesson — `runpodctl` insists on writing
its config to `$HOME/.runpod/` and `/root/` is the small partition that
fills first.)

Then on your local machine, from the repo root:

```bash
./runpodctl.exe receive <one-shot-code>
mv arxiv_humanized.npz data/features/        # if it didn't land there
```

The evaluation harness expects it at `data/features/arxiv_humanized.npz`.

---

## 5. What's next (after this file is on local)

Both the clean and humanized arxiv feature caches are then on local:

```
data/features/arxiv.npz           # 2 574 rows  (pod 1)
data/features/arxiv_humanized.npz # 3 861 rows  (this pod)
```

The evaluation pod runs:

1. The two in-house combination models (best neural + best classical) on
   both caches.
2. Three newly-trained single-modality classifiers (NELA-only,
   StyleDecipher-only, TRACE-only) on both caches.
3. The six baseline detectors on both raw text jsonl inputs
   (`arxiv_merged.jsonl` and `arxiv_humanized_ai_only.jsonl`), with the
   pod-2 fixes baked in (HF cache to `/workspace`, `pip install ollama`,
   vendored R-Detect cloned).

See the upcoming evaluation runbook for that. The single-modality
classifiers will be trained from the local `train.npz` cache (no GPU
needed) and committed before the eval pod boots, so the eval pod only
loads checkpoints — no training time on the rented GPU.

---

## 6. Common pitfalls

Same set as the clean extraction; the two big new ones are covered in §1
(HF cache + Ollama model path). One extra worth flagging:

| Pitfall | Quick check | Fix |
|---|---|---|
| Wrong row count at the end (≠ 3 861) | `python -c "import numpy as np; print(np.load('data/features/arxiv_humanized.npz', allow_pickle=True)['ids'].shape[0])"` | inspect `/tmp/extract.log` for skipped records; re-launch the §2 tmux command to resume |
| Per-source breakdown is missing one humanizer (only 2 of the 3 expected) | the §2.3 `collections.Counter` block | the merge step on pod 2 may have run with `--allow-missing-humanizer`; check `wc -l` on the two humanizer jsonls — both should be 1287 |
| `--splits arxiv_humanized` rejected with "unknown split" | `python -c "from training.build_dataset import OOD_SPLITS; print(list(OOD_SPLITS))"` | you're on an old commit of `build_dataset.py` — pull the latest from `main` |
