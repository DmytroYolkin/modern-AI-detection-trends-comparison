#!/usr/bin/env bash
# Pod-2 v2: lean eval-pod pipeline for the 4 paper-grade baseline detectors.
#
# What this is for:
#   Pod-2 v1 (scripts/pod2_full_pipeline.sh) ran humanizers + 6 baselines and
#   crashed on disk-fill + missing-dep issues. The humanizers + R-Detect +
#   RAIDAR are now handled separately (humanizers done locally / on pod 3;
#   R-Detect and RAIDAR overnight on the user's laptop). What's left is just
#   4 paper-grade baselines that genuinely need 32 GB+ VRAM.
#
# What it runs:
#   1. fast_detect_gpt  (GPT-Neo-2.7B + GPT-J-6B)
#   2. binoculars       (Falcon-7B + Falcon-7B-instruct, 4-bit)
#   3. detect_gpt       (t5-large + GPT-Neo-2.7B, 10 perturbations)  <-- N=10 to fit budget
#   4. radar            (TrustSafeAI/RADAR-Vicuna-7B)
#
#   Each on:
#     - data/testing_dataset/arxiv_final/arxiv_merged.jsonl        (clean, 2574 rows)
#     - data/testing_dataset/arxiv_final/arxiv_humanized_ai_only.jsonl (humanized AI-only, 2574 rows)
#   Total ~5150 records per detector; ~8 h wall-clock on 5090, ~$6.50.
#
# Pod-2 v1 lessons baked in:
#   - HF_HUB_CACHE redirected to /workspace (avoids /root disk-fill)
#   - --skip-if-exists so a re-launch resumes per-detector
#   - Pipeline exits non-zero if any detector JSON contains "error"
#     (the "ALL DONE" message in v1 lied because it never checked).
#
# Launch from its own tmux session so it survives SSH drops:
#
#   tmux new-session -d -s eval_v2 \
#       "bash scripts/pod_eval_v2.sh 2>&1 | tee /tmp/eval_v2.log"
#
#   tmux attach -t eval_v2   # Ctrl-b d to detach

set -u  # error on unset vars (no `set -e`: per-detector failures are caught downstream)

PROJ="/workspace/modern-AI-detection-trends-comparison"
cd "$PROJ" || { echo "[FATAL] cannot cd to $PROJ"; exit 1; }

# Redirect HuggingFace cache to /workspace (60 GB volume).
# /root is typically 50 GB on RunPod templates and Falcon-7B-pair + GPT-J-6B
# + RADAR-Vicuna-7B together easily fill it.
export HF_HOME=/workspace/.hf_cache
export HF_HUB_CACHE=/workspace/.hf_cache/hub
mkdir -p "$HF_HUB_CACHE"

ARXIV_DIR="data/testing_dataset/arxiv_final"
OUT_CLEAN="models/baseline_results/arxiv_clean"
OUT_HUMAN="models/baseline_results/arxiv_humanized"
CFG="scripts/baselines_paper_faithful_lean.json"
DETECTORS="fast_detect_gpt,binoculars,detect_gpt,radar"

mkdir -p "$OUT_CLEAN" "$OUT_HUMAN"

stage() {
    echo
    echo "============================================================"
    echo "[$(date '+%F %T')] $1"
    echo "============================================================"
}

stage "Stage 1/2 -- baselines on clean arxiv (n=2574)"
python -m test.compare_baselines \
    --detectors "$DETECTORS" \
    --input-jsonl "$ARXIV_DIR/arxiv_merged.jsonl" \
    --output "$OUT_CLEAN" \
    --detector-config "$CFG" \
    --skip-if-exists \
    || echo "[WARN] stage 1 non-zero exit; per-detector failures logged into JSONs"

stage "Stage 2/2 -- baselines on humanized AI-only (n=2574; humans reused from stage 1)"
python -m test.compare_baselines \
    --detectors "$DETECTORS" \
    --input-jsonl "$ARXIV_DIR/arxiv_humanized_ai_only.jsonl" \
    --output "$OUT_HUMAN" \
    --detector-config "$CFG" \
    --skip-if-exists \
    || echo "[WARN] stage 2 non-zero exit; per-detector failures logged into JSONs"

# Honest exit-code: scan every output JSON; if any has an "error" field, fail.
stage "Post-flight check -- looking for per-detector errors"
fail=0
for d in $(echo "$DETECTORS" | tr ',' ' '); do
    for outdir in "$OUT_CLEAN" "$OUT_HUMAN"; do
        # Match either the stem-prefixed names (e.g. arxiv_merged__detect_gpt.metrics.json)
        # or the plain detector name (e.g. detect_gpt.metrics.json).
        for f in "$outdir"/*"${d}".metrics.json; do
            [ -f "$f" ] || continue
            if python -c "import json,sys; print(json.load(open(sys.argv[1])).get('error'))" "$f" 2>/dev/null | grep -qv '^None$'; then
                err=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('error'))" "$f")
                echo "[FAIL] $f -> $err"
                fail=1
            fi
        done
    done
done

stage "ALL DONE (real this time -- exit code reflects detector failures)"
if [ "$fail" -ne 0 ]; then
    echo "Some detectors errored. Inspect /tmp/eval_v2.log and the per-detector JSONs."
    exit 3
fi

echo
echo "Artifacts to send back to local (via runpodctl, with HOME=/workspace):"
echo "  $OUT_CLEAN/"
echo "  $OUT_HUMAN/"
echo
echo "On the pod (tar each dir before sending so runpodctl gets a single file):"
echo "  tar -czf /workspace/bl_arxiv_clean_v2.tar.gz $OUT_CLEAN/"
echo "  tar -czf /workspace/bl_arxiv_humanized_v2.tar.gz $OUT_HUMAN/"
echo "  HOME=/workspace runpodctl send /workspace/bl_arxiv_clean_v2.tar.gz"
echo "  HOME=/workspace runpodctl send /workspace/bl_arxiv_humanized_v2.tar.gz"
