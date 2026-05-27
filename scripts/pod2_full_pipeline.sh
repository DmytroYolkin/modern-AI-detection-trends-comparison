#!/usr/bin/env bash
# Pod #2 full pipeline -- humanize arxiv AI texts, then run paper-faithful
# baselines on both the clean and the humanized eval sets.
#
# This script is the single fire-and-forget command for pod #2. It is
# **fully resumable**:
#   - Both humanizers skip records whose source_text_id is already in their
#     output JSONL (see scripts/_humanizer_common.py).
#   - The merge step is idempotent.
#   - compare_baselines.py is launched with --skip-if-exists, so any
#     detector whose successful output JSON already exists is skipped on
#     re-run.
#
# Launch this from inside its own tmux session so it survives SSH drops:
#
#   tmux new-session -d -s pipeline \
#       "bash scripts/pod2_full_pipeline.sh 2>&1 | tee /tmp/pipeline.log"
#
# Re-running the same tmux command after an interruption picks up where
# the script stopped without redoing finished work.

set -u   # error on unset vars
# NOTE: NOT using `set -e` -- we want a single detector's OOM/error to be
# logged and the rest of the pipeline to continue.

PROJ="/workspace/modern-AI-detection-trends-comparison"
cd "$PROJ" || { echo "[FATAL] cannot cd to $PROJ"; exit 1; }

ARXIV_DIR="data/testing_dataset/arxiv_final"
OUT_CLEAN="models/baseline_results/arxiv_clean"
OUT_HUMAN="models/baseline_results/arxiv_humanized"
CFG="scripts/baselines_paper_faithful.json"

mkdir -p "$OUT_CLEAN" "$OUT_HUMAN"

stage() {
    echo
    echo "============================================================"
    echo "[$(date '+%F %T')] $1"
    echo "============================================================"
}

# --- Stage 1: Adversarial Paraphrasing humanizer (resumable) ---------------
stage "Stage 1/5 -- Adversarial Paraphrasing humanizer"
python scripts/humanize_arxiv_adversarial.py --device cuda \
    || echo "[WARN] stage 1 returned non-zero; continuing with whatever it wrote"

# --- Stage 2: TempParaphraser humanizer (resumable) ------------------------
stage "Stage 2/5 -- TempParaphraser humanizer"
python scripts/humanize_arxiv_temppara.py --device cuda \
    || echo "[WARN] stage 2 returned non-zero; continuing with whatever it wrote"

# --- Stage 3: Merge into eval sets (idempotent) ----------------------------
stage "Stage 3/5 -- Merge humanized + humans into eval sets"
python scripts/merge_humanized_evalset.py \
    || { echo "[FATAL] merge failed; cannot proceed to baselines"; exit 2; }

# --- Stage 4: Baselines on clean arxiv (resumable per-detector) ------------
stage "Stage 4/5 -- Baselines on clean arxiv (2574 rows, humans + Claude rewrites)"
python -m test.compare_baselines \
    --detectors all \
    --input-jsonl "$ARXIV_DIR/arxiv_merged.jsonl" \
    --output "$OUT_CLEAN" \
    --detector-config "$CFG" \
    --skip-if-exists \
    || echo "[WARN] stage 4 returned non-zero; per-detector failures are recorded in JSONs"

# --- Stage 5: Baselines on humanized AI-only (resumable per-detector) ------
# Note: humans are NOT re-scored here -- the 1287 human scores from stage 4
# are reused by test/evaluate_arxiv.py when the user runs the local report.
stage "Stage 5/5 -- Baselines on humanized AI-only (2574 rows, Adv-P + TempPara)"
python -m test.compare_baselines \
    --detectors all \
    --input-jsonl "$ARXIV_DIR/arxiv_humanized_ai_only.jsonl" \
    --output "$OUT_HUMAN" \
    --detector-config "$CFG" \
    --skip-if-exists \
    || echo "[WARN] stage 5 returned non-zero; per-detector failures are recorded in JSONs"

stage "ALL DONE"
echo "Artifacts to download back to local:"
echo "  $ARXIV_DIR/arxiv_humanized_adv.jsonl"
echo "  $ARXIV_DIR/arxiv_humanized_temp.jsonl"
echo "  $ARXIV_DIR/arxiv_eval_with_humanizers.jsonl"
echo "  $ARXIV_DIR/arxiv_humanized_ai_only.jsonl"
echo "  $OUT_CLEAN/  (tar.gz then runpodctl send)"
echo "  $OUT_HUMAN/  (tar.gz then runpodctl send)"
