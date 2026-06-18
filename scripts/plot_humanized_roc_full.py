"""Recreate test/results/arxiv_eval/roc_curves/humanized.png but INCLUDING the
three detectors (binoculars, r_detect, fast_detect_gpt) that the standard run
drops because their `ai_only__*` and `eval_with_humanizers__*` JSONs collide on
the same detector name (the single-class ai_only file wins and gets skipped).

Reuses evaluate_arxiv's own in-house scoring + `_plot_roc_panel` so the in-house
curves and the plot style are byte-for-byte the same as the original; only the
three baselines are overridden from their full-set (3861-row) JSONs.

Writes a NEW file -- the original humanized.png is left untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from test import evaluate_arxiv as E
import training.paths as paths

ROOT = Path(__file__).resolve().parents[1]
FEAT_CLEAN = ROOT / "data" / "features" / "arxiv.npz"
FEAT_HUM = ROOT / "data" / "features" / "arxiv_humanized.npz"
REC_HUM = ROOT / "data" / "testing_dataset" / "arxiv_final" / "arxiv_eval_with_humanizers.jsonl"
BL_CLEAN = ROOT / "models" / "baseline_results" / "arxiv_clean"
BL_HUM = ROOT / "models" / "baseline_results" / "arxiv_humanized"
MODELS = ROOT / "models" / "ready_models"
OUT = ROOT / "test" / "results" / "arxiv_eval" / "roc_curves" / "humanized_with_baselines.png"

# detectors whose full-set (human + humanized-AI) scores live in the
# eval_with_humanizers JSONs but are dropped by the standard pipeline
FULL_BASELINES = {
    "baseline_binoculars": "arxiv_eval_with_humanizers__binoculars.metrics.json",
    "baseline_r_detect": "arxiv_eval_with_humanizers__r_detect.metrics.json",
    "baseline_fast_detect_gpt": "arxiv_eval_with_humanizers__fast_detect_gpt.metrics.json",
}


def main() -> int:
    device = paths.resolve_device("auto")
    records_humanized = E._load_records(REC_HUM)

    # replicate main()'s humanized baseline merge (ai_only + clean humans)
    humanized_raw = E._load_baseline_dir(BL_HUM)
    clean_raw = E._load_baseline_dir(BL_CLEAN)
    override = None
    if humanized_raw and clean_raw:
        override = E._merge_humanized_with_clean_humans(
            humanized_raw, clean_raw, records_humanized)

    # in-house models scored on the humanized cache -- identical to the run that
    # produced the original humanized.png
    _hm_metrics, hm_scores = E._collect_results(
        FEAT_HUM, REC_HUM, BL_HUM, MODELS, device, "humanized",
        baselines_override=override,
    )

    # override the three with their full-set scores
    for name, fname in FULL_BASELINES.items():
        d = json.loads((BL_HUM / fname).read_text())
        hm_scores[name] = (np.asarray(d["y_true"], dtype=int),
                           np.asarray(d["y_scores"], dtype=float))
        print(f"  added {name}: n={len(d['y_true'])}")

    det_list = [(n, y, s) for n, (y, s) in hm_scores.items()]
    E._plot_roc_panel(det_list, "ROC -- humanized (all detectors)", OUT)
    print(f"wrote {OUT} with {len(det_list)} detectors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
