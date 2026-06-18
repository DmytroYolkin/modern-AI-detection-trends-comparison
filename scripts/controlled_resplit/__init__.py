"""Multi-seed controlled (in-distribution) USE re-split experiment (protocol A).

Stage 1: ``resplit_seed``  -- per-seed author-disjoint 70/15/15 re-split.
Stage 2: ``run_seed``      -- retrain all 14 models per seed + held-out TEST eval.
Analysis: ``models/controlled_resplit_results.ipynb``.
"""
