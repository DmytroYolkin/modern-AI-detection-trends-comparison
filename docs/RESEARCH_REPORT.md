# A Research Report on Multimodal Human-vs-AI Text Detection: Fusion, Modality Dropout, and Out-of-Distribution Robustness

*Author: Internal research note. Status: interim — paper-grade pod-2-v2 baseline numbers pending.*

## Abstract

This report documents an empirical study of human-vs-AI text detection that combines three orthogonal feature modalities — NELA surface-linguistic features ([Horne, Khedr, Adali, 2018](https://arxiv.org/abs/1803.10124)), StyleDecipher style-distance features in the style of [Patel et al. (2024)](https://arxiv.org/abs/2410.12757), and TRACE author-style embeddings ([Liang et al., 2024](https://arxiv.org/abs/2402.16638)) — into a 225-dimensional representation that is consumed by four neural fusion variants and seven classical classifiers. Six contemporary AI-text detectors are wrapped as baselines. The central observations are: (i) on the in-distribution USE-essay slice, every model saturates at macro-F1 ≥ 0.987 because seventeen NELA features each separate the classes by more than one pooled standard deviation; (ii) the gated fusion variant is the only architecture that survives a deliberate double distribution shift (essay → academic abstract; open-source 7B → Claude-haiku) with usable ranking power (ROC-AUC = 0.81); (iii) under a strict false-positive-rate ≤ 1 % operating regime, no system in the present comparison is deployment-grade out of distribution, with the best true-positive rate dropping to 0.089; (iv) two well-known detectors (Binoculars with a GPT-2 observer and our `classical_svm` baseline) exhibit a sign-flip on academic prose; and (v) the principal failure mode of TRACE on out-of-distribution academic text is the Reddit-trained Wegmann encoder, not author-context coverage. We organise the report into four sections: interesting observations, method descriptions, results, and an integrated discussion of methodology and limitations.

---

## Section 1 — Interesting observations and proofs

Each finding below is anchored to a specific cell, file, or table in the repository. Numbers are quoted as observed; no smoothing or rounding beyond what the source artefact already applies.

### 1.1 NELA dominance on the in-distribution slice

The in-distribution training matrix `data/features/train.npz` (n = 4 051; 703 human / 3 348 AI) was profiled with per-feature Cohen's d on each of the three modality blocks, as recorded in [`docs/NELA_DOMINANCE_ANALYSIS.md`](NELA_DOMINANCE_ANALYSIS.md) §5. The summary across modalities is reproduced from §5.3:

| Modality | dims | max \|d\| | median \|d\| | dims with \|d\| > 1.0 | dims with \|d\| > 0.5 |
|---|---:|---:|---:|---:|---:|
| NELA  |  87 | **3.39** | 0.48 | **17** | **42** |
| Style |  10 | 1.16 | 0.21 |  1 |  2 |
| TRACE | 128 | 0.86 | 0.33 |  0 | 29 |

The top NELA dimension is `ttr` (type-token ratio) at |d| = 3.39 between the human and AI classes ([NELA_DOMINANCE_ANALYSIS.md §5.1, rank 1](NELA_DOMINANCE_ANALYSIS.md)), followed by `word_count` at 2.94 and `stops` (stopword rate) at 2.62. The single best Style feature is `cos_emb_mean` at 1.16; the single best TRACE feature is `trace_023` at 0.86. **Not one of TRACE's 128 dimensions clears |d| = 1.0** in this slice.

This explains the saturation phenomenon visible in [`models/ready_models/pipeline_summary.json`](../models/ready_models/pipeline_summary.json), where every one of eleven trained models achieves macro-F1@0.5 ≥ 0.987 on the held-out test split. Even a linear classifier provided only with `ttr` and `word_count` would draw a near-perfect boundary; the seventeen >1-σ-separating axes simultaneously available make the classification problem trivial in the population from which the cache was constructed. The corollary, often missed in detector papers, is that any single-extractor in-distribution number must be read as describing the *dataset*, not the detector.

### 1.2 Why `fusion_gating` rather than the three sibling fusion variants

[`fusion/combination_all.py`](../fusion/combination_all.py) implements four fusion strategies on the three projected modality vectors (n, s, t) ∈ ℝ²⁵⁶:

| Strategy | Operation (after per-modality `Linear→ReLU→LayerNorm` projection) |
|---|---|
| `concat`    | concatenate `[n; s; t]` ∈ ℝ⁷⁶⁸ → `Linear → ReLU → LayerNorm` ([combination_all.py:21-27](../fusion/combination_all.py#L21-L27)) |
| `mlp`       | identical to `concat` but the post-projection is an explicit MLP block ([combination_all.py:29-35](../fusion/combination_all.py#L29-L35)) |
| `attention` | 4-head self-attention over the 3-vector sequence; mean-pool the output ([combination_all.py:37-39](../fusion/combination_all.py#L37-L39)) |
| `gating`    | `g = σ(Linear([n; s; t]))`, output `g · n + (1 − g) · (s + t) / 2` ([combination_all.py:41-43, 61-64](../fusion/combination_all.py#L41-L64)) |

In-distribution test-set numbers from [`pipeline_summary.json`](../models/ready_models/pipeline_summary.json) collapse the four variants into a 0.1 pp ROC-AUC band (`neural_concat` and `neural_mlp` tie at 0.99998, `neural_gating` 0.99987, `neural_attention` 0.99948). At face value the gating variant is no better than its siblings, and is the weakest neural classifier at the strict-FPR operating point (TPR@strict = 0.9898 versus 0.9942 for `concat`/`mlp`).

The picture inverts under distribution shift. On the arxiv OOD cache (n = 2 574; 1 287 human abstracts and 1 287 Claude-haiku rewrites), [`models/interim_results.ipynb`](../models/interim_results.ipynb) §B reports the in-house default-threshold AUCs (sorted): `fusion_gating` 0.8063, `rf_nela_only` 0.7222, `classical_svm` 0.5710, `rf_style_only` 0.5546, `rf_trace_only` 0.5000. The three non-gating fusion variants are not in §B but the cross-cutting commentary in §B.takeaway records their behaviour at the same operating point as approximately 0.50–0.51 AUC, consistent with the modality-collapse hypothesis: the symmetric variants memorise the in-distribution NELA solution and have no learned mechanism for re-weighting modalities away from it.

The gating layer's parameterisation is exactly the mechanism that allows for asymmetric, instance-conditioned re-weighting: at inference, each sample's `g` vector decides how much of the NELA-dominated projection to lean on versus the Style + TRACE residual. When the test-time NELA distribution drifts, the gate can in principle attenuate it; the symmetric `concat`/`mlp` heads cannot. This explains the ranking inversion between in-distribution and OOD across the four variants. The instance-conditioned gate is the same mechanism that underlies modality-attention literature in vision-language fusion; the asymmetric modality-dropout regulariser ([`training/config.py:34-36`](../training/config.py#L34-L36), rates NELA 0.5 / Style 0.2 / TRACE 0.2) following [Neverova et al., TPAMI 2016 ("ModDrop")](https://ieeexplore.ieee.org/document/7373645) is what trains the gate to actually use the residual modalities during fitting rather than collapse them to zero weight.

### 1.3 Per-extractor permutation importance versus stand-alone modality strength

Two complementary measurements of "what does each modality buy us" exist in the repository, and they should not be conflated.

The first is the **marginal contribution** view, captured by block-wise permutation importance in [`models/analysis.ipynb`](../models/analysis.ipynb) §3b and reproduced in [METHODOLOGY.md §7.2](../METHODOLOGY.md). Permutation importance measures how much test-set macro-F1 drops when one modality block's values are shuffled across samples while the others are held fixed. On the held-out test split:

| Model | NELA ΔF1 | StyleDecipher ΔF1 | TRACE ΔF1 | NELA share |
|---|---:|---:|---:|---:|
| `neural_mlp`        | +0.2813 | +0.0566 | +0.0297 | 76.5 % |
| `neural_concat`     | +0.2782 | +0.0607 | +0.0292 | 75.6 % |
| `neural_attention`  | +0.3464 | +0.0552 | +0.0200 | 82.2 % |
| `neural_gating`     | +0.4120 | +0.0403 | +0.0055 | 90.0 % |
| `classical_logreg`  | +0.4945 | +0.0097 | +0.0142 | 95.4 % |
| `classical_xgboost` | +0.4735 | +0.0030 | +0.0003 | 99.3 % |

(Source: [`NELA_DOMINANCE_ANALYSIS.md` §4](NELA_DOMINANCE_ANALYSIS.md), which executes 10 permutations per block at seed 42.)

The second is the **stand-alone** view: train one classifier per modality and ask how well that single modality can solve the task in isolation. We trained three Random Forest checkpoints to that end and the val-split results sit in `models/ready_models/`:

| Model | val accuracy | val macro-F1 | source |
|---|---:|---:|---|
| `rf_random_forest_nela`  | 0.9966 | **0.9941** | [`clf_random_forest_nela.metrics.json:5-6`](../models/ready_models/clf_random_forest_nela.metrics.json) |
| `rf_random_forest_style` | 0.8964 | 0.7913 | [`clf_random_forest_style.metrics.json:5-6`](../models/ready_models/clf_random_forest_style.metrics.json) |
| `rf_random_forest_trace` | 0.8523 | 0.5872 | [`clf_random_forest_trace.metrics.json:5-6`](../models/ready_models/clf_random_forest_trace.metrics.json) |

The two views answer different questions. Permutation importance is a *redundancy-given-others* measurement: it gives near-zero ΔF1 to a modality that is informative but whose information is already provided by another modality at the operating point being interrogated. Stand-alone measurement is an *absolute-signal* quantity: it tells us each modality, in isolation, is far from useless — Style-only reaches macro-F1 = 0.7913 and TRACE-only reaches 0.5872 (well above the no-skill baseline of ≈ 0.46 on this 1 : 4.76 imbalance). Both views matter when arguing about the design of a multimodal detector: §7.2's table is a poor argument for *dropping* Style or TRACE, because their stand-alone capacity is non-trivial. It is, however, a strong argument that NELA's information is sufficient to saturate this particular task — which is what motivates introducing distribution shift to probe each modality's robustness independently.

### 1.4 TRACE failure mode on OOD is the encoder, not the author corpus

A natural first hypothesis for TRACE's collapse to AUC = 0.5000 on the arxiv set (`rf_trace_only` in [`interim_results.ipynb`](../models/interim_results.ipynb) §B) is that the arxiv anchors lack sibling context, since TRACE's input contract is "≥ 1 usable other-text-by-same-author". This is empirically false. The arxiv evaluation corpus carries 99 distinct authors with a median of 11 papers per author; the in-distribution USE training cache has an average of 3.5 siblings per anchor ([`data/features/meta.json`](../data/features/meta.json) avg sibling stats; [`NELA_DOMINANCE_ANALYSIS.md` §3 framing](NELA_DOMINANCE_ANALYSIS.md)). The arxiv set has *more* author context per anchor, not less.

The remaining candidate explanation is the encoder itself. [METHODOLOGY.md §2.3](../METHODOLOGY.md) documents the substitution of [`AnnaWegmann/Style-Embedding`](https://huggingface.co/AnnaWegmann/Style-Embedding) — a RoBERTa-base contrastively trained on Reddit conversations ([Wegmann, Schraagen, Nguyen, RepL4NLP 2022](https://aclanthology.org/2022.repl4nlp-1.26/)) — for the TRACE paper's unreleased contrastive encoder. The Wegmann encoder learns a style-vs-content axis where the positive class is "two Reddit posts by the same user", a register in which inter-author variation is large and the lexical surface mixes register codes freely. On arxiv academic prose, every author is writing in approximately the same formal-academic register, and the relevant variation has been crushed into a narrow subspace that the encoder was not optimised to discriminate over. The consequence is that same-author and different-author distances become statistically indistinguishable on this corpus. This is a *representation generalisation* failure, not a *data coverage* failure, and it is the cleanest example in the present study of an encoder-substitution decision propagating to deployment-time behaviour. (For the in-distribution slice, the Wegmann substitution does add isolated signal: in [METHODOLOGY.md §7.3](../METHODOLOGY.md), swapping `all-mpnet-base-v2` for Wegmann lifts `classical_svm` TRACE ΔF1 from −0.0004 to +0.0215.)

### 1.5 Threshold calibration, not ranking, is the binding OOD constraint

The arxiv strict-FPR ≤ 1 % table from [`interim_results.ipynb`](../models/interim_results.ipynb) §B.2:

| Model | acc | macro-F1 | TPR | threshold | confusion `[[TN, FP], [FN, TP]]` |
|---|---:|---:|---:|---:|---|
| `fusion_gating` | 0.5400 | 0.4228 | 0.0894 | 0.9995 | `[[1275, 12], [1172, 115]]` |
| `rf_nela_only`  | 0.5377 | 0.4283 | 0.0637 | 0.9975 | `[[1255, 32], [1158, 129]]` |
| `classical_svm` | 0.5023 | 0.3465 | 0.0140 | 0.9633 | `[[1275, 12], [1269, 18]]` |
| `rf_trace_only` | 0.5000 | 0.3460 | 0.0117 | 0.9625 | `[[1268, 19], [1268, 19]]` |
| `rf_style_only` | 0.4977 | 0.3370 | 0.0047 | 0.9850 | `[[1274, 13], [1280, 7]]` |

`fusion_gating` retains a meaningful AUC of 0.81 on arxiv — the model can *rank* AI rewrites above human abstracts well — but at the val-calibrated threshold of 0.9995 it abstains on the AI side: only 115 of 1 287 Claude-haiku rewrites are caught (TPR = 8.94 %). The strict-FPR ≤ 1 % constraint pushes the threshold to the very top of the score distribution to honour the human false-positive budget, sacrificing recall in the process. This is the *correct* deployment behaviour — an unsafe-but-rank-aware detector is being asked to be safe-and-correspondingly-quiet — but it makes the OOD numbers look worse than the AUC alone suggests. The "deployment-grade tax" of operating at FPR ≤ 1 % is the difference between AUC 0.81 (rank quality) and TPR 0.09 (caught-AI fraction).

The same column in the val table ([§A.2 of `interim_results.ipynb`](../models/interim_results.ipynb)) shows TPR ≥ 0.998 in distribution, confirming that the tax is paid almost entirely at the moment of distribution shift, not at the moment of the strict threshold itself.

### 1.6 Sign-flip phenomenon on academic prose

Two distinct detectors invert their scoring direction on the arxiv set. The first is the in-house `classical_svm`: raw AUC drops from 0.99995 in-distribution ([`pipeline_summary.json:59`](../models/ready_models/pipeline_summary.json)) to 0.5710 on arxiv ([`interim_results.ipynb`](../models/interim_results.ipynb) §B.1) and to 0.3346 on the humanized arxiv ([`interim_results.ipynb`](../models/interim_results.ipynb) §F), at which point the sign-flipped AUC_eff (= 1 − 0.3346 = 0.6654) is back above chance. The second is Binoculars with the GPT-2 / GPT-2-medium observer/performer pair: raw AUC = 0.2067 on arxiv ([`interim_results.ipynb`](../models/interim_results.ipynb) §C), with sign-flipped AUC_eff = 0.7933.

The mechanism is the same in both cases. Binoculars compares the observer's perplexity against the observer-performer cross-perplexity; the score is calibrated so that higher = more AI ([Hans et al., 2024](https://arxiv.org/abs/2401.12070)). On arxiv academic prose, GPT-2 finds the *human* abstracts (researcher-written, mid-1990s-to-2010s technical English) more surprising than Claude-haiku's rewrites (which sit closer to the contemporary mid-register that dominates GPT-2's training distribution). The score direction therefore inverts versus the regime the method was calibrated on. For `classical_svm` the analogous statement is that the RBF kernel boundary fitted in-distribution maps the "obvious-rewrite" cluster to one side; on academic prose the entire feature distribution rotates within the kernel space, and the boundary that was correctly oriented in-distribution now points the wrong way. This effect is a methodological warning for any future detector deployment across domains: do not assume the score direction transfers, and report AUC_eff = max(AUC, 1 − AUC) when audit-trailing pre-paper results.

---

## Section 2 — Method descriptions with citations

### 2.1 Three feature extractors

**NELA (87-dim, [Horne, Khedr, Adali, 2018](https://arxiv.org/abs/1803.10124))** — A surface-linguistic feature toolkit that produces, per text, 49 part-of-speech and punctuation rates, 4 readability indices (Flesch-Kincaid, SMOG, Coleman-Liau, LIX), 3 lexical-diversity measures (TTR, average word length, word count), 6 hedge/factive/assertative/implicative/report-verb/bias lexicon counts, 9 VADER + SentiWords sentiment scores, 11 moral-foundation counts, and 2 named-entity counts (locations, dates). The implementation in [`extractors/nela_extractor.py`](../extractors/nela_extractor.py) wraps the `nela_features` PyPI package and exposes 100 % per-sample coverage on the cached splits. NELA was the original credibility-classification feature set for news media; its inclusion here follows the long lineage of *surface features as a hard-to-beat baseline* in stylistic-classification work, including Horne et al.'s own AAAI workshop paper.

**StyleDecipher (10-dim, building on [Patel et al., 2024](https://arxiv.org/abs/2410.12757))** — For each anchor text, we generate five paraphrases using locally hosted Ollama models (LLaMA-3, Mistral, Gemma, Phi-3, Qwen2) and compute five similarity metrics between the anchor and each rewrite: 1-, 2-, 3-gram Jaccard overlap, normalised character-level edit similarity (`1 − Levenshtein / max_len`), and embedding cosine similarity using `sentence-transformers/all-mpnet-base-v2`. The 5 × 5 metric matrix is collapsed to a 10-dimensional vector by reporting the mean and standard deviation across the five rewriters per metric. The conceptual lineage runs through StyleDistance (Patel et al., synthetic-parallel-rewrite training for content-independent style embeddings) and the contemporaneous detector-via-rewriting work (RAIDAR; [Mao et al., 2024](https://arxiv.org/abs/2401.12970)). The choice of `all-mpnet-base-v2` as the embedding backbone is deliberate — a general semantic encoder, rather than a style-trained one, so that the *style* signal arrives from the multi-rewriter variance pattern rather than from the embedding axis. Implementation: [`extractors/styledecipher_extractor.py`](../extractors/styledecipher_extractor.py).

**TRACE (128-dim, [Liang et al., 2024](https://arxiv.org/abs/2402.16638))** — Author-context-aware style fingerprint. For each anchor text, the pipeline (i) extracts the top-5 most TF-IDF-salient sentences from the anchor and its author-context texts; (ii) encodes them through `AnnaWegmann/Style-Embedding` ([Wegmann et al., RepL4NLP 2022](https://aclanthology.org/2022.repl4nlp-1.26/)), a RoBERTa-base contrastively fine-tuned on same-Reddit-user pairs for content-independent style similarity; (iii) mean-pools token embeddings under the attention mask (768-dim); (iv) applies a fixed-seed `Linear → ReLU → Linear` projection to 128-dim (Johnson-Lindenstrauss style dimensionality reduction); and (v) mean-pools the per-sentence 128-vectors into a single anchor embedding. The author context for an anchor is defined as "all other human texts by the same author, excluding the source text of an LLM rewrite", with the no-source-leakage exclusion implemented in [`training/rebuild_trace_author.py::trace_human_siblings`](../training/rebuild_trace_author.py). The substitution of Wegmann's open-source encoder for the TRACE paper's unreleased contrastive checkpoint is documented in [METHODOLOGY.md §2.3](../METHODOLOGY.md); it is the closest contrastively-trained open-source style encoder available, and avoids the methodologically-suspect alternative of leaving the encoder randomly initialised.

### 2.2 Fusion architectures

The four variants in [`fusion/combination_all.py`](../fusion/combination_all.py) all share a per-modality projection block (`Linear(d_m → 256) → ReLU → LayerNorm`, [combination_all.py:10-19](../fusion/combination_all.py#L10-L19)) before they differ:

- **`concat`** concatenates the three 256-dim vectors and applies one post-projection block. Symmetric in the three modalities, with no learned weighting.
- **`mlp`** matches `concat` structurally but replaces the post-projection with an explicit MLP block. Distinction is mostly nominal.
- **`attention`** stacks the three vectors into a 3-token sequence and applies a 4-head self-attention block. Allows pairwise modality interaction but does not provide an instance-conditioned modality gate.
- **`gating`** computes a learned gate `g = σ(Linear([n; s; t]))` and outputs `g · n + (1 − g) · (s + t) / 2`. The structure deliberately privileges NELA (the "primary" modality) while the Style and TRACE projections share the residual half. Of the four variants, this is the only one in which a single scalar (per hidden dimension, per sample) decides how much weight NELA carries — and is therefore the only one that can attenuate NELA at inference when its distribution drifts.

A 2-layer MLP head with dropout 0.3 maps the fused 256-vector to a two-class logit; cross-entropy is class-weighted by inverse frequency ([`training/config.py:47`](../training/config.py#L47)). The classifier wrapper, including all four variants, is in [`training/model.py::FusionClassifier`](../training/model.py).

**Modality-dropout regulariser ([Neverova, Wolf, Taylor, Nebout, TPAMI 2016, "ModDrop"](https://ieeexplore.ieee.org/document/7373645))** — At training time only, each of the three modality blocks is independently zeroed with per-block probability (NELA 0.5, Style 0.2, TRACE 0.2; defaults in [`training/config.py:34-36`](../training/config.py#L34-L36)). Kept activations are rescaled by `1 / (1 − p)` so the expected magnitude at evaluation is unchanged. The asymmetric rates are a deliberate design choice: NELA's per-sample contribution is so large that symmetric dropout rates do not break the NELA-collapse the §1.1 Cohen's d analysis predicts, while a NELA drop rate of 0.5 forces the fusion head to also extract usable signal from Style and TRACE roughly half the time. The Neverova et al. paper introduces the technique in a gesture-recognition setting; the principle of per-block Bernoulli ablation transfers to text-detection unchanged.

### 2.3 Classical baselines

The 225-dimensional flat concatenation of `[nela | style | trace]` is fed to seven classical backends in [`training/classical.py`](../training/classical.py): `xgboost`, `random_forest`, `logreg` (L2-regularised, C = 1), `svm` (RBF kernel), `mlp` (sklearn `MLPClassifier(256, 128)`), `hist_gbm` (sklearn `HistGradientBoostingClassifier`), and `gradient_boosting` (sklearn plain). All use inverse-frequency `class_weight` where supported and the same train-fit feature normaliser the neural fusion track uses. We retain this many backends for two reasons. First, classical models give per-feature importances (tree-internal or coefficient magnitudes) that complement the model-agnostic permutation importance and are the source of the per-classical extractor-importance fractions referenced in [METHODOLOGY.md §4 and `clf_*.metrics.json:19-22`](../models/ready_models/). Second, they form a "no fusion architecture" reference: if a plain SVM on the flat 225-dim vector matches the four neural fusion variants on the in-distribution test split (it does — [`pipeline_summary.json`](../models/ready_models/pipeline_summary.json), `classical_svm` ROC-AUC = 0.99995 versus `neural_concat`/`neural_mlp` at 0.99998), then the in-distribution argument for the fusion architecture is exclusively about the modality-dropout regulariser and not about the head's expressive power. Random Forest is also the NELA paper's own choice for the credibility classification baseline, providing direct continuity with [Horne et al., 2018](https://arxiv.org/abs/1803.10124).

### 2.4 Single-modality classifiers

We further trained three Random Forest checkpoints — `clf_random_forest_nela`, `clf_random_forest_style`, `clf_random_forest_trace` — each restricted to one modality block by `select_blocks` in [`training/classical.py`](../training/classical.py). Their methodological purpose is the per-modality absolute-strength measurement of §1.3: stand-alone macro-F1 of (0.9941, 0.7913, 0.5872) respectively on val ([metrics JSONs in `models/ready_models/`](../models/ready_models/)). This view answers a question the permutation-importance table cannot: in the absence of the other two modalities entirely, how much of the labelled task does each modality solve? The answer informs which modalities are *necessary* (NELA, by far) and which are *retained for robustness* (Style, TRACE, which carry weak but non-trivial in-distribution signal and become differentially valuable under modality shift in §3).

### 2.5 Baseline AI-text detectors

Six contemporary detectors are wrapped in [`test/baselines/`](../test/baselines/). For each we describe the method, the wrapper's configurable knobs, and which configuration the present results were computed under. Two reference detector-configuration JSONs are shipped in the repo: [`scripts/baselines_paper_faithful.json`](../scripts/baselines_paper_faithful.json) and [`scripts/baselines_paper_faithful_lean.json`](../scripts/baselines_paper_faithful_lean.json) — the lean variant relaxes DetectGPT's `n_perturbations` from the paper's 100 to 10 so the full sweep fits a 15-hour GPU pod budget.

**Fast-DetectGPT ([Bao, Zhao, Teng, Yang, Wan, 2024](https://arxiv.org/abs/2310.05130))** — Replaces DetectGPT's perturbation step with a single forward pass. For a candidate, the conditional-probability curvature score is the per-token log-probability under a scoring LM minus the per-token log-probability under a reference LM, normalised by its standard deviation. Wrapper defaults: GPT-2 / GPT-2 scorer-reference pair ([`fast_detect_gpt.py:38-43`](../test/baselines/fast_detect_gpt.py#L38-L43)). Paper-faithful pair: EleutherAI/gpt-neo-2.7B + EleutherAI/gpt-j-6B ([`baselines_paper_faithful.json`](../scripts/baselines_paper_faithful.json)). The local arxiv result was run at the GPT-2 / GPT-2 default; under that configuration the scoring and reference distributions are identical and the discrepancy degenerates mathematically to zero, leaving AUC at 0.500 as a noise floor ([`interim_results.ipynb`](../models/interim_results.ipynb) §C). The GPT-Neo-2.7B / GPT-J-6B paper-grade run is pending pod-2-v2.

**DetectGPT ([Mitchell, Lee, Khazatsky, Manning, Finn, 2023](https://arxiv.org/abs/2301.11305))** — Perturbation-based local-log-probability test. For a candidate `x`, generates N perturbations by T5-mask-filling random spans and scores `s(x) = log p(x) − mean_i log p(x_tilde_i)`. Positive and large s indicates a local maximum on the scoring LM's log-probability surface, which the paper characterises as the LLM-text signature. Wrapper defaults: GPT-2 scoring model, T5-base mask filler, N = 10, mask fraction 0.15, span length 2 ([`detect_gpt.py:37-46`](../test/baselines/detect_gpt.py#L37-L46)). The paper uses N = 100; the lean configuration's N = 10 is at the low end of the paper's own ablation grid (their Appendix shows N = 10 retains most of the signal at a tenth of the cost). Paper-faithful configuration: scoring model EleutherAI/gpt-neo-2.7B, mask filler t5-large ([`baselines_paper_faithful.json`](../scripts/baselines_paper_faithful.json)). DetectGPT clean arxiv numbers exist locally ([`models/baseline_results/arxiv_clean/arxiv_merged__detect_gpt.metrics.json`](../models/baseline_results/arxiv_clean/)) but were not yet folded into the §3 tables at the time of writing.

**Binoculars ([Hans, Schwarzschild, Cherepanova, Kazemi, Saha, Goldblum, Geiping, Goldstein, 2024](https://arxiv.org/abs/2401.12070))** — Perplexity / cross-perplexity ratio between same-family observer and performer LMs. For a candidate `x`, the score is `ppl_obs(x) / x_ppl_obs_perf(x)`, lower indicating more AI-like. Wrapper defaults: gpt2 observer + gpt2-medium performer ([`binoculars.py:54-60`](../test/baselines/binoculars.py#L54-L60)); paper-faithful pair: Falcon-7B + Falcon-7B-Instruct, loaded in 4-bit ([`baselines_paper_faithful.json`](../scripts/baselines_paper_faithful.json)). The arxiv result here is from the GPT-2 pair; per §1.6 it exhibits the sign-flip phenomenon.

**R-Detect ([Zhang et al., ICLR 2024, OpenReview z9j7wctoGV](https://openreview.net/forum?id=z9j7wctoGV); reference implementation [xLearn-AU/R-Detect](https://github.com/xLearn-AU/R-Detect))** — Deep-kernel relative two-sample test. For a candidate, the wrapper repeatedly subsamples HWT and MGT reference sets, encodes everything with `roberta-base-openai-detector`, and tests whether the candidate's distribution is significantly closer to HWT or MGT in the learned kernel space ([`r_detect.py:7-43`](../test/baselines/r_detect.py#L7-L43)). Wrapper configuration: 20 rounds, 1 000 reference samples drawn from `data/dataset_ready_final/train.jsonl` ([config block in `arxiv_merged__r_detect.metrics.json:5-15`](../models/baseline_results/arxiv_clean/arxiv_merged__r_detect.metrics.json)). The R-Detect implementation is vendored at `test/third_party/r_detect/`.

**RADAR ([Hu, Chen, Ho, 2023](https://arxiv.org/abs/2307.03838))** — Adversarially-trained Vicuna-7B sequence classifier (`TrustSafeAI/RADAR-Vicuna-7B`). Thin wrapper: tokenize, forward, softmax, take the AI-probability column. RADAR is in the local results for clean arxiv ([`arxiv_merged__radar.metrics.json` exists](../models/baseline_results/arxiv_clean/)) but does not yet appear in the §3 numbers because the laptop GPU did not fit the 7B fp16 footprint; it will be re-run on the eval pod.

**RAIDAR ([Mao, Wang, Mu, Pu, Sun, Sapiro, Vondrick, 2024](https://arxiv.org/abs/2401.12970))** — "geneRative AI Detection viA Rewriting". For a candidate, an Ollama LLM is asked to rewrite it; the score is the edit similarity between candidate and rewrite. Human text gets edited more heavily than AI text (the rewriter is closer in style to AI input). Wrapper: rewriter `llama3`, threshold 0.65, 120 s timeout ([`raidar.py:32-40`](../test/baselines/raidar.py#L32-L40); [`arxiv_merged__raidar.metrics.json` config](../models/baseline_results/arxiv_clean/arxiv_merged__raidar.metrics.json)).

---

## Section 3 — Results: in-distribution vs. out-of-distribution

### 3.1 In-distribution (val.npz, n = 589; 102 human / 487 AI)

The val split is the calibration set — early stopping and threshold-search are tuned on it — so the headline numbers here represent a saturated upper bound rather than a held-out generalisation estimate. With that caveat, the in-house results at the default 0.5 threshold are ([`interim_results.ipynb`](../models/interim_results.ipynb) §A.1):

| Model | acc | macro-F1 | ROC-AUC |
|---|---:|---:|---:|
| `fusion_gating` | 0.9983 | 0.9970 | 0.9997 |
| `classical_svm` | 0.9983 | 0.9970 | 1.0000 |
| `rf_nela_only`  | 0.9966 | 0.9941 | 0.9998 |
| `rf_style_only` | 0.8930 | 0.7824 | 0.9316 |
| `rf_trace_only` | 0.8404 | 0.5287 | 0.8659 |

Under the strict-FPR ≤ 1 % constraint ([§A.2](../models/interim_results.ipynb)) the picture barely changes for the combined models: `classical_svm` retains macro-F1 = 0.9970 (TPR = 1.0000 at threshold 0.6560), `fusion_gating` and `rf_nela_only` retain 0.9941 (TPR = 0.9979). The combined-modality and NELA-only models are at the saturation ceiling.

The single-modality story diverges sharply under strict FPR. `rf_style_only` collapses to macro-F1 = 0.4904 (TPR = 0.3758) and `rf_trace_only` to macro-F1 = 0.2463 (TPR = 0.0575). The 0.5872 / 0.7913 macro-F1 the stand-alone Style/TRACE models achieve at the default threshold therefore *cannot be operationalised at FPR ≤ 1 %* — they hold up only in a regime that tolerates substantially more human false positives than the deployment constraint allows.

**Take-home for §3.1.** `fusion_gating`, `classical_svm`, and `rf_nela_only` all reach ≥ 0.99 macro-F1 under the strict-FPR ≤ 1 % regime in distribution; the floor set by Style-alone (0.78) and TRACE-alone (0.53) macro-F1 quantifies the per-modality stand-alone signal. The in-distribution upper bound is a property of the matched-rewrite construction and the seventeen >1-σ NELA dims it produces, not a property of any single model's expressive power.

### 3.2 OOD clean (arxiv.npz, n = 2 574; 1 287 human abstracts + 1 287 Claude-haiku rewrites)

The arxiv set is OOD on two axes simultaneously: domain (high-school exam essays → academic abstracts) and LLM family (open-source 7B rewriters → Claude-haiku). The 14-detector unified table at the strict-FPR ≤ 1 % operating point, sorted by clean ROC-AUC descending, is computed from [`test/results/arxiv_eval/summary_clean.csv`](../test/results/arxiv_eval/summary_clean.csv) and the per-detector JSONs in [`models/baseline_results/arxiv_clean/`](../models/baseline_results/arxiv_clean/) and [`models/ready_models/`](../models/ready_models/):

| Detector | AUC | AUC_eff | acc | macro-F1 | TPR @ FPR ≤ 1 % |
|---|---:|---:|---:|---:|---:|
| `classical_logreg`         | 0.8946 | 0.8946 | 0.5944 | 0.5189 | 0.1981 |
| `baseline:detect_gpt`      | 0.8276 | 0.8276 | 0.5136 | 0.3703 | 0.0365 |
| `fusion_gating`            | 0.8063 | 0.8063 | 0.5400 | 0.4228 | 0.0894 |
| `classical_xgboost`        | 0.7962 | 0.7962 | 0.5171 | 0.3775 | 0.0435 |
| `classical_hist_gbm`       | 0.7898 | 0.7898 | 0.5198 | 0.3830 | 0.0490 |
| `rf_nela_only`             | 0.7222 | 0.7222 | 0.5377 | 0.4283 | 0.0637 |
| `classical_mlp`            | 0.6829 | 0.6829 | 0.6807 | 0.6520 | 0.3217 |
| `baseline:radar` (4-bit)   | 0.6669 | 0.6669 | 0.5703 | 0.4781 | 0.1500 |
| `baseline:raidar`          | 0.6506 | 0.6506 | 0.5167 | 0.3767 | 0.0427 |
| `baseline:r_detect`        | 0.5608 | 0.5608 | 0.5082 | 0.3733 | 0.0210 |
| `classical_svm`            | 0.5710 | 0.5710 | 0.5023 | 0.3465 | 0.0140 |
| `rf_style_only`            | 0.5546 | 0.5546 | 0.4977 | 0.3370 | 0.0047 |
| `rf_trace_only`            | 0.5000 | 0.5000 | 0.5000 | 0.3460 | 0.0117 |
| `baseline:binoculars`      | 0.2067 | 0.7933 | 0.4949 | 0.3311 | 0.0000 |
| `baseline:fast_detect_gpt` | 0.5000 | 0.5000 | 0.5000 | 0.3333 | 0.9984 |

**Correction.** The 2026-05-28 draft of this report listed `fusion_gating` as the strongest in-house ranker on OOD clean text (AUC = 0.8063). The corrected reading from the full 14-detector sweep in [`summary_clean.csv:10`](../test/results/arxiv_eval/summary_clean.csv) is that **`classical_logreg` is the strongest in-house ranker at AUC = 0.8946** — a 9-pp gap above `fusion_gating`. The previous §3.2 narrative pre-dated the full sweep and was working from a 7-detector subset of the §E notebook cell. The mechanism that places linear regression above gated fusion on this corpus is that `classical_logreg` exploits the strong NELA decision space directly, without paying the asymmetric modality-dropout regularisation cost that the four neural fusion variants pay at training time ([`training/config.py:34-36`](../training/config.py#L34-L36)). On a corpus where NELA still discriminates strongly — and Claude-haiku rewrites of arxiv abstracts apparently still leave seventeen-σ-class NELA evidence in place — the regularised gating layer is *trading off* expected NELA signal for hypothetical NELA-distribution-shift robustness, and on this particular OOD slice the trade does not pay.

Two further observations follow from the leaderboard:

- `baseline:detect_gpt` at AUC = 0.8276 ([`models/baseline_results/arxiv_clean/arxiv_merged__detect_gpt.metrics.json:10326`](../models/baseline_results/arxiv_clean/arxiv_merged__detect_gpt.metrics.json)) is **above all four in-house neural fusion variants** on OOD ranking, even at the non-paper-grade GPT-2 + T5-small + N = 10 small-model configuration documented in [`scripts/baselines_detect_gpt_local.json`](../scripts/baselines_detect_gpt_local.json). A perturbation-based zero-shot method beats the in-house learned fusion heads on the OOD ranking task. The DetectGPT paper recommends N = 100 with GPT-Neo-2.7B + T5-large; the present N = 10 / T5-small run is therefore a *floor*, and the paper-grade pod-2-v2 number is expected to be a few points higher still.
- `baseline:radar` ([`scripts/baselines_radar_4bit.json`](../scripts/baselines_radar_4bit.json), 4-bit local) achieves AUC = 0.6669 with per-source accuracy of 70 % on arxiv humans and 54 % on arxiv_rewrite ([`arxiv_merged__radar.metrics.json:10322-10362`](../models/baseline_results/arxiv_clean/arxiv_merged__radar.metrics.json)). The 4-bit quantisation may slightly under-perform fp16 paper-grade; the fp16 Vicuna-7B run is still pending pod-2-v2.

`fusion_gating`'s AUC = 0.8063 is therefore the **third-strongest** in-house-or-baseline ranker on this corpus, not the strongest. It remains the only neural-fusion variant with above-chance OOD AUC (the three symmetric variants `concat`/`mlp`/`attention` collapse to 0.36–0.68 AUC) and that distinction still matters: when the inter-variant comparison is held fixed, `fusion_gating` is the one fusion strategy whose architecture admits OOD generalisation at all.

The Fast-DetectGPT row is anomalous: AUC = 0.500 (the GPT-2/GPT-2 noise floor) yet TPR = 0.9984 at the strict-FPR threshold. The mechanism is the same one that gives Fast-DetectGPT acc = 0.50 on the balanced set in §C — with identical scoring and reference models the discrepancy collapses to zero, every sample's score is identical, and any threshold drawn through that point classifies all samples on one side. It is not a real detection signal.

**Take-home for §3.2.** The strongest OOD ranker on this corpus is the linear classical baseline `classical_logreg` (AUC = 0.8946), followed by the perturbation-based zero-shot baseline DetectGPT (0.8276) and then `fusion_gating` (0.8063). The in-house fusion architecture is **not** the strongest detector on OOD academic prose; this is a correction to the prior draft. At the strict-FPR ≤ 1 % operating point, only `classical_logreg` (TPR = 0.198), `classical_mlp` (0.322 — but see §3.3 for its prior-shift artefact), and `baseline:radar` (0.150) cross 10 % recall. Paper-grade Falcon-7B Binoculars, GPT-J-6B Fast-DetectGPT, fp16 Vicuna-7B RADAR, and T5-large + N = 100 DetectGPT numbers from the eval pod will supersede the small-model entries above.

### 3.3 Humanized arxiv (arxiv_humanized.npz, n = 3 861; full cache)

The humanized cache is complete: 3 861 rows comprising 1 287 human abstracts, 1 287 Adv-P rewrites of the Claude-haiku AI half, and 1 287 TempParaphraser rewrites of the same AI half. Per-detector clean → humanized AUC delta, sorted by attack severity ((AUC_eff_clean − AUC_eff_humanized) descending), is computed from [`summary_clean.csv`](../test/results/arxiv_eval/summary_clean.csv) and [`summary_humanized.csv`](../test/results/arxiv_eval/summary_humanized.csv):

| Detector | AUC_clean | AUC_humanized | AUC_eff_clean | AUC_eff_humanized | ΔAUC_eff |
|---|---:|---:|---:|---:|---:|
| `classical_xgboost`        | 0.7962 | 0.4546 | 0.7962 | 0.5454 | −0.2508 |
| `classical_svm`            | 0.5710 | 0.2612 | 0.5710 | 0.7388 | +0.1678 |
| `baseline:binoculars`      | 0.2067 | 0.4525 | 0.7933 | 0.5475 | −0.2458 |
| `classical_hist_gbm`       | 0.7898 | 0.4609 | 0.7898 | 0.5391 | −0.2507 |
| `rf_nela_only`             | 0.7222 | 0.4459 | 0.7222 | 0.5541 | −0.1681 |
| `baseline:radar` (4-bit)   | 0.6669 | 0.5040 | 0.6669 | 0.5040 | −0.1629 |
| `fusion_gating`            | 0.8063 | 0.6444 | 0.8063 | 0.6444 | −0.1619 |
| `classical_logreg`         | 0.8946 | 0.8187 | 0.8946 | 0.8187 | −0.0759 |
| `baseline:detect_gpt`      | 0.8276 | 0.8399 | 0.8276 | 0.8399 | +0.0123 |
| `classical_mlp`            | 0.6829 | 0.7117 | 0.6829 | 0.7117 | +0.0288 |
| `rf_style_only`            | 0.5546 | 0.4826 | 0.5546 | 0.5174 | −0.0372 |
| `rf_trace_only`            | 0.5000 | 0.5000 | 0.5000 | 0.5000 |  0.0000 |
| `baseline:raidar`          | 0.6506 | 0.4332 | 0.6506 | 0.5668 | −0.0838 |
| `baseline:r_detect`        | 0.5608 | (single-class on humanized AI-only JSON, see below) | | | |
| `baseline:fast_detect_gpt` | 0.5000 | (single-class) | | | |

(AUC_eff is the sign-flipped maximum `max(AUC, 1 − AUC)` defined in §1.6; ΔAUC_eff = AUC_eff_clean − AUC_eff_humanized, so positive values mean humanization *helped* the detector after the sign-flip correction.) `classical_svm` again shows the rare positive eff-delta because of the sign-flip behaviour of §1.6: humanized AI rows shift the SVM-kernel feature distribution further from the in-distribution boundary, and after sign-flipping the result is a discrimination improvement.

**DetectGPT is the strongest humanizer-robust baseline in the report.** From the AI-only humanized JSON ([`arxiv_humanized_ai_only__detect_gpt.metrics.json:10354-10364`](../models/baseline_results/arxiv_humanized/arxiv_humanized_ai_only__detect_gpt.metrics.json)), per-source AI catch rate on the humanized AI half is **Adv-P 96.3 %** and **TempParaphraser 95.6 %**. Compared to its clean Claude-haiku rewrite catch rate of **91.8 %** ([`arxiv_merged__detect_gpt.metrics.json:10361-10363`](../models/baseline_results/arxiv_clean/arxiv_merged__detect_gpt.metrics.json)), DetectGPT catches **more humanized AI than clean AI**. This is the strongest humanizer-robustness signal in the entire results table. A plausible mechanism — to be verified rather than asserted — is that DetectGPT's perturbation-based score interrogates local-maximum-of-log-probability structure under a base LM (here GPT-2): humanizers that distort style without changing semantic content tend to introduce additional local-log-probability concavities that GPT-2 finds *more* surprising than the unhumanized AI text, not less. If correct, this would imply that perturbation-based detectors interrogate a signature that humanizers structurally *amplify*, while discriminative-classifier detectors interrogate a signature that humanizers structurally *disguise*. The DetectGPT clean → humanized AUC actually goes **up by 1.2 pp** in the merged 3 861-row calculation ([`summary_humanized.csv:3`](../test/results/arxiv_eval/summary_humanized.csv)), the only baseline detector in the table to gain AUC under humanization at all.

**RADAR shows the converse pattern.** Per-source AI catch on humanized AI is **Adv-P 34.1 %** and **TempParaphraser 17.5 %** ([`arxiv_humanized_ai_only__radar.metrics.json:10354-10362`](../models/baseline_results/arxiv_humanized/arxiv_humanized_ai_only__radar.metrics.json)), compared to RADAR's clean arxiv_rewrite catch of **53.7 %** ([`arxiv_merged__radar.metrics.json:10359-10361`](../models/baseline_results/arxiv_clean/arxiv_merged__radar.metrics.json)). Humanization therefore drops RADAR's per-source AI accuracy by **20–36 pp**, with TempParaphraser the dominant attack. This is the strongest single humanizer-attack signal among baseline detectors. Discriminative classifiers trained against earlier-generation adversarial paraphrasers ([Hu, Chen, Ho, 2023, arXiv:2307.03838](https://arxiv.org/abs/2307.03838)) appear not to generalise to the newer Adv-P / TempParaphraser attack family. (4-bit quantisation could be a contributing factor; the fp16 Vicuna-7B pod run will arbitrate.)

**classical_mlp is the only model whose macro-F1 *improves* under humanization.** Its clean macro-F1 is 0.6520 and humanized macro-F1 is 0.7281 — a **+7.6 pp gain** ([`summary_clean.csv:11`](../test/results/arxiv_eval/summary_clean.csv); [`summary_humanized.csv:11`](../test/results/arxiv_eval/summary_humanized.csv)). This is not a humanizer-robustness story; it is a **class-prior-alignment artefact**. The clean evaluation set is balanced (1 287 human / 1 287 AI), while the humanized evaluation set has a 1 : 2 prior (1 287 human / 2 574 AI = 1 287 Adv-P + 1 287 Temp). The `classical_mlp` classifier was trained with inverse-frequency balanced class weights against the in-distribution 1 : 4.76 ratio and post-hoc re-thresholded at strict-FPR ≤ 1 %, leaving it with an AI-leaning decision boundary that happens to align with the humanized prior shift far better than with the clean balanced set. Per-source data confirms the mechanism: `classical_mlp` AI catch is 96.7 % on `arxiv_rewrite` clean and 95.7 % / 85.9 % on Adv-P / Temp humanized ([`per_source_humanized.csv:11`](../test/results/arxiv_eval/per_source_humanized.csv)). Honest framing: the macroF1 gain is a *prior-shift advantage*, not a *humanizer-robustness advantage*.

**R-Detect on humanized arxiv** ([`arxiv_eval_with_humanizers__r_detect.metrics.json:15502-15514`](../models/baseline_results/arxiv_humanized/arxiv_eval_with_humanizers__r_detect.metrics.json), full 3 861-row cache, superseding the 3 275-row partial number in the prior draft): arxiv (human) = 0.4141, arxiv_humanized_adv = **0.7327**, arxiv_humanized_temp = **0.2704**. TempParaphraser drives R-Detect's per-source AI accuracy to 27 %, catastrophic for the kernel test, while Adv-P holds at 73 %. The kernel relative test compares candidate-to-HWT distance against candidate-to-MGT distance in the deep-kernel-network feature space; TempParaphraser's multi-temperature stitching ([upstream HJJWorks/TempParaphraser, EMNLP 2025](https://huggingface.co/huangjj877/TempParaphraser)) preserves the candidate's content distribution while softening its lexical surface, which is exactly the manipulation that moves the candidate's feature representation toward the HWT reference cluster without crossing into the MGT cluster. Adv-P, in contrast, performs detector-guided beam search ([Cheng et al., 2025, arXiv:2506.07001](https://arxiv.org/abs/2506.07001)) and produces stylistically aggressive rewrites that R-Detect's kernel still partially separates from human prose. The general pattern — **kernel methods especially vulnerable to *content-distribution-preserving* paraphrase, more robust to *adversarial-style-rewrite*** — is a documentable finding for the methodology section of any kernel-based detector paper, and it still holds against the larger and balanced humanized cache.

**Take-home for §3.3.** Three robustness regimes emerge from the full sweep. (i) Perturbation-based detection (DetectGPT) appears the most humanizer-robust family: it gains per-source AI accuracy under humanization (91.8 % → 96 %) and modestly gains AUC (0.83 → 0.84). (ii) Discriminative-classifier detection (RADAR) collapses by 20–36 pp per-source under the same humanizers, with TempParaphraser the dominant attack. (iii) Kernel-based detection (R-Detect) is uniquely vulnerable to *content-distribution-preserving* paraphrase (TempParaphraser at 27 %) while being roughly twice as robust to *adversarial-style-rewrite* (Adv-P at 73 %). The strongest in-house ranker on clean (`classical_logreg`, AUC 0.8946) loses only 7.6 pp AUC under humanization, second only to DetectGPT among detectors with both above-chance clean and humanized rankings. `fusion_gating` loses ~16 pp, approximately matching the `rf_nela_only` drop (also ~17 pp). `classical_mlp`'s macroF1 *gain* under humanization is a class-prior-alignment artefact, not a robustness story.

---

## Section 4 — Methodology, setup, and discussion

### 4.1 Dataset construction

The training cache is built from three pools harmonised into a single JSONL schema in [`data/dataset_ready_final/`](../data/dataset_ready_final/) ([METHODOLOGY.md §1.1](../METHODOLOGY.md)): 1 498 USE student essays (human, with stable `author_id`), 6 051 LLM rewrites of those essays produced by LLaMA-3, Mistral, Gemma, Phi-3, and Qwen2 (label = ai, inheriting the source essay's `author_id`), and 5 537 auxiliary HC3 / ArguGPT / RAID records that are dropped by the post-filter pipeline. The USE-essay + 5-rewriters construction is a deliberate *matched-condition* experimental design: topic, source author, and approximate length are held constant across the human/AI pair (only "writer identity" is swapped). This isolates the modelling target — *mechanical writing style* — from confounders (topic, length, vocabulary domain). The same general principle motivates the rewrite-attribution framework of [Patel et al. (2024)](https://arxiv.org/abs/2410.12757) and the contrastive author-style training in [Wegmann et al. (2022)](https://aclanthology.org/2022.repl4nlp-1.26/): if two texts share content but differ in style, then any classifier that separates them is *necessarily* using a style signal.

Two filters narrow the kept set:

- `--require-known-author` drops records whose `author_id` is unknown ([`training/build_dataset.py:301`](../training/build_dataset.py)). This removes HC3 human answers, which have no per-author identity.
- `--min-human-siblings 2` drops records whose author contributes fewer than two other *human* texts as TRACE context, *excluding the source text of any LLM rewrite* ([`training/build_dataset.py:303`](../training/build_dataset.py); exclusion logic in [`training/rebuild_trace_author.py::trace_human_siblings`](../training/rebuild_trace_author.py)).

These filters are not optional. TRACE's input contract requires ≥ 1 usable human sibling after the rewrite-source exclusion; for an LLM-rewrite anchor the exclusion eats one sibling, so the author must contribute ≥ 2 total human texts. Relaxing either filter would collapse TRACE to near-zero on the affected anchors and effectively remove it from the comparison ([METHODOLOGY.md §1.3](../METHODOLOGY.md), [`NELA_DOMINANCE_ANALYSIS.md` §3](NELA_DOMINANCE_ANALYSIS.md)). The post-filter per-split counts, from [`data/features/meta.json`](../data/features/meta.json):

| Split | records | human | ai | human : ai |
|---|---:|---:|---:|---:|
| train | 4 051 | 703 | 3 348 | 1 : 4.76 |
| val   |   667 | 116 |   551 | 1 : 4.75 |
| test  |   829 | 144 |   685 | 1 : 4.76 |

The exact 1 : ≈4.76 ratio is structural: each kept human essay survives with all five of its LLM rewrites attached. Inverse-frequency class weighting compensates during training; the strict-FPR ≤ 1 % evaluation regime is per-class-rate and unaffected by counts ([METHODOLOGY.md §1.3](../METHODOLOGY.md)).

The arxiv OOD evaluation set is a deliberate *double shift*: domain (USE-essay → academic abstract) and LLM family (open-source 7B rewriter ensemble → Claude-haiku). This is more aggressive than the field's typical one-axis shift; the OOD numbers in §3 should be read as a pessimistic upper bound on adversarial deployment difficulty. The reverse-engineering implication is that an in-distribution AUC = 1.000 number from any detector paper that does not test against double-shift transfer should be treated as uninformative for deployment.

### 4.2 Humanizers as adversarial attacks

Two humanizers were applied to the AI half of the arxiv set, producing the `arxiv_humanized_adv` and `arxiv_humanized_temp` sub-corpora that drive §3.3.

**Adversarial Paraphrasing (Adv-P)** ([Cheng et al., 2025, arXiv:2506.07001](https://arxiv.org/abs/2506.07001)) — Training-free, detector-guided beam search: at each generation step, candidate continuations from a LLaMA-3-8B-Instruct rewriter are re-scored against `roberta-base-openai-detector` and the candidate with the lowest detection-side score is greedy-selected. The resulting text is semantically faithful to the input and locally optimised to defeat the guidance detector ([`scripts/humanize_arxiv_adversarial.py:1-40`](../scripts/humanize_arxiv_adversarial.py); upstream repo and paper README at [`scripts/_workdirs/adversarial_paraphrasing/README.md:1-12`](../scripts/_workdirs/adversarial_paraphrasing/README.md)). The reported insight from the paper — that detector-guided rewrites transfer across detectors because all well-trained detectors converge to a shared notion of human-distribution-typicality — is consistent with the AUC degradation pattern we observe across multiple in-house and baseline detectors in §3.3.

**TempParaphraser** ([HJJWorks/TempParaphraser, upstream README; EMNLP 2025 system #1607](https://huggingface.co/huangjj877/TempParaphraser)) — A fine-tuned paraphraser served (in the upstream) via vLLM, that generates candidates at multiple sampling temperatures and stitches per-temperature outputs into a single rewrite. The local driver ([`scripts/humanize_arxiv_temppara.py:1-40`](../scripts/humanize_arxiv_temppara.py)) loads `huangjj877/TempParaphraser` via `transformers` (skipping vLLM service management for the ~1.3 k records). Conceptually TempParaphraser is a *content-distribution-preserving* paraphrase, in contrast to Adv-P's *adversarial-style-rewrite* — and the empirical asymmetry between the two on R-Detect (per-source AI accuracy of 0.7327 versus 0.2704; §3.3) is the cleanest demonstration in our results that *what kind of perturbation* the humanizer introduces matters more than *how aggressive* the perturbation is.

The methodological point is that humanizers simulate the realistic threat model: an adversary deliberately disguising LLM output. Evaluation under humanization is therefore the closest analogue we currently have to deployment-time robustness, and is the regime in which the gated fusion architecture's 17-pp AUC drop is the right number to publish alongside its in-distribution 1.000.

### 4.3 The strict-FPR ≤ 1 % operating-point philosophy

[METHODOLOGY.md §6.2](../METHODOLOGY.md) defines the strict-FPR ≤ 1 % evaluation regime: from the val-split ROC curve, select the lowest threshold at which val-FPR ≤ 0.01; report test-set TPR, precision, and FPR at that threshold. The motivation is straightforward — in deployment, a human falsely flagged as AI is much costlier than a missed AI, and a detector that operates at 5–10 % human FPR (the implicit acc@0.5 operating point on a balanced test set) is unusable in any real moderation pipeline. The acc@0.5 / macro-F1@0.5 / ROC-AUC trio that dominates the detector literature is a *ranking-quality* report; the strict-FPR TPR is an *operational-quality* report, and the two diverge sharply under distribution shift (§1.5). We retain both in [`pipeline_summary.json`](../models/ready_models/pipeline_summary.json) and in every individual `.metrics.json` to allow comparison against the prevailing literature convention while making the deployment-relevant quantity primary.

### 4.4 Discussion: which models win

**In distribution**, the classical `classical_svm` and `classical_logreg` baselines tie or beat all four neural fusion variants. From [`pipeline_summary.json`](../models/ready_models/pipeline_summary.json): `classical_logreg` achieves the highest test macro-F1@0.5 (0.9979) and ROC-AUC (1.0000); `classical_svm` at 0.9937 / 0.99995 is essentially indistinguishable from `neural_concat` and `neural_mlp` (0.9937 / 0.99998). The kernel and linear methods exploit the seventeen-strong NELA decision space at full strength — they pay no modality-dropout tax — and on a saturated task that is enough to land at the joint ceiling. The neural fusion variants are not winning the in-distribution argument; they are paying the modality-dropout regularisation cost in exchange for what the OOD ranking turns up.

**On OOD ranking**, the strongest in-house detector is `classical_logreg` at AUC = 0.8946 ([`summary_clean.csv:10`](../test/results/arxiv_eval/summary_clean.csv)), not `fusion_gating`. Among baselines, `baseline:detect_gpt` (N = 10, GPT-2 + T5-small) reaches AUC = 0.8276 — above all four in-house neural fusion variants. `fusion_gating` is in third place at 0.8063, with `rf_nela_only` at 0.7222 and the symmetric fusion siblings (`concat`/`mlp`/`attention`) collapsing to 0.36–0.68. The methodological argument for `fusion_gating` is therefore more nuanced than the prior draft suggested: it is the only neural-fusion variant whose architecture survives the double distribution shift at all (the symmetric variants fall to or below chance), and it loses less to humanization than the symmetric variants. But it is not the strongest OOD ranker on the full leaderboard, and the prior draft's claim that "modality-dropout-trained fusion generalises better than the kernel/tree classical models" needs to be restricted to the comparison with `classical_svm` (0.5710 OOD AUC, collapse confirmed) — the linear `classical_logreg` baseline manifestly *does* generalise.

**On OOD humanized**, three distinct robustness mechanisms emerge from §3.3, all worth documenting separately.

1. **DetectGPT (perturbation) gains AI-side accuracy under humanization.** Per-source AI catch goes 91.8 % (clean Claude-haiku rewrite) → 96.3 % (Adv-P) / 95.6 % (TempParaphraser); merged AUC goes 0.8276 → 0.8399 ([`summary_humanized.csv:3`](../test/results/arxiv_eval/summary_humanized.csv)). The only baseline detector in the table that gains AUC under humanization. Hypothesised mechanism (to be verified): humanizers introduce extra stylistic distortions that GPT-2 finds more surprising than the unhumanized AI text, *amplifying* rather than disguising the local-log-probability signature DetectGPT interrogates.
2. **classical_mlp (prior-shift advantage) gains macroF1 under humanization.** Clean macroF1 0.6520 → humanized macroF1 0.7281 (the only model with a positive Δ macroF1 in the table). This is a class-prior-alignment artefact rather than a robustness mechanism per se: the inverse-frequency-weighted classifier paired with strict-FPR re-thresholding has an AI-leaning decision boundary that aligns with the humanized 1 : 2 prior shift. It is not a property of `classical_mlp` that transfers to balanced humanized deployment.
3. **`fusion_gating` and `rf_nela_only` lose ~16–17 pp** under humanization, while `classical_logreg` loses only 7.6 pp ([`summary_humanized.csv:10`](../test/results/arxiv_eval/summary_humanized.csv); AUC 0.8946 → 0.8187). The linear classifier is more humanizer-robust on NELA features than the regularised gated fusion head. RADAR drops 16 pp (0.6669 → 0.5040) and per-source on humanized AI collapses to 34 % / 17 % — discriminative-classifier methods appear structurally vulnerable to the newer Adv-P / TempParaphraser attack family.

**On the detector arms race.** The §3.3 results suggest three distinct vulnerability profiles by detector family — perturbation-based (DetectGPT-style; humanizer-amplification, *more* robust), discriminative-classifier (RADAR-style; humanizer-disguise, *less* robust), and kernel-based (R-Detect-style; vulnerable specifically to *content-preserving paraphrase*, comparatively robust to *adversarial-style-rewrite*). The field has not yet converged on which detector family is structurally most robust against the next generation of humanizers, and the present results offer one piece of evidence: at fixed evaluation, the perturbation-based and linear-classical families generalise better than the discriminative-classifier and kernel-based families against the current Adv-P / TempParaphraser attack pair. This is a publishable finding category and should inform detector-design decisions for the field.

**Humanizer effect** is real and measurable but does not collapse detection to chance. A 7-to-17 pp AUC drop on the strongest detectors is consistent with the published behaviour of Adv-P ([Cheng et al., 2025](https://arxiv.org/abs/2506.07001)) and TempParaphraser; it leaves enough residual signal that future detector designs can in principle re-close the gap. The DetectGPT result also shows that *negative* humanizer impact (i.e. humanizers helping the detector) is achievable for one detector family. The two effects together suggest that the humanizer arms race is not yet decided in either direction.

### 4.5 Limitations and future work

The present report should be read with the following caveats:

- **val.npz is the calibration set.** The in-distribution numbers in §3.1 are an upper bound, not a held-out generalisation estimate. The training pipeline's early-stopping and threshold-search both consume the val split.
- **Local test.npz is missing.** Only the pre-filter `test.npz.bak` (1 975 rows, different filter regime) exists locally. The canonical 829-row filtered test split needs to be rebuilt via `python -m training.build_dataset --splits test --require-known-author --min-human-siblings 2` before the in-distribution held-out numbers can be reported in the §3 tables. The [`pipeline_summary.json`](../models/ready_models/pipeline_summary.json) test numbers were produced at training time and are still trustworthy as a snapshot, but a fresh local rebuild is needed before publication.
- **Paper-grade pod-2-v2 baselines pending.** Falcon-7B Binoculars, GPT-J-6B Fast-DetectGPT, T5-large mask-filler DetectGPT, and Vicuna-7B RADAR are not local at time of writing; the report will be updated when those land via [`scripts/baselines_paper_faithful.json`](../scripts/baselines_paper_faithful.json) / [`baselines_paper_faithful_lean.json`](../scripts/baselines_paper_faithful_lean.json) on the eval pod (see [`docs/RUNPOD_eval_v2.md`](RUNPOD_eval_v2.md)).
- **TRACE encoder substitution.** The pipeline uses `AnnaWegmann/Style-Embedding`, not the TRACE paper's unreleased contrastive checkpoint. The decision is documented in [METHODOLOGY.md §2.3](../METHODOLOGY.md); the §1.4 OOD failure mode is partially attributable to this substitution and reviewers may legitimately object that "TRACE as deployed here" is not "TRACE as published". The mitigation is to report results as "TRACE/Wegmann" and to demonstrate, via the §7.3 of METHODOLOGY, that the Wegmann choice is materially superior to the obvious fallback (mpnet-base-v2) in distribution.

Future work, in priority order: (i) rebuild the local 829-row test cache and re-run all §3 tables against a held-out split; (ii) complete pod-2-v2 paper-grade baselines and supersede the small-model Binoculars / Fast-DetectGPT rows in §3.2 / §3.3; (iii) experiment with the four interventions catalogued in [`NELA_DOMINANCE_ANALYSIS.md` §7](NELA_DOMINANCE_ANALYSIS.md) (length-matched-TTR-controlled rewrites, multi-essay-per-author out-of-domain corpora, tighter asymmetric ModDrop schedules, NELA-residual auxiliary loss); and (iv) probe TRACE with a non-Reddit-trained contrastive style encoder to disentangle the §1.4 representation-generalisation failure from any residual data-coverage effect.

---

## References

- Bao, G., Zhao, Y., Teng, Z., Yang, L., Wan, Y. (2024). *Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature.* arXiv:2310.05130. https://arxiv.org/abs/2310.05130
- Cheng, Y., Sadasivan, V. S., et al. (2025). *Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text.* arXiv:2506.07001. https://www.arxiv.org/abs/2506.07001
- Hans, A., Schwarzschild, A., Cherepanova, V., Kazemi, H., Saha, A., Goldblum, M., Geiping, J., Goldstein, T. (2024). *Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text.* arXiv:2401.12070. https://arxiv.org/abs/2401.12070
- Horne, B. D., Khedr, S., Adali, S. (2018). *Sampling the News Producers: A Large News and Feature Data Set for the Study of the Complex Media Landscape.* AAAI ICWSM Workshop. arXiv:1803.10124. https://arxiv.org/abs/1803.10124
- Hu, X., Chen, P.-Y., Ho, T.-Y. (2023). *RADAR: Robust AI-Text Detection via Adversarial Learning.* arXiv:2307.03838. https://arxiv.org/abs/2307.03838
- Liang, Y., Wu, T., Zhang, J., et al. (2024). *TRACE: TRAnsformer-based attribution using Contrastive Embeddings.* arXiv:2402.16638. https://arxiv.org/abs/2402.16638
- Mao, C., Wang, M., Mu, J., Pu, A., Sun, M., Sapiro, G., Vondrick, C. (2024). *RAIDAR: geneRative AI Detection viA Rewriting.* arXiv:2401.12970. https://arxiv.org/abs/2401.12970
- Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., Finn, C. (2023). *DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature.* arXiv:2301.11305. https://arxiv.org/abs/2301.11305
- Neverova, N., Wolf, C., Taylor, G., Nebout, F. (2016). *ModDrop: Adaptive Multi-Modal Gesture Recognition.* IEEE Transactions on Pattern Analysis and Machine Intelligence 38(8). https://ieeexplore.ieee.org/document/7373645
- Patel, A., et al. (2024). *StyleDistance: Stronger Content-Independent Style Embeddings with Synthetic Parallel Examples.* arXiv:2410.12757. https://arxiv.org/abs/2410.12757
- Wegmann, A., Schraagen, M., Nguyen, D. (2022). *Same Author or Just Same Topic? Towards Content-Independent Style Representations.* Proceedings of the 7th Workshop on Representation Learning for NLP (RepL4NLP), ACL 2022. https://aclanthology.org/2022.repl4nlp-1.26/
- Zhang, S., et al. (2024). *R-Detect: A Deep-Kernel Relative Test for Machine-Generated Text Detection.* ICLR 2024. OpenReview ID z9j7wctoGV. https://openreview.net/forum?id=z9j7wctoGV ; reference implementation at https://github.com/xLearn-AU/R-Detect

---

*End of report.*
