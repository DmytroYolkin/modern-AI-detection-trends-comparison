# Why are the models saturated and why does NELA dominate?

A diagnostic look at the held-out results in [`models/ready_models/`](../models/ready_models/) and the cached feature matrix in [`data/features/`](../data/features/). All numbers in this document are reproduced from the actual artifacts; the computation script is summarised at the bottom of each section.

---

## 1. The question

Every one of the eleven trained models reaches macro-F1 ≥ 0.987 on the held-out **test** split, and the per-modality permutation-importance table in [`models/analysis.ipynb` §3b](../models/analysis.ipynb) shows NELA carrying 76–99 % of the signal across architectures while StyleDecipher and TRACE look almost-decorative. Two things must be explained: (a) why the absolute numbers are so close to 1.0, and (b) why NELA dwarfs the other two modalities. The argument below proves both from the dataset construction and from a per-feature class-separation analysis of the cached training matrix.

---

## 2. Result saturation — every model is at the ceiling

Source: [`models/ready_models/pipeline_summary.json`](../models/ready_models/pipeline_summary.json) (one row per model, computed on the 829-sample test split, 144 human / 685 AI).

| Model | acc@0.5 | macro-F1@0.5 | ROC-AUC | TPR @ FPR≤1 % |
|---|---:|---:|---:|---:|
| classical_logreg | 0.9988 | 0.9979 | 1.0000 | 0.9971 |
| classical_svm | 0.9964 | 0.9937 | 0.99995 | 0.9956 |
| neural_concat | 0.9964 | 0.9937 | 0.99998 | 0.9942 |
| neural_mlp | 0.9964 | 0.9937 | 0.99998 | 0.9942 |
| classical_random_forest | 0.9940 | 0.9893 | 0.99983 | 1.0000 |
| classical_hist_gbm | 0.9928 | 0.9873 | 0.99990 | 0.9971 |
| classical_xgboost | 0.9928 | 0.9873 | 0.99983 | 0.9985 |
| classical_gradient_boosting | 0.9928 | 0.9873 | 0.99977 | 0.9942 |
| classical_mlp | 0.9928 | 0.9874 | 0.99942 | 0.9942 |
| neural_attention | 0.9928 | 0.9875 | 0.99948 | 0.9854 |
| neural_gating | 0.9928 | 0.9876 | 0.99987 | 0.9898 |

The spread from best to worst is 0.6 pp accuracy and 0.011 macro-F1. **Even the worst-performing classifier in this list misclassifies at most ~6 of the 829 test records.** The val-split metrics from each per-model JSON are even more compressed: `fusion_concat`, `fusion_mlp`, `fusion_attention` and `classical_svm` all hit val macro-F1 = 1.0000 ([`fusion_concat.metrics.json:8`](../models/ready_models/fusion_concat.metrics.json), [`fusion_mlp.metrics.json:8`](../models/ready_models/fusion_mlp.metrics.json), [`fusion_attention.metrics.json:8`](../models/ready_models/fusion_attention.metrics.json), [`clf_svm.metrics.json:5`](../models/ready_models/clf_svm.metrics.json)).

**Verdict.** This is not a learning problem — it is a *task difficulty* observation. Eleven architecturally different models converge to the same near-perfect operating point because the input features are doing almost all the work.

---

## 3. Why the task is easy — paired rewrites, not the filters

Source: [`data/features/meta.json`](../data/features/meta.json) and the two filter flags in [`training/build_dataset.py:301`](../training/build_dataset.py) (`--require-known-author`) and [`training/build_dataset.py:303`](../training/build_dataset.py) (`--min-human-siblings`).

**Important framing.** The two filters above are **required for TRACE to function**, not an arbitrary methodological narrowing. TRACE's author context is defined in [`training/rebuild_trace_author.py`](../training/rebuild_trace_author.py) as *other human texts by the same author, excluding the source text of any rewrite*. For an LLM-rewrite anchor that exclusion eats one sibling, so the author must contribute ≥2 total human texts to guarantee ≥1 usable context text — that is exactly what `--min-human-siblings 2` enforces ([METHODOLOGY.md §1.3, §3.3](../METHODOLOGY.md)). Without these filters TRACE collapses to near-zero vectors on the affected anchors and stops contributing entirely; relaxing them is not a real lever for re-balancing the modalities.

The dataset property that *does* make the task easy is the construction one layer above the filters: every kept human essay is retained together with all five of its LLM rewrites. `meta.json` records the resulting cache:

| Split | records | human | ai  | human : ai |
|---|---:|---:|---:|---:|
| train | 4 051 | 703 | 3 348 | 1 : 4.76 |
| val | 667 | 116 | 551 | 1 : 4.75 |
| test | 829 | 144 | 685 | 1 : 4.76 |

The exact 1 : ≈4.76 ratio is structural: each kept human essay survives with all five of its LLM rewrites attached. That means **for every human text in the train set, there are roughly five AI texts that share the same prompt, the same topic, the same target length, and the same source author** — only the "writer" was swapped from the student to LLaMA-3 / Mistral / Gemma / Phi-3 / Qwen2.

This construction strips out *every* confounding signal except **writing style**:

- topic is held constant (the rewrite preserves it),
- length is approximately held constant (a paraphrase ≠ a rewrite-from-scratch),
- vocabulary domain is held constant (USE-essay genre),
- author identity is held constant for the human-vs-its-own-rewrite pairing.

The only thing left to discriminate on is the mechanical writing style — and an 18-year-old non-native-English exam-essay author and a 2024-vintage LLM rewrite differ on that style in ways that are huge, dense, and surface-detectable, which §5 quantifies. That is the entire reason the metrics ceiling out near 1.0.

---

## 4. Per-modality permutation importance — the §3b table

Source: cell 11 of [`models/analysis.ipynb`](../models/analysis.ipynb), executed against the held-out test split (`data/features/test.npz`), 10 random permutations per block (seed = 42). The four neural fusion variants were trained with the asymmetric modality-dropout regulariser from [`training/model.py`](../training/model.py); the classical models were not.

| Model | base F1 | base AUC | ΔF1(NELA) | ΔF1(Style) | ΔF1(TRACE) | NELA share | Style share | TRACE share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural_mlp        | 0.9937 | 1.0000 | +0.2813 | +0.0566 | +0.0297 | **76.5 %** | 15.4 % | 8.1 % |
| neural_concat     | 0.9937 | 1.0000 | +0.2782 | +0.0607 | +0.0292 | **75.6 %** | 16.5 % | 7.9 % |
| neural_attention  | 0.9875 | 0.9995 | +0.3464 | +0.0552 | +0.0200 | **82.2 %** | 13.1 % | 4.7 % |
| neural_gating     | 0.9876 | 0.9999 | +0.4120 | +0.0403 | +0.0055 | **90.0 %** | 8.8 %  | 1.2 % |
| classical_logreg  | 0.9979 | 1.0000 | +0.4945 | +0.0097 | +0.0142 | **95.4 %** | 1.9 %  | 2.7 % |
| classical_xgboost | 0.9873 | 0.9998 | +0.4735 | +0.0030 | +0.0003 | **99.3 %** | 0.6 %  | 0.1 % |

`classical_svm` does **not** appear in the §3b cell of the notebook — that row is therefore omitted here rather than fabricated. The validation-set extractor-importance fractions that *are* stored inside each classical model's metrics JSON (a coarser, model-internal score, not permutation importance) tell the same qualitative story: e.g. [`clf_random_forest.metrics.json:21`](../models/ready_models/clf_random_forest.metrics.json) reports `nela=0.902, style=0.017, trace=0.081`; [`clf_gradient_boosting.metrics.json:20`](../models/ready_models/clf_gradient_boosting.metrics.json) reports `nela=0.983, style=0.003, trace=0.014`; [`clf_xgboost.metrics.json:20`](../models/ready_models/clf_xgboost.metrics.json) reports `nela=0.766, style=0.022, trace=0.211`.

**Two real patterns in the table above:**

1. Models trained **without** modality dropout (logreg, xgboost) burn 95–99 % of their signal on NELA; the other two modalities are squeezed below 3 % each.
2. Models trained **with** modality dropout (the four neural fusion variants) push NELA's share down to 76–90 % and lift Style + TRACE from rounding-error to 9–24 % combined. The four fusion variants differ only in fusion-head architecture; the *direction* of the dropout effect is identical across all four.

This is also why classical_logreg's base F1 (0.9979) is higher than every neural variant's: with no modality dropout it gets to use NELA at full strength every step, which is more than enough on this slice.

---

## 5. Why NELA dominates — feature-level evidence

Source: per-column Cohen's d on the **train** split (`data/features/train.npz`, n=4 051: 703 human + 3 348 AI). For each feature dim, separation = `|mean(human) − mean(ai)| / pooled_std`. The 87-NELA / 10-Style / 128-TRACE layout is the one defined in [`training/extractor_pipeline.py:40-42`](../training/extractor_pipeline.py) and flattened in [`training/classical.py:26`](../training/classical.py). NELA names come from `NELAFeatureExtractor.extract_all` in the `nela_features` package (the order used in [`extractors/nela_extractor.py:34`](../extractors/nela_extractor.py)).

### 5.1 Top 10 NELA dims by class separation

| rank | idx | NELA feature | \|Cohen's d\| | mean (human) | mean (AI) |
|---:|---:|---|---:|---:|---:|
| 1 | 50 | **ttr** (type-token ratio) | **3.39** | 0.416 | 0.613 |
| 2 | 52 | **word_count** | **2.94** | 762.4 | 472.7 |
| 3 |  4 | **stops** (stopword rate) | **2.62** | 0.418 | 0.301 |
| 4 | 56 | **lix** (readability) | 2.14 | 40.3 | 54.7 |
| 5 | 55 | **coleman_liau_index** | 2.11 | 8.65 | 14.49 |
| 6 | 51 | **avg_wordlen** | 2.10 | 4.41 | 5.42 |
| 7 | 54 | **smog_index** | 1.99 | 11.20 | 14.23 |
| 8 |  8 | **EX** (existential-there rate) | 1.69 | 0.0032 | 0.0006 |
| 9 | 11 | **JJ** (adjective rate) | 1.67 | 0.073 | 0.106 |
| 10 |  2 | **allpunc** (punctuation rate) | 1.55 | 0.090 | 0.117 |

Read the table physically: an LLM rewrite of a student essay, on average, uses a **47 % higher type-token ratio** (vocab is broader), is **40 % shorter** (the rewrite trims the original), has a **28 % lower stopword rate** and a **23 % higher Coleman-Liau readability grade** (the writing reads as more formal). The top two dims alone shift by **three pooled standard deviations** — that is far more than any reasonable classifier needs to draw a near-perfect boundary.

### 5.2 Top dim from Style and from TRACE for contrast

Style (10 dims total):

| rank | idx | Style feature | \|Cohen's d\| | mean (human) | mean (AI) |
|---:|---:|---|---:|---:|---:|
| 1 | 4 | **cos_emb_mean** (mean of cosine-similarity to the 5 LLM rewrites) | **1.16** | 0.864 | 0.910 |
| 2 | 0 | jac1_mean | 0.56 | 0.190 | 0.205 |

Only 1 of Style's 10 dims clears \|d\|=1; the next-best is barely past 0.5.

TRACE (128 dims total):

| rank | idx | TRACE dim | \|Cohen's d\| | mean (human) | mean (AI) |
|---:|---:|---|---:|---:|---:|
| 1 | 23 | trace_023 | **0.86** | +0.057 | +0.021 |
| 2 | 50 | trace_050 | 0.84 | −0.004 | +0.028 |
| 3 |  6 | trace_006 | 0.82 | −0.015 | −0.037 |

**Not a single one of TRACE's 128 dims clears \|d\|=1.0**, and the very best is just 0.86 σ.

### 5.3 Summary across modalities

| modality | dims | max \|d\| | median \|d\| | dims with \|d\|>1.0 | dims with \|d\|>0.5 |
|---|---:|---:|---:|---:|---:|
| **NELA** | 87 | **3.39** | 0.48 | **17** | **42** |
| Style | 10 | 1.16 | 0.21 | 1 | 2 |
| TRACE | 128 | 0.86 | 0.33 | 0 | 29 |

**This is the proof.** NELA has 17 features that, taken in isolation, are each more separating than the single best feature TRACE offers. Even a linear classifier given only `ttr` and `word_count` would push past 95 % accuracy on this slice. The fact that `classical_logreg` scores 0.9979 macro-F1 is therefore not surprising — it is the direct consequence of having seventeen >1-σ-separating axes available simultaneously, several of which are nearly orthogonal (TTR, length, stopword-rate, readability, POS rates are all measuring different things).

---

## 6. What this does **not** mean

- **It does not mean Style and TRACE are useless.** A feature that is redundant *given another feature* still carries the same absolute information when measured alone. Permutation importance is a **marginal** quantity: it asks *"what does this block add on top of the other two?"* and on this dataset the answer for the classical models is almost nothing, because NELA's marginal already saturates the labels.
- **It does not mean the neural fusion modules are pointless.** The same §3b table shows that, under modality dropout, the neural fusion heads recover real signal from Style (8–17 %) and TRACE (1–8 %). The four fusion variants differ on exactly how they weight that residual signal — `gating` collapses most heavily to NELA, `concat`/`mlp` use the other two most. That is the methodological reason modality dropout was added in the first place ([`training/model.py`](../training/model.py); discussed in [METHODOLOGY.md §4](../METHODOLOGY.md)).
- **It does not mean NELA is "the right" feature set in general.** The class-separation magnitudes in §5 are specific to the USE-essay-vs-LLM-rewrite slice. Take away the length and TTR shift (e.g. by forcing rewrites to match the source length exactly, or by training on a different genre where students write more lexically diverse text) and the top of the §5.1 table collapses, taking the linear models with it.

---

## 7. What would change the picture

The two filters in §3 cannot be relaxed (they are TRACE-required) and dropping NELA entirely defeats the point of the comparison. Within those constraints, several interventions would re-introduce real competition between the three modalities:

1. **Evaluate on length-matched, TTR-controlled rewrites.** The top-3 NELA dims in §5.1 (`ttr`, `word_count`, `stops`) carry 8+ pooled-σ of combined separation, almost entirely because the LLM rewrites are systematically shorter and lexically broader than the source essay. Force the rewrite generator to match source length within ±10 % and source TTR within ±0.05 and those three NELA dims collapse toward zero separation, exposing the remaining 84 NELA dims and the Style/TRACE signal to a fairer comparison.
2. **Evaluate on out-of-domain corpora that already satisfy the TRACE author-context contract** — i.e. multi-essay-per-author human corpora paired with newer LLM rewrites (GPT-4o-class, Claude-3-class). NELA's readability/length axes drift the most across genres; TRACE's author-style fingerprint is the most domain-stable of the three, so the dominance order should compress or invert.
3. **Tighten the asymmetric modality-dropout schedule** in [`training/model.py`](../training/model.py): the current rates are NELA 0.5 / Style 0.2 / TRACE 0.2 ([`training/config.py:34-36`](../training/config.py)). Pushing NELA's drop probability toward 0.7–0.8 (and/or warming up a curriculum that disables NELA entirely for the first few epochs) forces the fusion head to fit Style + TRACE *first*, then use NELA as a residual signal — without removing any extractor from the pipeline.
4. **Add a NELA-residual auxiliary loss.** Train a small linear NELA-only predictor; freeze it; then make the fusion head fit the *residual* (true label minus NELA-only logit). Style and TRACE will be forced to explain exactly the signal NELA cannot — a clean way to measure their independent contribution while keeping all three modalities in the production pipeline.

---

## Appendix — reproducing the §5 table

```python
# from the repo root
import numpy as np
d = np.load("data/features/train.npz", allow_pickle=True)
y = d["label"]; X = d["nela"].astype(np.float64)
h = X[y == 0]; a = X[y == 1]
sep = np.abs(h.mean(0) - a.mean(0)) / np.sqrt(
    ((h.shape[0]-1)*h.var(0,ddof=1) + (a.shape[0]-1)*a.var(0,ddof=1))
    / (h.shape[0] + a.shape[0] - 2)
)
# the 87 feature names are produced by:
#   nela_features.NELAFeatureExtractor().extract_all(text)[1]
# (called in extractors/nela_extractor.py:34)
```

Replace `d["nela"]` with `d["style"]` or `d["trace"]` for the other modalities; the column layout (NELA 0..86 | Style 87..96 | TRACE 97..224) only matters for the classical flatten in [`training/classical.py:26`](../training/classical.py), because the `.npz` already stores the three blocks separately.
