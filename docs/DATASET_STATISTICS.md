# Dataset statistics and transformations

A fully-sourced record of every dataset in this study and every transformation
applied — from the two raw source corpora (USE student essays and arXiv
abstracts), through AI-rewriting, cleaning, the author/TRACE filters, and the
train/val + external-arXiv split used by the paper.

Every number below is read directly from a committed artifact (`stats.json`,
`meta.json`, or the `.npz` caches themselves) and the source is cited inline.
This is the data-provenance companion to [../METHODOLOGY.md](../METHODOLOGY.md).

---

## 1. The two source corpora

| corpus | genre | role | provenance |
|---|---|---|---|
| **USE student essays** | student exam essays | training + in-distribution validation | Uppsala Student English Corpus (Axelsson 2000); each essay has a stable `author_id` |
| **arXiv CS abstracts** | academic paper abstracts | **external OOD test** | Zenodo 7404702 |

The study deliberately targets the *hard* case: AI text produced by an LLM
**rewriting** an existing human text, so topic/content are matched between
classes and only style differs ([../METHODOLOGY.md §1](../METHODOLOGY.md)).

---

## 2. The raw harmonised corpus (`dataset_ready_final/`)

Source: [`data/dataset_ready_final/stats.json`](../data/dataset_ready_final/stats.json),
[`data/README.md`](../data/README.md).

### 2.1 Composition (13,086 records)

| type | label | count | sources |
|---|---|---:|---|
| Human texts | `human` | 3,415 | USE essays + HC3 human answers |
| AI-original | `ai` | 3,620 | ArguGPT, RAID, HC3 ChatGPT answers |
| LLM rewrites | `ai` | 6,051 | USE essays rewritten by 5 cloud LLMs |
| **total** | | **13,086** | |

The HC3 / ArguGPT / RAID records are auxiliary AI-detection sources that ship in
the raw bundle but are removed by the §3 filters; the trained corpus is
USE-only.

### 2.2 Transformation — the rewrite step (manufacturing the `ai` label)

Each eligible USE human essay was rewritten **up to 5×**, once per cloud-API LLM,
to produce style-transferred AI text with matched content:

| provider | model | rewrites |
|---|---|---:|
| Google AI Studio | `gemini-3.1-flash-lite` | 1,270 |
| Mistral AI | `mistral-small-latest` | 1,272 |
| Groq (Meta LLaMA) | `llama-3.1-8b-instant` | 1,272 |
| Cohere | `command-r-plus-08-2024` | 965 |
| Anthropic | `claude-haiku-4-5` | 1,272 |
| **total** | | **6,051** |

- 430 unique authors have rewrites; mean **4.76** rewrites/source text (min 4, max 5).
- The Cohere shortfall (965) is a per-day API budget cap, not a filter.

> **Do not confuse rewriters with feature-extraction models.** These 5 cloud
> LLMs are the *label generators*. The 5 **Ollama** models (`llama3`, `mistral`,
> `gemma`, `phi3`, `qwen2`) named in [../METHODOLOGY.md §2.2](../METHODOLOGY.md)
> are used only *inside the StyleDecipher extractor* to score each text against
> self-paraphrases — they never create labels.

### 2.3 Cleaning applied at build time

The build pipeline ([data/dataset_ready_final/scripts/pipeline/](../data/dataset_ready_final/scripts/)):
length filter to **100–1000 words**, drop empty/degenerate/refusal/echo rewrites
(`quality_check.py`), assign stable IDs and backlink each rewrite to its
`source_text_id`. Two bad/missing rewrites were regenerated (`fixes/regen_*.py`).

### 2.4 Raw three-way split (raw JSONL)

Author-disjoint for USE, prompt-disjoint for HC3, seed 42:

| split | total | human | ai | rewrites |
|---|---:|---:|---:|---:|
| train | 9,097 | 2,390 | 6,707 | 4,185 |
| val | 1,952 | 501 | 1,451 | 899 |
| test | 2,037 | 524 | 1,513 | 967 |

---

## 3. Filtering to the study corpus (USE-only, 5,547 records)

Two stacked record-level filters are applied at feature-extraction time, both to
satisfy the TRACE author-fingerprint contract
([training/build_dataset.py:301,303](../training/build_dataset.py)):

1. **`--require-known-author`** — drop any record with no `author_id`.
2. **`--min-human-siblings 2`** — drop any record whose author contributes fewer
   than **2 other** human texts as TRACE context, *excluding* the source human
   of an LLM rewrite (no-source-leakage rule,
   [training/rebuild_trace_author.py::trace_human_siblings](../training/rebuild_trace_author.py)).

These are **TRACE input-contract requirements, not preferences** — relaxing
either collapses TRACE on the affected anchors ([../METHODOLOGY.md §1.3](../METHODOLOGY.md)).
TRACE also switches to `trace_context="author"` here, building each anchor's
embedding from the author's *other human texts*.

**Net effect:** every non-USE source (HC3, ArguGPT, RAID) is dropped because it
carries no per-author identity. The corpus contracts **13,086 → 5,547**, USE-only:

| split | records | human | ai | sources |
|---|---:|---:|---:|---|
| train | 4,051 | 703 | 3,348 | use 703, use_rewrite 3,348 |
| val | 667 | 116 | 551 | use 116, use_rewrite 551 |
| test | 829 | 144 | 685 | use 144, use_rewrite 685 |
| **total** | **5,547** | **963** | **4,584** | |

Class ratio ≈ **1 : 4.76** human:ai, intrinsic to the construction (1 human
essay → ~5 rewrites); training compensates with inverse-frequency `class_weight`.

---

## 4. The current in-distribution caches (`data/features/`)

Source: [`data/features/meta.json`](../data/features/meta.json),
[scripts/resplit_90_10.py](../scripts/resplit_90_10.py), direct `.npz` inspection.

Once arXiv was adopted as a **fully external** held-out test set, the local
`test` split became redundant. [scripts/resplit_90_10.py](../scripts/resplit_90_10.py)
pools the 5,547 USE-only records and re-derives an author-disjoint 90/10
train/val split (`GroupShuffleSplit(test_size=0.10, random_state=42)` grouped on
`author_id`; an assertion enforces disjoint author sets), then writes new
`train.npz` + `val.npz`.

| split | records | human | ai | unique authors | sources |
|---|---:|---:|---:|---:|---|
| `train.npz` | 4,958 | 861 | 4,097 | 387 | use 861, use_rewrite 4,097 |
| `val.npz` | 589 | 102 | 487 | 43 | use 102, use_rewrite 487 |
| **total** | **5,547** | **963** | **4,584** | 430 | |

No local `test.npz` exists — it is replaced by the arXiv sets (§5).

> **Caveat:** `val.npz` is also the calibration set (early-stopping, LR patience,
> strict-FPR ≤ 1 % threshold), so in-distribution numbers are an *upper bound*
> on held-out generalisation. The arXiv sets are the true held-out evaluation.

The paper's **controlled-set** numbers come from a complementary multi-seed
experiment: a fixed 70/15/15 author-disjoint split of the same 5,547 USE-only
cache, repeated over 5 seeds, all 14 models retrained per seed. Results (means ±
std, bootstrap CIs, McNemar tests) are in
[models/ready_models_resplit/results/](../models/ready_models_resplit/results/);
the analysis is in
[models/controlled_resplit_results.ipynb](../models/controlled_resplit_results.ipynb).

---

## 5. External test corpus — arXiv (`testing_dataset/arxiv_final/`)

Built entirely outside the training pipeline. Source:
[`arxiv_stats.json`](../data/testing_dataset/arxiv_final/arxiv_stats.json),
[`arxiv_summary.csv`](../data/testing_dataset/arxiv_summary.csv), `.npz` inspection.

### 5.1 Human pool

- 1,287 arXiv CS abstracts, **99** unique authors (5–34 texts/author).
- Origin: Zenodo 7404702 (arXiv CS abstracts, single-authored).
- Median ≈ 123–128 words; mean ≈ 138 words (per `arxiv_summary.csv`).

### 5.2 OOD clean set — `arxiv.npz` (n = 2,574)

| source | label | n |
|---|---|---:|
| `arxiv` | human | 1,287 |
| `arxiv_rewrite` | ai | 1,287 |

- Rewriter: `claude-haiku-4-5`, 1× per abstract, "keep approx. same length" prompt.
- Tests **domain shift** (exam essay → academic abstract). The rewriter is one of
  the 5 training rewriters, so this is *not* a generator-family shift — the
  dominant delta is genre/register/length.

### 5.3 OOD humanized set — `arxiv_humanized.npz` (n = 3,861)

The 1,287 Claude-haiku AI rewrites are each passed through **two humanizers**,
yielding a matched quadruple keyed by the same source abstract:

| source | label | n | tool |
|---|---|---:|---|
| `arxiv` (humans, reused) | human | 1,287 | — |
| `arxiv_humanized_adv` | ai | 1,287 | Adversarial Paraphrasing ([humanize_arxiv_adversarial.py](../scripts/humanize_arxiv_adversarial.py)) |
| `arxiv_humanized_temp` | ai | 1,287 | TempParaphraser ([humanize_arxiv_temppara.py](../scripts/humanize_arxiv_temppara.py)) |
| **total** | | **3,861** | 1,287 human + 2,574 humanized AI (1 : 2) |

Baselines are scored on the AI-only slice `arxiv_humanized_ai_only.jsonl` (2,574
rows) to avoid re-scoring the byte-identical humans; the evaluation harness
re-joins clean-human scores at report time.

---

## 6. End-to-end transformation chronology

```
data/dataset_ready_final/  ── raw, 13,086 records ──────────────────────────────
   USE human essays + HC3/ArguGPT/RAID + 6,051 LLM rewrites (5 cloud models)
   clean: 100–1000 words, drop refusals/echoes; author-disjoint 3-way split
        │
        │  extract NELA(87)+Style(10)+TRACE(128)
        │  + --require-known-author + --min-human-siblings 2 (trace_context="author")
        │  ⇒ drop HC3/ArguGPT/RAID (no author_id)
        ▼
USE-only filtered corpus  ── 5,547 records ─────────────────────────────────────
   train 4,051 / val 667 / test 829   (human:ai ≈ 1:4.76)
        │
        │  resplit_90_10.py: pool train+val+test, author-disjoint 90/10, seed 42
        │  drop local test (replaced by external arXiv)
        ▼
data/features/  ── current ─────────────────────────────────────────────────────
   train.npz 4,958 (861H/4,097AI, 387 authors)
   val.npz     589 (102H/ 487AI,  43 authors)        ← in-distribution
   arxiv.npz          2,574 (1287H/1287AI)           ← OOD clean
   arxiv_humanized.npz 3,861 (1287H/2574AI)          ← OOD humanized

External, built separately:
arXiv abstracts (Zenodo 7404702, 1,287 human, 99 authors)
   → claude-haiku rewrite ×1   → arxiv.npz
   → + Adv-P + TempParaphraser → arxiv_humanized.npz
```

---

## 7. Notes

- **Feature dimensionality is constant:** NELA 87 + Style 10 + TRACE 128 = 225-d;
  style coverage 1.0 across every split.
- **Reproduce any count** with:
  ```python
  import numpy as np, collections
  d = np.load("data/features/train.npz", allow_pickle=True)
  print(d["nela"].shape, collections.Counter(d["label"].tolist()),
        collections.Counter(map(str, d["sources"])))
  ```
