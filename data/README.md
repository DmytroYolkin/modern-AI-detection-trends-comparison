# Dataset: Human vs. AI Text Detection

This folder contains the finalized dataset used for training and evaluating the AI text detection pipeline. All ready-to-use files are in `dataset_ready_final/`.

---

## Overview

The **study corpus used by the paper is USE-only**: Uppsala Student English
Corpus (USE) essays as the `human` class, and paraphrases of those same essays
by five cloud-API LLMs as the `ai` class. After the TRACE author/sibling filters
(see [METHODOLOGY.md §1.3](../METHODOLOGY.md) and
[docs/DATASET_STATISTICS.md](../docs/DATASET_STATISTICS.md)) the trained corpus is:

| Class | Count |
|---|---|
| Human (USE essays) | 963 |
| AI (LLM rewrites of those essays) | 4,584 |

The arXiv abstract set is held out as the external **out-of-distribution test**
(see [`testing_dataset/`](testing_dataset/)); it is not part of this training
corpus.

### Raw bundle on disk

The `.jsonl` files in `dataset_ready_final/` are a **superset** of the study
corpus — they also carry auxiliary AI-detection sources (HC3, ArguGPT, RAID)
that were explored during construction but **excluded from every trained model**,
because they have no per-author identity and are dropped by the TRACE filters.
Full composition of the raw bundle:

| Type | Label | Count | Description |
|---|---|---|---|
| Human texts | `human` | 3,415 | USE essays + HC3 human answers |
| AI-original texts | `ai` | 3,620 | Directly AI-generated (ArguGPT, RAID, HC3 ChatGPT answers) — auxiliary, excluded |
| LLM rewrites | `ai` | 6,051 | USE essays rewritten by 5 LLMs (the study's AI class) |
| **Total** | | **13,086** | |

The raw bundle's author-disjoint train/val/test split:

| Split | Total | Human | AI |
|---|---|---|---|
| `train` | 9,097 | 2,390 | 6,707 |
| `val` | 1,952 | 501 | 1,451 |
| `test` | 2,037 | 524 | 1,513 |

The split is **author-disjoint**: all texts by the same human author are kept in a single split, preventing the contrastive author-fingerprint model (TRACE) from leaking author identity across train/val/test.

---

## Data Sources

- **USE corpus** — Uppsala Student English Corpus (Axelsson 2000). Human essays
  with author IDs; the study's `human` class and the source for the LLM rewrites.
- **LLM Rewrites** — Each USE human text was rewritten by up to 5 cloud-API
  models to produce style-transferred AI text while preserving factual content
  (the study's `ai` class):

| Provider | Model | Rewrites |
|---|---|---|
| Gemini | `gemini-3.1-flash-lite` | 1,270 |
| Mistral | `mistral-small-latest` | 1,272 |
| Groq (Llama) | `llama-3.1-8b-instant` | 1,272 |
| Cohere | `command-r-plus-08-2024` | 965 |
| Anthropic | `claude-haiku-4-5` | 1,272 |

On average each source text has **4.76 rewrites** (min 4, max 5), covering 430 unique authors.

- *Auxiliary (excluded from the study):* **ArguGPT / RAID** (AI-generated
  essays) and **HC3** (Hello-SimpleAI/HC3 — paired human + ChatGPT answers
  across `reddit_eli5`, `finance`, `medicine`, `open_qa`, `wiki_csai`) ship in
  the raw bundle but are dropped by the TRACE author/sibling filters, so no
  trained model in the paper sees them.

---

## Folder Structure

```
dataset_ready_final/
├── human_texts.jsonl          # Human-authored records only
├── ai_texts.jsonl             # AI-original records only
├── rewritten_texts.jsonl      # All LLM rewrites (consolidated, all providers)
├── merged.jsonl               # All records combined (human + AI + rewrites)
├── train.jsonl                # Training split
├── val.jsonl                  # Validation split
├── test.jsonl                 # Test split
├── stats.json                 # Counts and distribution statistics
│
├── intermediate/              # Raw per-provider rewrite outputs (pre-consolidation)
│   ├── rewritten_texts_real.jsonl       # Gemini
│   ├── rewritten_texts_mistral.jsonl    # Mistral
│   ├── rewritten_texts_groq.jsonl       # Groq / Llama
│   ├── rewritten_texts_cohere.jsonl     # Cohere
│   ├── rewritten_texts_anthropic.jsonl  # Claude Haiku
│   └── rewritten_texts_clean.jsonl      # Quality-filtered subset (no refusals/echoes)
│
├── scripts/
│   ├── pipeline/              # Main build pipeline (run in order below)
│   │   ├── prepare_ready.py       # 1. Assign IDs, backlink rewrites → source texts
│   │   ├── finalize_dataset.py    # 2. Length filter (100–1000 words), author-disjoint split
│   │   ├── add_hc3.py             # 3. Augment with HC3 paired corpus
│   │   ├── consolidate.py         # 4. Merge all provider rewrites into rewritten_texts.jsonl
│   │   └── quality_check.py       # 5. Scan rewrites for refusals, echoes, length outliers
│   ├── rewrites/              # Per-provider LLM rewrite generators
│   │   ├── generate_rewrites.py            # Gemini (free tier, 480 RPD budget)
│   │   ├── generate_rewrites_mistral.py    # Mistral (free tier)
│   │   ├── generate_rewrites_anthropic.py  # Claude Haiku (hard $9 spending cap)
│   │   └── generate_rewrites_cohere.py     # Cohere Command R+ (200/day budget)
│   └── fixes/                 # One-off patch scripts (already applied)
│       ├── regen_one.py           # Re-generated one bad Claude rewrite (h_0054)
│       └── regen_gemini_two.py    # Re-generated 2 missing Gemini rewrites (h_0110, h_0831)
│
└── logs/
    └── generate_rewrites.log  # Gemini rewrite run log
```

---

## Record Schema

Every record in every `.jsonl` file uses the same fields:

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique record ID — prefix encodes type: `h_` human, `a_` AI-original, `r_` rewrite |
| `text` | `str` | The text content |
| `is_ai` | `bool` | `True` for AI-generated or rewritten text |
| `label` | `str` | `"human"` or `"ai"` (redundant with `is_ai`, kept for readability) |
| `source` | `str` | Origin: `"use"` (human essay), `"argugpt"`, `"raid"`, `"hc3"`, `"use_rewrite"` (LLM rewrite) |
| `domain` | `str` | Text domain (e.g. essay topic, HC3 domain like `"finance"`) |
| `author_id` | `str\|null` | Author identifier for USE texts; `null` for AI-original and HC3 records |
| `model` | `str\|null` | Generating model name for AI/rewrite records; `null` for human texts |
| `exam_type` | `str\|null` | Exam type for USE essays; `null` otherwise |
| `prompt` | `str\|null` | Rewrite instruction or HC3 question; `null` for direct human/AI texts |
| `source_text_id` | `str\|null` | For rewrites: the `id` of the original human text; `null` otherwise |
| `text_length_words` | `int` | Word count (all records filtered to 100–1000 words) |
| `split` | `str` | `"train"`, `"val"`, or `"test"` |

---

## Loading the Data

```python
import json
from pathlib import Path

BASE = Path("data/dataset_ready_final")

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# Recommended entry points
train = load_jsonl(BASE / "train.jsonl")
val   = load_jsonl(BASE / "val.jsonl")
test  = load_jsonl(BASE / "test.jsonl")

# Or load everything at once
merged = load_jsonl(BASE / "merged.jsonl")

# Filter by type
human_train    = [r for r in train if r["label"] == "human"]
ai_train       = [r for r in train if r["label"] == "ai"]
rewrites_train = [r for r in train if r["source"] == "use_rewrite"]
```

For most use cases, **use `train.jsonl` / `val.jsonl` / `test.jsonl` directly** — they are subsets of `merged.jsonl` and already carry the `split` field.

---

## Preprocessing API (`preprocessing/`)

The `preprocessing/` package wraps the raw `.jsonl` files in classifier-ready
objects. Feature extractors (NELA, StyleDecipher) consume a single `.text`;
TRACE additionally needs other texts by the **same author**, which this layer
provides through a unified `author_id`.

**Unified author identity** — every record exposes one `author_id` usable for
"give me other texts by the same author", whether the text is human or AI:

| Record type | `author_id` resolves to | `generator` |
|---|---|---|
| USE human essay | the human author (`"0219"`) | `None` |
| LLM rewrite | the human author it was rewritten from (inherited) | the rewriting model |
| AI-original (ArguGPT/RAID/HC3) | the generating model (`"chatgpt"`) | the model |
| HC3 human answer | its own record id — authorless singleton | `None` |

This keeps both axes queryable: `texts_by_author()` for the human-author axis,
`texts_by_generator()` for the "which AI wrote it" axis.

```python
from data.preprocessing import Dataset

train = Dataset.load("train")          # "train" | "val" | "test" | None (=merged)

for sample in train:
    x        = sample.text                       # NELA / StyleDecipher input
    siblings = train.author_siblings(sample)     # TRACE: same-author context

# Group / look up
train.texts_by_author("0219")                    # human author + their rewrites
train.texts_by_generator("claude-haiku-4-5")     # everything one model produced
train.filter(label="ai", source="use_rewrite")   # subset -> new Dataset
```

Run `python data/preprocessing/dataset.py` to print a per-split summary.

> Author-disjoint splits keep every author entirely within one split, so
> sibling lookups never leak across train/val/test.

---

## Reproducing the Dataset

The pipeline was run on a remote Linux machine. To rebuild from scratch:

```bash
# 1. Prepare source files and assign IDs
python scripts/pipeline/prepare_ready.py

# 2. Run LLM rewrite generators in parallel (each is resumable)
GEMINI_API_KEY=...   python scripts/rewrites/generate_rewrites.py
MISTRAL_API_KEY=...  python scripts/rewrites/generate_rewrites_mistral.py
ANTHROPIC_API_KEY=.. python scripts/rewrites/generate_rewrites_anthropic.py
COHERE_API_KEY=...   python scripts/rewrites/generate_rewrites_cohere.py

# 3. Consolidate all provider rewrites into one file
python scripts/pipeline/consolidate.py

# 4. Apply length filter and author-disjoint train/val/test split
python scripts/pipeline/finalize_dataset.py

# 5. Augment with HC3 paired corpus
python scripts/pipeline/add_hc3.py

# 6. (Optional) Check rewrite quality
python scripts/pipeline/quality_check.py
```

All rewrite generators are **resumable**: they skip records whose `source_text_id` already appears in the output file, so they can be interrupted and restarted without duplicating work.
