"""
Augment the dataset with HC3 (Human ChatGPT Comparison Corpus).

HC3 is paired data: same question, one human answer + one ChatGPT answer.
Domains: reddit_eli5, finance, medicine, open_qa, wiki_csai.

This expands the dataset beyond essays + abstracts (the current bottleneck
where domain perfectly correlates with label) and adds a modern AI model
(ChatGPT) to balance the outdated ArguGPT models.

Inputs : current dataset_ready/ (after prepare_ready.py + finalize_dataset.py)
Outputs: same files, augmented
"""

import json
import random
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("/home/konrado/dataset_ready")
HC3_URL    = "https://huggingface.co/api/datasets/Hello-SimpleAI/HC3/parquet/all/train/0.parquet"

# Cap per (domain, label) to avoid HC3 dominating the corpus.
MAX_PER_DOMAIN = 600


def main() -> None:
    rng = random.Random(42)

    # 1. Load existing dataset (already filtered + split)
    def load(name):
        path = OUTPUT_DIR / name
        return [json.loads(l) for l in open(path, encoding="utf-8")]

    human    = load("human_texts.jsonl")
    ai       = load("ai_texts.jsonl")
    rew      = load("rewritten_texts.jsonl")

    # 2. Pull HC3 parquet
    print("Fetching HC3 parquet ...")
    df = pd.read_parquet(HC3_URL)
    print(f"  HC3 raw: {len(df)} pairs across {df['source'].nunique()} domains")

    # 3. Convert to our schema (one record per answer)
    next_human_id = max(int(r["id"][2:]) for r in human if r["id"].startswith("h_")) + 1
    next_ai_id    = max(int(r["id"][2:]) for r in ai    if r["id"].startswith("a_")) + 1

    def w(text):  # word count
        return len(text.split())

    new_human, new_ai = [], []
    per_group_count = {}
    rows = list(df.itertuples(index=False))
    rng.shuffle(rows)

    for row in rows:
        domain = row.source  # reddit_eli5 / finance / medicine / open_qa / wiki_csai
        question = row.question

        # First answer of each list (lists can have multiple, take first)
        h_list = list(row.human_answers)
        a_list = list(row.chatgpt_answers)
        if not h_list or not a_list:
            continue
        h_text = h_list[0].strip()
        a_text = a_list[0].strip()

        # length filter (matches finalize_dataset.py)
        if not (100 <= w(h_text) <= 1000 and 100 <= w(a_text) <= 1000):
            continue

        cap_key = ("human", domain)
        if per_group_count.get(cap_key, 0) >= MAX_PER_DOMAIN:
            continue
        per_group_count[cap_key] = per_group_count.get(cap_key, 0) + 1
        per_group_count[("ai", domain)] = per_group_count.get(("ai", domain), 0) + 1

        new_human.append({
            "id":                f"h_{next_human_id:04d}",
            "text":              h_text,
            "is_ai":             False,
            "label":             "human",
            "source":            "hc3",
            "domain":            domain,
            "author_id":         None,
            "model":             None,
            "exam_type":         None,
            "prompt":            question,
            "source_text_id":    None,
            "text_length_words": w(h_text),
            "split":             None,  # filled later
        })
        new_ai.append({
            "id":                f"a_{next_ai_id:04d}",
            "text":              a_text,
            "is_ai":             True,
            "label":             "ai",
            "source":            "hc3",
            "domain":            domain,
            "author_id":         None,
            "model":             "chatgpt",
            "exam_type":         None,
            "prompt":            question,
            "source_text_id":    None,
            "text_length_words": w(a_text),
            "split":             None,
        })
        next_human_id += 1
        next_ai_id    += 1

    print(f"  HC3 added: {len(new_human)} human / {len(new_ai)} AI (length-filtered + capped)")
    print(f"  HC3 distribution per domain:")
    for (lbl, dom), n in sorted(per_group_count.items()):
        print(f"    {lbl:5s} / {dom:14s}: {n}")

    # 4. Assign splits to the new records
    # HC3 has no author_id — split at record level, 70/15/15
    new_records = new_human + new_ai
    rng.shuffle(new_records)
    n = len(new_records)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    for i, r in enumerate(new_records):
        if i < n_train:
            r["split"] = "train"
        elif i < n_train + n_val:
            r["split"] = "val"
        else:
            r["split"] = "test"

    # 5. Merge with existing
    human_out = human + [r for r in new_records if not r["is_ai"]]
    ai_out    = ai    + [r for r in new_records if     r["is_ai"]]
    merged    = human_out + ai_out + rew

    def write(path, records):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write(OUTPUT_DIR / "human_texts.jsonl",     human_out)
    write(OUTPUT_DIR / "ai_texts.jsonl",        ai_out)
    write(OUTPUT_DIR / "merged.jsonl",          merged)

    for split in ("train", "val", "test"):
        write(
            OUTPUT_DIR / f"{split}.jsonl",
            [r for r in merged if r["split"] == split],
        )

    # 6. Stats
    from collections import Counter
    print()
    print("Final dataset:")
    print(f"  Total records: {len(merged)}")
    print(f"  Human: {len(human_out)}, AI: {len(ai_out)}, Rewrites: {len(rew)}")
    print(f"  By source:  {dict(Counter(r['source'] for r in merged))}")
    print(f"  By domain:  {dict(Counter(r['domain'] for r in merged))}")
    print(f"  By split:   {dict(Counter(r['split'] for r in merged))}")
    print()
    print("Per (split, label):")
    for split in ("train", "val", "test"):
        sub = [r for r in merged if r["split"] == split]
        c = Counter(r["label"] for r in sub)
        print(f"  {split}: human={c.get('human',0)}, ai={c.get('ai',0)}, total={len(sub)}")


if __name__ == "__main__":
    main()
