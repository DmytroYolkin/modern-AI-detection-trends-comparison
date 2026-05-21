"""
consolidate.py — vereint alle 4 echten LLM-Rewrites zu einem finalen
rewritten_texts.jsonl, ersetzt die alten Placeholder-Rewrites, und baut
merged.jsonl + train/val/test.jsonl neu.

Inputs:
  rewritten_texts_real.jsonl     (Gemini)
  rewritten_texts_mistral.jsonl  (Mistral Small)
  rewritten_texts_groq.jsonl     (Llama 3.1 8B via Groq)
  rewritten_texts_cohere.jsonl   (Command R+ via Cohere)
  human_texts.jsonl              (Source-Texte mit author_id + split)
  ai_texts.jsonl                 (Original AI-Texte)

Outputs (überschreibt):
  rewritten_texts.jsonl          (alle 4790 echten Rewrites, fortlaufende IDs r_0001…)
  merged.jsonl
  train.jsonl, val.jsonl, test.jsonl
  stats.json
"""

import json
from collections import Counter
from pathlib import Path

DIR = Path("/home/konrado/dataset_ready")

PROVIDER_FILES = [
    ("gemini",    "rewritten_texts_real.jsonl"),
    ("mistral",   "rewritten_texts_mistral.jsonl"),
    ("groq",      "rewritten_texts_groq.jsonl"),
    ("cohere",    "rewritten_texts_cohere.jsonl"),
    ("anthropic", "rewritten_texts_anthropic.jsonl"),
]


def load_jsonl(name):
    path = DIR / name
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def write_jsonl(name, records):
    path = DIR / name
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    # 1. Lade Source-Texte für split + author_id Lookup
    human = load_jsonl("human_texts.jsonl")
    ai    = load_jsonl("ai_texts.jsonl")
    src_lookup = {r["id"]: r for r in human}

    # 2. Lade alle Rewrites pro Provider, vergebe fortlaufende IDs
    all_rewrites = []
    per_provider = Counter()
    next_idx = 1
    for provider_name, file_name in PROVIDER_FILES:
        records = load_jsonl(file_name)
        print(f"  {provider_name}: {len(records)} records")
        for r in records:
            src_id = r.get("source_text_id")
            src = src_lookup.get(src_id)
            if not src:
                continue
            # Schema-vereinheitlichung — Source-Text gibt split + author_id authoritativ vor
            new = {
                "id":                f"r_{next_idx:04d}",
                "text":              r["text"],
                "is_ai":             True,
                "label":             "ai",
                "source":            "use_rewrite",
                "domain":            src["domain"],
                "author_id":         src["author_id"],
                "model":             r.get("model"),
                "exam_type":         None,
                "prompt":            r.get("prompt"),
                "source_text_id":    src_id,
                "text_length_words": len(r["text"].split()),
                "split":             src.get("split"),
            }
            all_rewrites.append(new)
            per_provider[provider_name] += 1
            next_idx += 1

    print(f"\nTotal rewrites: {len(all_rewrites)}")

    # 3. Sanity: jede Rewrite hat eindeutige ID und valides source_text_id
    ids = [r["id"] for r in all_rewrites]
    assert len(ids) == len(set(ids)), "duplicate IDs"
    for r in all_rewrites:
        assert r["text"], f"empty text in {r['id']}"
        assert r["source_text_id"] in src_lookup, f"unknown source: {r['source_text_id']}"

    # 4. Schreibe finale Files
    write_jsonl("rewritten_texts.jsonl", all_rewrites)

    merged = human + ai + all_rewrites
    write_jsonl("merged.jsonl", merged)

    for split in ("train", "val", "test"):
        write_jsonl(f"{split}.jsonl",
                    [r for r in merged if r.get("split") == split])

    # 5. Stats
    stats = {
        "human":          len(human),
        "ai_original":    len(ai),
        "rewrites_total": len(all_rewrites),
        "rewrites_per_provider": dict(per_provider),
        "rewrites_per_model":    dict(Counter(r["model"] for r in all_rewrites)),
        "merged_total":   len(merged),
        "by_split": {
            split: {
                "total":     sum(1 for r in merged if r.get("split") == split),
                "human":     sum(1 for r in merged if r.get("split") == split and r["label"] == "human"),
                "ai":        sum(1 for r in merged if r.get("split") == split and r["label"] == "ai"),
                "rewrites":  sum(1 for r in merged if r.get("split") == split and r["source"] == "use_rewrite"),
            }
            for split in ("train", "val", "test")
        },
        "unique_authors_with_rewrites": len({r["author_id"] for r in all_rewrites if r["author_id"]}),
        "rewrites_per_source_text": {
            "min": min(Counter(r["source_text_id"] for r in all_rewrites).values()),
            "max": max(Counter(r["source_text_id"] for r in all_rewrites).values()),
            "mean": round(len(all_rewrites) / len(set(r["source_text_id"] for r in all_rewrites)), 2),
        },
    }
    (DIR / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
