"""
Prepare the dataset for delivery to Dmytro.

Tasks:
1. Assign globally unique IDs (h_0001 / a_0001 / r_0001) to all records.
2. For each rewrite, recover which specific human text it was generated from
   (the original merge_final.py used rng.choice() per author but did not
   record the source). Recovery works because the rewrite is deterministic
   (text.replace(" is ", " was ").replace(" the ", " a ")) — for each
   rewrite we re-apply the transform to all of the author's human texts and
   pick the one that matches.

Inputs : /tmp/angepasst_check/final_dataset/*.jsonl
Outputs: /home/konrado/dataset_ready/{human,ai,rewritten,merged}.jsonl
"""

import json
from collections import Counter
from pathlib import Path

INPUT_DIR  = Path("/tmp/angepasst_check/final_dataset")
OUTPUT_DIR = Path("/home/konrado/dataset_ready")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def simulate_rewrite(text: str) -> str:
    return text.replace(" is ", " was ").replace(" the ", " a ")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    human = load_jsonl(INPUT_DIR / "human_texts.jsonl")
    ai    = load_jsonl(INPUT_DIR / "ai_texts.jsonl")
    rew   = load_jsonl(INPUT_DIR / "rewritten_texts.jsonl")

    # 1. IDs --------------------------------------------------------------
    for i, r in enumerate(human, start=1):
        r["id"] = f"h_{i:04d}"
    for i, r in enumerate(ai, start=1):
        r["id"] = f"a_{i:04d}"
    for i, r in enumerate(rew, start=1):
        r["id"] = f"r_{i:04d}"

    # 2. Backreference rewrite -> original human text --------------------
    human_by_author: dict[str, list[dict]] = {}
    for r in human:
        human_by_author.setdefault(r["author_id"], []).append(r)

    matched, unmatched = 0, 0
    for r in rew:
        candidates = human_by_author.get(r["author_id"], [])
        match_id: str | None = None
        for cand in candidates:
            if simulate_rewrite(cand["text"]) == r["text"]:
                match_id = cand["id"]
                break
        r["source_text_id"] = match_id
        if match_id:
            matched += 1
        else:
            unmatched += 1

    # 3. Merged file (consistent column order) ---------------------------
    merged = human + ai + rew

    # 4. Write -----------------------------------------------------------
    write_jsonl(OUTPUT_DIR / "human_texts.jsonl",     human)
    write_jsonl(OUTPUT_DIR / "ai_texts.jsonl",        ai)
    write_jsonl(OUTPUT_DIR / "rewritten_texts.jsonl", rew)
    write_jsonl(OUTPUT_DIR / "merged.jsonl",          merged)

    # 5. Sanity checks + stats ------------------------------------------
    all_ids = [r["id"] for r in merged]
    assert len(all_ids) == len(set(all_ids)), "duplicate IDs!"
    for r in merged:
        assert r["text"], f"empty text in {r['id']}"

    stats = {
        "human":           len(human),
        "ai":              len(ai),
        "rewrites":        len(rew),
        "merged":          len(merged),
        "rewrites_with_source_text_id":   matched,
        "rewrites_without_source_text_id": unmatched,
        "ai_by_source":    dict(Counter(r["source"] for r in ai)),
        "ai_by_model":     dict(Counter(r["model"]  for r in ai if r["model"])),
        "ai_by_domain":    dict(Counter(r["domain"] for r in ai)),
        "unique_authors":  len({r["author_id"] for r in human}),
    }
    (OUTPUT_DIR / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False)
    )

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
