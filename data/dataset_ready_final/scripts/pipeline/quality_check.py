"""
quality_check.py — Scannt alle Rewrites auf Qualitäts-Probleme:
  1. Refusal-Patterns ("Sorry, I cannot...", "Here is the rewrite:", "As an AI...")
  2. Echo-Output (LLM hat den Input einfach wiederholt)
  3. Length-Outliers (Rewrite < 30 % oder > 200 % der Original-Länge)
  4. Near-Duplicates (Rewrite >95% identisch zum Original)
  5. Leere/zu kurze Rewrites (< 30 Wörter)

Schreibt bereinigte rewritten_texts_clean.jsonl, plus quality_report.txt.
"""

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

DIR = Path("/home/konrado/dataset_ready")

REFUSAL_PATTERNS = [
    r"^I (?:can'?t|cannot|am unable|won'?t)",
    r"^I'?m sorry,? (?:but )?I",
    r"^I apologi[sz]e,? (?:but )?I",
    r"^Sorry,? (?:but )?I",
    r"^As an AI",
    r"^As a language model",
    r"^Here'?s? (?:the|your|a) rewrit",
    r"^Sure,? here'?s",
    r"^Of course[,!]?\s+Here",
    r"^Certainly[,!]?\s+Here",
    r"^Rewrite:?\s",
    r"^Rewritten text:?\s",
    r"^\*\*Rewrite",
    r"^\(?Note:",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def near_duplicate_ratio(a: str, b: str) -> float:
    """Quick similarity check on first 1000 chars (full SequenceMatcher is O(n²))."""
    return SequenceMatcher(None, a[:1000], b[:1000]).ratio()


def main():
    rewrites = load_jsonl(DIR / "rewritten_texts.jsonl")
    humans   = load_jsonl(DIR / "human_texts.jsonl")
    src_lookup = {r["id"]: r for r in humans}

    print(f"Scanning {len(rewrites)} rewrites for quality issues ...\n")

    issues = {
        "refusal":         [],
        "echo":            [],
        "too_short":       [],
        "too_long":        [],
        "near_duplicate":  [],
        "missing_source":  [],
    }

    for r in rewrites:
        text = r["text"]
        src  = src_lookup.get(r.get("source_text_id"))

        # 1. Refusal-Pattern
        if REFUSAL_RE.search(text[:200]):
            issues["refusal"].append(r)
            continue

        # 2. Missing source (sollte nicht passieren)
        if not src:
            issues["missing_source"].append(r)
            continue

        # 3. Echo (LLM hat den input wiederholt)
        if text.strip() == src["text"].strip():
            issues["echo"].append(r)
            continue

        # 4. Length-Outliers
        orig_len = len(src["text"].split())
        new_len  = len(text.split())
        if new_len < 30 or new_len < 0.30 * orig_len:
            issues["too_short"].append(r)
            continue
        if new_len > 2.0 * orig_len:
            issues["too_long"].append(r)
            continue

        # 5. Near-Duplicate (rewrite ist >95% identisch zum Original)
        ratio = near_duplicate_ratio(text, src["text"])
        if ratio > 0.95:
            issues["near_duplicate"].append(r)
            continue

    # Report
    total_issues = sum(len(v) for v in issues.values())
    print(f"{'Issue':<20} {'Count':>6}  {'%':>6}")
    print("-" * 36)
    for k, v in issues.items():
        pct = 100 * len(v) / len(rewrites)
        print(f"{k:<20} {len(v):>6}  {pct:>5.2f}%")
    print("-" * 36)
    print(f"{'TOTAL issues':<20} {total_issues:>6}  {100*total_issues/len(rewrites):>5.2f}%")
    print(f"{'CLEAN':<20} {len(rewrites)-total_issues:>6}  {100*(len(rewrites)-total_issues)/len(rewrites):>5.2f}%")

    # Per provider breakdown
    print("\nIssues per provider:")
    print(f"{'Provider':<25} {'Refusal':>9} {'Echo':>6} {'Short':>7} {'Long':>6} {'NearDup':>9} {'Total':>7}")
    print("-" * 75)
    providers = set(r["model"] for r in rewrites)
    for p in sorted(providers):
        per_p = {k: sum(1 for r in v if r["model"] == p) for k, v in issues.items()}
        total_p = sum(per_p.values())
        n_provider = sum(1 for r in rewrites if r["model"] == p)
        print(f"{p:<25} {per_p['refusal']:>9} {per_p['echo']:>6} {per_p['too_short']:>7} "
              f"{per_p['too_long']:>6} {per_p['near_duplicate']:>9} "
              f"{total_p:>7}/{n_provider}")

    # Beispiele
    print("\n\nBeispiele für Refusals (max 3):")
    for r in issues["refusal"][:3]:
        print(f"  [{r['id']}] {r['model']}: '{r['text'][:120]}...'")
    print("\nBeispiele für Length too_short (max 3):")
    for r in issues["too_short"][:3]:
        src = src_lookup[r["source_text_id"]]
        print(f"  [{r['id']}] {r['model']}: orig {len(src['text'].split())} -> rewrite {r['text_length_words']} words")
    print("\nBeispiele für Length too_long (max 3):")
    for r in issues["too_long"][:3]:
        src = src_lookup[r["source_text_id"]]
        print(f"  [{r['id']}] {r['model']}: orig {len(src['text'].split())} -> rewrite {r['text_length_words']} words")
    print("\nBeispiele für Near-Duplicates (max 3):")
    for r in issues["near_duplicate"][:3]:
        print(f"  [{r['id']}] {r['model']}: similarity > 0.95")

    # Schreibe bereinigte Datei
    bad_ids = {r["id"] for v in issues.values() for r in v}
    clean = [r for r in rewrites if r["id"] not in bad_ids]
    out_path = DIR / "rewritten_texts_clean.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nClean rewrites geschrieben: {out_path} ({len(clean)} records)")


if __name__ == "__main__":
    main()
