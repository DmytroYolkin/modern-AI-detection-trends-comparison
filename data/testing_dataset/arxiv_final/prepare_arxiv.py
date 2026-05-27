"""
prepare_arxiv.py — arXiv CS abstracts (Zenodo 7404702) → Schema-konformes
Human-Test-Set für AI-Detection. Separates Test-Set (Dmytro: train auf altem
Dataset, test auf diesem).

Output: arxiv_human.jsonl  (Schema identisch zum Haupt-Dataset)
"""
import csv
import json
from collections import Counter
from pathlib import Path

DIR = Path("/home/konrado/arxiv_testset")
MIN_WORDS, MAX_WORDS = 50, 1000   # Abstracts sind kürzer als Essays


def main():
    with open(DIR / "arxiv_authors.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    author_to_id = {}
    human = []
    idx = 1
    for r in rows:
        text = r["abstract"].strip()
        wc = len(text.split())
        if wc < MIN_WORDS or wc > MAX_WORDS:
            continue
        name = r["author"]
        if name not in author_to_id:
            author_to_id[name] = f"ax_{len(author_to_id)+1:03d}"
        human.append({
            "id":                f"axh_{idx:04d}",
            "text":              text,
            "is_ai":             False,
            "label":             "human",
            "source":            "arxiv",
            "domain":            "arxiv_cs",
            "author_id":         author_to_id[name],
            "model":             None,
            "exam_type":         None,
            "prompt":            None,
            "source_text_id":    None,
            "text_length_words": wc,
            "split":             "test",
        })
        idx += 1

    with open(DIR / "arxiv_human.jsonl", "w", encoding="utf-8") as f:
        for r in human:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Autor-Mapping separat speichern (Name ↔ ID), falls später gebraucht
    (DIR / "arxiv_author_map.json").write_text(
        json.dumps(author_to_id, indent=2, ensure_ascii=False)
    )

    per_author = Counter(h["author_id"] for h in human)
    print(f"arxiv_human.jsonl: {len(human)} Texte, {len(per_author)} Autoren")
    print(f"Texte/Autor: min={min(per_author.values())}, "
          f"median={sorted(per_author.values())[len(per_author)//2]}, "
          f"max={max(per_author.values())}")
    print(f"Alle >=2 Texte: {all(c >= 2 for c in per_author.values())}")


if __name__ == "__main__":
    main()
