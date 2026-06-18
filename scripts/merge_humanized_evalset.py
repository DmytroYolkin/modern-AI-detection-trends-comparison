"""Build the final humanized evaluation JSONL.

Concatenates:
  * the 1287 ``arxiv`` (human) rows from ``arxiv_merged.jsonl``
  * all rows from ``arxiv_humanized_adv.jsonl``  (Adv-P output)
  * all rows from ``arxiv_humanized_temp.jsonl`` (TempParaphraser output)

into ``data/testing_dataset/arxiv_final/arxiv_eval_with_humanizers.jsonl``.

Expected total = 1287 (human) + 1287 (Adv-P) + 1287 (Temp) = 3861.

Also writes ``arxiv_humanized_ai_only.jsonl`` -- the same Adv-P + Temp rows
*without* the 1287 human rows (2574 rows). The clean-arxiv baseline run
already scored those identical human texts, so the humanized baseline sweep
can skip them and reuse the clean human scores via
``test/evaluate_arxiv.py``'s merge logic. This saves ~33% of pod #2 wall time
on the slow baselines (detect_gpt @ 100 perturbations especially).

Prints a summary at the end:
  * total row count for both files
  * label distribution
  * per-source counts
  * per-humanizer counts
  * author count and min/max texts per author
  * asserts >= 2 texts per author (TRACE input-contract requirement)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
ARXIV_DIR = REPO_ROOT / "data" / "testing_dataset" / "arxiv_final"

DEFAULT_INPUTS = {
    "merged": ARXIV_DIR / "arxiv_merged.jsonl",
    "adv":    ARXIV_DIR / "arxiv_humanized_adv.jsonl",
    "temp":   ARXIV_DIR / "arxiv_humanized_temp.jsonl",
}
DEFAULT_OUTPUT = ARXIV_DIR / "arxiv_eval_with_humanizers.jsonl"
DEFAULT_AI_ONLY_OUTPUT = ARXIV_DIR / "arxiv_humanized_ai_only.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{ln}: invalid JSON ({e})")
    return rows


def _write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> None:
    """``tmp``-then-rename so a SIGKILL never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.jsonl")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merged", type=Path, default=DEFAULT_INPUTS["merged"])
    p.add_argument("--adv", type=Path, default=DEFAULT_INPUTS["adv"])
    p.add_argument("--temp", type=Path, default=DEFAULT_INPUTS["temp"])
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--output-ai-only", type=Path, default=DEFAULT_AI_ONLY_OUTPUT,
                   help="path for the humanizer-only JSONL (Adv-P + Temp, no humans). "
                        "Used by the humanized-baseline run to skip re-scoring the "
                        "1287 human texts that are byte-identical to the clean set.")
    p.add_argument("--allow-missing-humanizer", action="store_true",
                   help="if one humanizer output is missing, continue with the other")
    p.add_argument("--require-author-coverage", type=int, default=2,
                   help="assert each author has at least this many texts (default: 2)")
    args = p.parse_args()

    print(f"[merge] reading humans from   {args.merged}")
    merged = _read_jsonl(args.merged)
    humans = [r for r in merged if r.get("source") == "arxiv"]
    print(f"[merge]   {len(humans)} human rows")

    adv_rows: list[dict] = []
    temp_rows: list[dict] = []
    if args.adv.exists():
        adv_rows = _read_jsonl(args.adv)
        print(f"[merge] reading adv from      {args.adv}: {len(adv_rows)} rows")
    elif not args.allow_missing_humanizer:
        raise SystemExit(f"missing adv humanizer output: {args.adv} "
                         "(pass --allow-missing-humanizer to override)")

    if args.temp.exists():
        temp_rows = _read_jsonl(args.temp)
        print(f"[merge] reading temp from     {args.temp}: {len(temp_rows)} rows")
    elif not args.allow_missing_humanizer:
        raise SystemExit(f"missing temp humanizer output: {args.temp} "
                         "(pass --allow-missing-humanizer to override)")

    all_rows = humans + adv_rows + temp_rows
    _write_jsonl_atomic(args.output, all_rows)
    print(f"[merge] wrote {len(all_rows)} rows -> {args.output}")

    # AI-only file: same Adv-P + Temp rows, no humans. The humanized baseline
    # sweep uses this file; humans are reused from the clean run (byte-identical
    # texts -> byte-identical baseline scores) via test/evaluate_arxiv.py merge
    # logic.
    ai_only_rows = adv_rows + temp_rows
    _write_jsonl_atomic(args.output_ai_only, ai_only_rows)
    print(f"[merge] wrote {len(ai_only_rows)} rows -> {args.output_ai_only}")

    # --------------------------------------------------------- summary
    print("\n=== summary ===")
    print(f"total rows (full):    {len(all_rows)}    [{args.output.name}]")
    print(f"total rows (ai-only): {len(ai_only_rows)}    [{args.output_ai_only.name}]")

    label_counts = Counter(r.get("label") for r in all_rows)
    print(f"label distribution:   {dict(label_counts)}")

    source_counts = Counter(r.get("source") for r in all_rows)
    print("per-source counts:")
    for src, n in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:<28} {n}")

    humanizer_counts = Counter(r.get("humanizer") for r in all_rows if r.get("humanizer"))
    print("per-humanizer counts:")
    if humanizer_counts:
        for h, n in sorted(humanizer_counts.items(), key=lambda x: -x[1]):
            print(f"  {h:<28} {n}")
    else:
        print("  (none -- humanized outputs missing?)")

    by_author: dict[str, int] = defaultdict(int)
    for r in all_rows:
        a = r.get("author_id")
        if a:
            by_author[a] += 1
    if by_author:
        counts = list(by_author.values())
        print(f"authors:              {len(by_author)}")
        print(f"texts per author:     min={min(counts)} max={max(counts)} "
              f"mean={sum(counts)/len(counts):.2f}")
        if args.require_author_coverage > 0:
            offenders = [a for a, c in by_author.items()
                         if c < args.require_author_coverage]
            assert not offenders, (
                f"{len(offenders)} author(s) have < {args.require_author_coverage} "
                f"texts (first 5: {offenders[:5]})"
            )
            print(f"author-coverage check OK (>= {args.require_author_coverage} per author)")

    # Sanity-check the expected total when both humanizers ran.
    if adv_rows and temp_rows:
        expected = len(humans) + len(adv_rows) + len(temp_rows)
        if expected != len(all_rows):
            print(f"WARNING: total {len(all_rows)} != expected {expected}")


if __name__ == "__main__":
    main()
