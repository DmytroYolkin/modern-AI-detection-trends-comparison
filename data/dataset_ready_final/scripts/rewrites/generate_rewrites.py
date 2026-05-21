"""
generate_rewrites.py
Generate real LLM rewrites of human texts for StyleDecipher's style-stability features.

Replaces the placeholder rewrites in rewritten_texts.jsonl with calls to Gemini.

Resumable: writes one JSONL line per SUCCESSFUL rewrite, atomically (append + fsync).
On restart, reads the output file and skips any source-text id that already has
a rewrite. If a rewrite attempt fails or is incomplete (None / empty / error),
NOTHING is written for that text — the next run retries it from scratch.

Rate-limit aware: targets the free tier of gemini-3.1-flash-lite (~15 RPM, ~500 RPD).
- Paces calls (~5 s gap → ~12 RPM, safely under 15)
- On HTTP 429 / RESOURCE_EXHAUSTED: exponential backoff, then give up the text
- Hard daily budget — stops at DAILY_BUDGET successful rewrites and reports;
  resume the next day with the same command

Usage:
    export GEMINI_API_KEY=...
    pip install --user google-genai
    python3 generate_rewrites.py            # process all pending USE texts
    python3 generate_rewrites.py --limit 5  # quick test on 5 texts first

Quit cleanly with Ctrl+C — the in-flight text won't be partially written.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from google import genai


# --- Configuration -----------------------------------------------------------
INPUT_PATH    = Path("/home/konrado/dataset_ready/human_texts.jsonl")
OUTPUT_PATH   = Path("/home/konrado/dataset_ready/rewritten_texts_real.jsonl")
LOG_PATH      = Path("/home/konrado/dataset_ready/generate_rewrites.log")

SOURCE_FILTER = "use"               # only rewrite USE records (the ones with author_id, needed by TRACE)
MODEL         = "gemini-3.1-flash-lite"
SLEEP_BETWEEN = 5.0                 # seconds between calls (~12 RPM, safely below 15 RPM limit)
MAX_RETRIES   = 3                   # per-text retry count on transient errors
DAILY_BUDGET  = 480                 # stop after this many successful rewrites/day (safety margin under 500 RPD)
MIN_OUTPUT_W  = 30                  # reject suspiciously short rewrites (likely refusals like "I cannot...")
INSTRUCTION   = (
    "Rewrite the following text. Preserve all factual content and meaning, "
    "but rephrase it entirely in your own style — vary sentence structure, "
    "vocabulary, and phrasing. Keep approximately the same length. "
    "Do not add any preamble, explanation, or framing — output only the rewritten text."
)
PROMPT_TMPL   = INSTRUCTION + "\n\nTEXT:\n{text}"
# -----------------------------------------------------------------------------


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def append_jsonl_atomic(path: Path, record: dict) -> None:
    """Append one record. Flush + fsync so a kill -9 leaves a consistent file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def done_source_ids(path: Path) -> set[str]:
    """Source-text ids that already have a successful rewrite."""
    return {r.get("source_text_id") for r in load_jsonl(path) if r.get("source_text_id")}


def call_gemini(client, text: str) -> str | None:
    """Call Gemini. Return clean rewrite text on success, None on any failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=PROMPT_TMPL.format(text=text),
            )
            out = (resp.text or "").strip()
            if not out or len(out.split()) < MIN_OUTPUT_W:
                log(f"  rejected: response too short ({len(out.split())} words)")
                return None
            # Guard against the model parroting the prompt back
            if out.startswith("Rewrite the following"):
                log(f"  rejected: model echoed the prompt")
                return None
            return out
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                wait = 60 * attempt   # 60s, 120s, 180s
                log(f"  rate-limited (attempt {attempt}/{MAX_RETRIES}), sleeping {wait}s")
                time.sleep(wait)
                continue
            # Other transient error: backoff once and retry
            if attempt < MAX_RETRIES:
                log(f"  transient error: {msg[:160]} — retrying in 10s")
                time.sleep(10)
                continue
            log(f"  giving up after {MAX_RETRIES} attempts: {msg[:160]}")
            return None
    return None  # exhausted retries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N pending texts (for quick testing)")
    args = ap.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("ERROR: set GEMINI_API_KEY env var. Get one at https://aistudio.google.com/apikey")

    client = genai.Client()

    # Source pool
    sources = load_jsonl(INPUT_PATH)
    if SOURCE_FILTER:
        sources = [r for r in sources if r["source"] == SOURCE_FILTER]
    log(f"Source pool: {len(sources)} texts (filter: source=={SOURCE_FILTER!r})")

    # Resume: skip texts whose source id already has a rewrite
    done = done_source_ids(OUTPUT_PATH)
    pending = [r for r in sources if r["id"] not in done]
    log(f"Already rewritten: {len(done)}, pending: {len(pending)}")

    if args.limit is not None:
        pending = pending[: args.limit]
        log(f"--limit {args.limit} active, will process {len(pending)} this run")

    if not pending:
        log("Nothing to do — all source texts already have rewrites.")
        return

    # Continue rewrite-id numbering
    next_idx = 1 + max(
        (int(r["id"][2:]) for r in load_jsonl(OUTPUT_PATH)
         if r.get("id", "").startswith("r_")),
        default=0,
    )

    today_done = 0
    try:
        for i, src in enumerate(pending, 1):
            if today_done >= DAILY_BUDGET:
                log(f"Daily budget reached ({DAILY_BUDGET}). Stopping cleanly. Resume tomorrow.")
                break

            log(f"[{i}/{len(pending)}] {src['id']} ({len(src['text'].split())} words) ...")
            rewrite = call_gemini(client, src["text"])
            if rewrite is None:
                log(f"  SKIPPED — will be retried on next run")
                time.sleep(SLEEP_BETWEEN)
                continue

            record = {
                "id":                f"r_{next_idx:04d}",
                "text":              rewrite,
                "is_ai":             True,
                "label":             "ai",
                "source":            "use_rewrite",
                "domain":            src["domain"],
                "author_id":         src["author_id"],
                "model":             MODEL,
                "exam_type":         None,
                "prompt":            INSTRUCTION,
                "source_text_id":    src["id"],
                "text_length_words": len(rewrite.split()),
                "split":             src["split"],
            }
            append_jsonl_atomic(OUTPUT_PATH, record)
            next_idx += 1
            today_done += 1
            log(f"  OK  {record['id']}  ({today_done}/{DAILY_BUDGET} today, "
                f"{len(rewrite.split())} words)")

            time.sleep(SLEEP_BETWEEN)
    except KeyboardInterrupt:
        log("Interrupted by user. In-flight text was NOT written. Resume any time.")
        sys.exit(0)

    log(f"Run complete. Wrote {today_done} rewrites this session.")
    log(f"Output: {OUTPUT_PATH}")
    remaining = len(pending) - today_done
    if remaining > 0:
        log(f"{remaining} pending — re-run tomorrow to continue.")


if __name__ == "__main__":
    main()
