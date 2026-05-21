"""
generate_rewrites_mistral.py
Wie generate_rewrites.py, aber für Mistral. Eigene Output-Datei, eigenes Log,
eigene PID. Läuft parallel zum Gemini-Job auf dem Pi.

Resumable per source_text_id (anhand der Mistral-Output-Datei). Beide Jobs
unabhängig — sie können sich nicht gegenseitig stören.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


# --- Configuration -----------------------------------------------------------
INPUT_PATH    = Path("/root/dataset_ready/human_texts.jsonl")
OUTPUT_PATH   = Path("/root/dataset_ready/rewritten_texts_mistral.jsonl")
LOG_PATH      = Path("/root/dataset_ready/generate_rewrites_mistral.log")
KEY_PATH      = Path("/root/.mistral-key")

SOURCE_FILTER = "use"
MODEL         = "mistral-small-latest"
SLEEP_BETWEEN = 1.5                  # Mistral Free Tier ~1 RPS, 1.5s buffer
MAX_RETRIES   = 3
DAILY_BUDGET  = 1500                 # Mistral hat kein hartes RPD, wir setzen freiwillig 1500
MIN_OUTPUT_W  = 30
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


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def append_jsonl_atomic(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def done_source_ids(path: Path) -> set[str]:
    return {r.get("source_text_id") for r in load_jsonl(path) if r.get("source_text_id")}


def call_mistral(api_key: str, text: str) -> str | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={"model": MODEL,
                      "messages": [{"role": "user",
                                    "content": PROMPT_TMPL.format(text=text)}]},
                timeout=120,
            )
            if resp.status_code == 429:
                raise RuntimeError("429 rate limited")
            resp.raise_for_status()
            out = (resp.json()["choices"][0]["message"]["content"] or "").strip()
            if not out or len(out.split()) < MIN_OUTPUT_W:
                log(f"  rejected: response too short ({len(out.split())} words)")
                return None
            if out.startswith("Rewrite the following"):
                log(f"  rejected: model echoed the prompt")
                return None
            return out
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower() or "quota" in msg.lower():
                wait = 60 * attempt
                log(f"  rate-limited (attempt {attempt}/{MAX_RETRIES}), sleeping {wait}s")
                time.sleep(wait)
                continue
            if attempt < MAX_RETRIES:
                log(f"  transient error: {msg[:160]} — retrying in 10s")
                time.sleep(10)
                continue
            log(f"  giving up after {MAX_RETRIES} attempts: {msg[:160]}")
            return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        sys.exit("ERROR: set MISTRAL_API_KEY env var")

    sources = load_jsonl(INPUT_PATH)
    if SOURCE_FILTER:
        sources = [r for r in sources if r["source"] == SOURCE_FILTER]
    log(f"Source pool: {len(sources)} texts (filter: source=={SOURCE_FILTER!r})")

    done = done_source_ids(OUTPUT_PATH)
    pending = [r for r in sources if r["id"] not in done]
    log(f"Already rewritten (Mistral): {len(done)}, pending: {len(pending)}")

    if args.limit is not None:
        pending = pending[: args.limit]
        log(f"--limit {args.limit} active, will process {len(pending)} this run")

    if not pending:
        log("Nothing to do — all source texts already have Mistral rewrites.")
        return

    next_idx = 1 + max(
        (int(r["id"][3:]) for r in load_jsonl(OUTPUT_PATH)
         if r.get("id", "").startswith("rm_")),
        default=0,
    )

    today_done = 0
    try:
        for i, src in enumerate(pending, 1):
            if today_done >= DAILY_BUDGET:
                log(f"Daily budget reached ({DAILY_BUDGET}). Stopping cleanly.")
                break

            log(f"[{i}/{len(pending)}] {src['id']} ({len(src['text'].split())} words) ...")
            rewrite = call_mistral(api_key, src["text"])
            if rewrite is None:
                log(f"  SKIPPED — will be retried on next run")
                time.sleep(SLEEP_BETWEEN)
                continue

            record = {
                "id":                f"rm_{next_idx:04d}",
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
        log("Interrupted by user. Resume any time.")
        sys.exit(0)

    log(f"Run complete. Wrote {today_done} Mistral rewrites this session.")


if __name__ == "__main__":
    main()
