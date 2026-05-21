"""
generate_rewrites_cohere.py — wie die anderen, aber mit Cohere v2 chat API.
Cohere Trial Free Tier: ~1000 calls/Monat — wir setzen DAILY_BUDGET = 200.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


INPUT_PATH    = Path("/root/dataset_ready/human_texts.jsonl")
OUTPUT_PATH   = Path("/root/dataset_ready/rewritten_texts_cohere.jsonl")
LOG_PATH      = Path("/root/dataset_ready/generate_rewrites_cohere.log")
KEY_PATH      = Path("/root/.cohere-key")

SOURCE_FILTER = "use"
MODEL         = "command-r-plus-08-2024"
SLEEP_BETWEEN = 4.0
MAX_RETRIES   = 3
DAILY_BUDGET  = 200
MIN_OUTPUT_W  = 30
INSTRUCTION   = (
    "Rewrite the following text. Preserve all factual content and meaning, "
    "but rephrase it entirely in your own style — vary sentence structure, "
    "vocabulary, and phrasing. Keep approximately the same length. "
    "Do not add any preamble, explanation, or framing — output only the rewritten text."
)
PROMPT_TMPL   = INSTRUCTION + "\n\nTEXT:\n{text}"


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


def call_cohere(api_key: str, text: str) -> str | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(
                "https://api.cohere.com/v2/chat",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={"model": MODEL,
                      "messages": [{"role": "user",
                                    "content": PROMPT_TMPL.format(text=text)}]},
                timeout=120,
            )
            if r.status_code == 429:
                raise RuntimeError("429 rate limited")
            r.raise_for_status()
            data = r.json()
            content = data.get("message", {}).get("content", [])
            out = ""
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    out += c.get("text", "")
            out = out.strip()
            if not out or len(out.split()) < MIN_OUTPUT_W:
                log(f"  rejected: response too short ({len(out.split())} words)")
                return None
            if out.startswith("Rewrite the following"):
                log(f"  rejected: model echoed prompt")
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

    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        sys.exit("ERROR: set COHERE_API_KEY env var")

    sources = load_jsonl(INPUT_PATH)
    sources = [r for r in sources if r["source"] == SOURCE_FILTER]
    log(f"Source pool: {len(sources)} texts")

    done = done_source_ids(OUTPUT_PATH)
    pending = [r for r in sources if r["id"] not in done]
    log(f"Already rewritten (Cohere): {len(done)}, pending: {len(pending)}")

    if args.limit is not None:
        pending = pending[: args.limit]
        log(f"--limit {args.limit} active")

    if not pending:
        log("Nothing to do.")
        return

    next_idx = 1 + max(
        (int(r["id"][3:]) for r in load_jsonl(OUTPUT_PATH)
         if r.get("id", "").startswith("rh_")),
        default=0,
    )

    today_done = 0
    try:
        for i, src in enumerate(pending, 1):
            if today_done >= DAILY_BUDGET:
                log(f"Daily budget reached ({DAILY_BUDGET}). Stopping.")
                break
            log(f"[{i}/{len(pending)}] {src['id']} ({len(src['text'].split())} words) ...")
            rewrite = call_cohere(api_key, src["text"])
            if rewrite is None:
                log("  SKIPPED — retry next run")
                time.sleep(SLEEP_BETWEEN)
                continue
            rec = {
                "id":                f"rh_{next_idx:04d}",
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
            append_jsonl_atomic(OUTPUT_PATH, rec)
            next_idx += 1
            today_done += 1
            log(f"  OK  {rec['id']}  ({today_done}/{DAILY_BUDGET} today, {len(rewrite.split())} words)")
            time.sleep(SLEEP_BETWEEN)
    except KeyboardInterrupt:
        log("Interrupted. Resume any time.")
        sys.exit(0)

    log(f"Run complete. Wrote {today_done} Cohere rewrites this session.")


if __name__ == "__main__":
    main()
