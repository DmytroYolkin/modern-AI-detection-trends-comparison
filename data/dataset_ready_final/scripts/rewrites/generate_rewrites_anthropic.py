"""
generate_rewrites_anthropic.py — Claude Haiku 4.5 Multi-LLM Rewrites
mit hartem Spending-Tracker (stoppt bei $9 verbrauchtem Budget, $10 Konto-Cap).

Resumable wie alle anderen Provider. Spending wird nach jedem Call neu
berechnet auf Basis der ECHTEN Token-Counts aus dem API-Response.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic


INPUT_PATH    = Path("/root/dataset_ready/human_texts.jsonl")
OUTPUT_PATH   = Path("/root/dataset_ready/rewritten_texts_anthropic.jsonl")
LOG_PATH      = Path("/root/dataset_ready/generate_rewrites_anthropic.log")
KEY_PATH      = Path("/root/.anthropic-key")
COST_LOG      = Path("/root/dataset_ready/anthropic_cost.json")

SOURCE_FILTER = "use"
MODEL         = "claude-haiku-4-5"
MAX_TOKENS    = 1200                       # Cap pro Rewrite
SLEEP_BETWEEN = 1.0                        # 60 RPM safe
MAX_RETRIES   = 3
HARD_BUDGET   = 9.0                        # Stopp bei $9 verbrauchtem Budget ($10 Konto-Cap)
MIN_OUTPUT_W  = 30

# Claude Haiku 4.5 Preise (pro 1M Tokens)
PRICE_INPUT   = 1.0
PRICE_OUTPUT  = 5.0

INSTRUCTION   = (
    "Rewrite the following text. Preserve all factual content and meaning, "
    "but rephrase it entirely in your own style — vary sentence structure, "
    "vocabulary, and phrasing. Keep approximately the same length. "
    "Do not add any preamble, explanation, or framing — output only the rewritten text."
)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_jsonl(p):
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def append_atomic(p, rec):
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def done_ids(p):
    return {r.get("source_text_id") for r in load_jsonl(p) if r.get("source_text_id")}


def load_cost():
    if COST_LOG.exists():
        return json.loads(COST_LOG.read_text())
    return {"input_tokens": 0, "output_tokens": 0, "calls": 0, "total_usd": 0.0}


def save_cost(c):
    COST_LOG.write_text(json.dumps(c, indent=2))


def update_cost(c, input_tokens, output_tokens):
    c["input_tokens"]  += input_tokens
    c["output_tokens"] += output_tokens
    c["calls"]         += 1
    c["total_usd"] = round(
        c["input_tokens"]  * PRICE_INPUT  / 1_000_000 +
        c["output_tokens"] * PRICE_OUTPUT / 1_000_000,
        4,
    )
    save_cost(c)
    return c


def call_anthropic(client, text):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": INSTRUCTION + "\n\nTEXT:\n" + text}],
            )
            out = resp.content[0].text.strip() if resp.content else ""
            usage = resp.usage
            input_tokens  = usage.input_tokens
            output_tokens = usage.output_tokens

            if not out or len(out.split()) < MIN_OUTPUT_W:
                log(f"  rejected: response too short ({len(out.split())} words)")
                return None, input_tokens, output_tokens
            if out.startswith("Rewrite the following"):
                log(f"  rejected: model echoed prompt")
                return None, input_tokens, output_tokens
            return out, input_tokens, output_tokens
        except anthropic.RateLimitError:
            wait = 60 * attempt
            log(f"  rate-limited (attempt {attempt}/{MAX_RETRIES}), sleeping {wait}s")
            time.sleep(wait)
            continue
        except anthropic.APIStatusError as e:
            msg = str(e)
            if "credit" in msg.lower() or "balance" in msg.lower():
                log(f"  ACCOUNT EMPTY: {msg[:200]}")
                return "BUDGET_EXHAUSTED", 0, 0
            if attempt < MAX_RETRIES:
                log(f"  API error: {msg[:160]} — retry in 10s")
                time.sleep(10)
                continue
            log(f"  giving up: {msg[:160]}")
            return None, 0, 0
        except Exception as e:
            if attempt < MAX_RETRIES:
                log(f"  error: {str(e)[:160]} — retry in 10s")
                time.sleep(10)
                continue
            log(f"  giving up: {str(e)[:160]}")
            return None, 0, 0
    return None, 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: set ANTHROPIC_API_KEY env var")

    client = anthropic.Anthropic(api_key=api_key)

    cost = load_cost()
    log(f"Cost so far: ${cost['total_usd']} ({cost['calls']} calls)")

    if cost["total_usd"] >= HARD_BUDGET:
        log(f"HARD BUDGET ${HARD_BUDGET} bereits erreicht. Stopping.")
        return

    sources = load_jsonl(INPUT_PATH)
    sources = [r for r in sources if r["source"] == SOURCE_FILTER]
    log(f"Source pool: {len(sources)} texts")

    done = done_ids(OUTPUT_PATH)
    pending = [r for r in sources if r["id"] not in done]
    log(f"Already rewritten (Anthropic): {len(done)}, pending: {len(pending)}")

    if args.limit is not None:
        pending = pending[: args.limit]
        log(f"--limit {args.limit} active")

    if not pending:
        log("Nothing to do.")
        return

    next_idx = 1 + max(
        (int(r["id"][3:]) for r in load_jsonl(OUTPUT_PATH)
         if r.get("id", "").startswith("ra_")),
        default=0,
    )

    session_done = 0
    try:
        for i, src in enumerate(pending, 1):
            if cost["total_usd"] >= HARD_BUDGET:
                log(f"HARD BUDGET ${HARD_BUDGET} reached (current ${cost['total_usd']}). Stopping cleanly.")
                break

            log(f"[{i}/{len(pending)}] {src['id']} ({len(src['text'].split())} words) — budget so far: ${cost['total_usd']}")
            rewrite, in_tok, out_tok = call_anthropic(client, src["text"])

            if rewrite == "BUDGET_EXHAUSTED":
                log("Account balance exhausted on Anthropic side. Stopping.")
                break

            if in_tok or out_tok:
                cost = update_cost(cost, in_tok, out_tok)

            if rewrite is None:
                log(f"  SKIPPED — retry next run")
                time.sleep(SLEEP_BETWEEN)
                continue

            rec = {
                "id":                f"ra_{next_idx:04d}",
                "text":              rewrite,
                "is_ai":             True,
                "label":             "ai",
                "source":            "use_rewrite",
                "domain":            src["domain"],
                "author_id":         src["author_id"],
                "model":             "claude-haiku-4-5",
                "exam_type":         None,
                "prompt":            INSTRUCTION,
                "source_text_id":    src["id"],
                "text_length_words": len(rewrite.split()),
                "split":             src["split"],
            }
            append_atomic(OUTPUT_PATH, rec)
            next_idx += 1
            session_done += 1
            log(f"  OK  {rec['id']}  ({len(rewrite.split())} words, "
                f"+{in_tok}/{out_tok} tok, total ${cost['total_usd']})")
            time.sleep(SLEEP_BETWEEN)
    except KeyboardInterrupt:
        log("Interrupted. Resume any time.")
        sys.exit(0)

    log(f"\nSession complete. Wrote {session_done} Anthropic rewrites.")
    log(f"Total cost: ${cost['total_usd']} ({cost['calls']} calls, "
        f"{cost['input_tokens']:,} input / {cost['output_tokens']:,} output tokens)")


if __name__ == "__main__":
    main()
