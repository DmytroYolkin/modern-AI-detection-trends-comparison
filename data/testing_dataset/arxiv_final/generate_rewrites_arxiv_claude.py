"""
generate_rewrites_arxiv_claude.py — 1 Claude-Rewrite pro arXiv-Abstract.

- Längen-erzwingender Prompt (behebt NELA-Length-Confound)
- Spending-Tracker mit hartem Stop (Default $3.50, weit unter $4.61 Restguthaben)
- Resumable, atomic writes
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic

DIR          = Path("/root/arxiv_testset")
INPUT_PATH   = DIR / "arxiv_human.jsonl"
OUTPUT_PATH  = DIR / "arxiv_rewritten.jsonl"
LOG_PATH     = DIR / "generate_rewrites_arxiv.log"
COST_LOG     = DIR / "arxiv_cost.json"

MODEL         = "claude-haiku-4-5"
MAX_TOKENS    = 800       # Abstracts sind kurz, 800 reicht für längere Rewrites
SLEEP_BETWEEN = 1.0
MAX_RETRIES   = 3
HARD_BUDGET   = 3.50      # Stop bei $3.50 — Sicherheitsabstand zu $4.61
MIN_OUTPUT_W  = 30
PRICE_INPUT   = 1.0       # $/1M
PRICE_OUTPUT  = 5.0

INSTRUCTION = (
    "Rewrite the following text. Preserve all factual content and meaning, "
    "but rephrase it entirely in your own style — vary sentence structure, "
    "vocabulary, and phrasing. Keep approximately the same length. "
    "Do not add any preamble, explanation, or framing — output only the rewritten text."
)


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


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


def update_cost(c, it, ot):
    c["input_tokens"] += it
    c["output_tokens"] += ot
    c["calls"] += 1
    c["total_usd"] = round(
        c["input_tokens"] * PRICE_INPUT / 1e6 + c["output_tokens"] * PRICE_OUTPUT / 1e6, 4
    )
    save_cost(c)
    return c


def call_claude(client, text):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": INSTRUCTION + "\n\nTEXT:\n" + text}],
            )
            out = resp.content[0].text.strip() if resp.content else ""
            it, ot = resp.usage.input_tokens, resp.usage.output_tokens
            if not out or len(out.split()) < MIN_OUTPUT_W:
                return None, it, ot
            return out, it, ot
        except anthropic.AuthenticationError:
            return "AUTH_ERROR", 0, 0
        except anthropic.RateLimitError:
            wait = 30 * attempt
            log(f"  rate-limited, sleeping {wait}s")
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            msg = str(e)
            if "credit" in msg.lower() or "balance" in msg.lower():
                return "BUDGET_EXHAUSTED", 0, 0
            if "authentication" in msg.lower() or "401" in msg:
                return "AUTH_ERROR", 0, 0
            if attempt < MAX_RETRIES:
                time.sleep(10)
            else:
                log(f"  giving up: {msg[:150]}")
                return None, 0, 0
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(10)
            else:
                log(f"  giving up: {str(e)[:150]}")
                return None, 0, 0
    return None, 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: set ANTHROPIC_API_KEY")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Pre-Flight-Check: Key wirklich gültig? Sonst sofort raus statt 1287x scheitern.
    try:
        client.messages.create(
            model=MODEL, max_tokens=5,
            messages=[{"role": "user", "content": "ok"}],
        )
        log("Pre-flight key check: OK")
    except anthropic.AuthenticationError:
        log("FATAL: API key invalid/revoked. Aborting (no calls made).")
        sys.exit(1)
    except Exception as e:
        log(f"FATAL: pre-flight check failed: {str(e)[:150]}. Aborting.")
        sys.exit(1)

    cost = load_cost()
    log(f"Cost so far: ${cost['total_usd']} ({cost['calls']} calls)")
    if cost["total_usd"] >= HARD_BUDGET:
        log(f"HARD BUDGET ${HARD_BUDGET} reached. Stop.")
        return

    sources = load_jsonl(INPUT_PATH)
    done = done_ids(OUTPUT_PATH)
    pending = [r for r in sources if r["id"] not in done]
    log(f"Source: {len(sources)}, done: {len(done)}, pending: {len(pending)}")
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        log("Nothing to do.")
        return

    next_idx = 1 + max(
        (int(r["id"][4:]) for r in load_jsonl(OUTPUT_PATH) if r.get("id", "").startswith("axr_")),
        default=0,
    )

    session = 0
    ratios = []
    try:
        for i, src in enumerate(pending, 1):
            if cost["total_usd"] >= HARD_BUDGET:
                log(f"HARD BUDGET reached (${cost['total_usd']}). Stop cleanly.")
                break
            orig_w = len(src["text"].split())
            log(f"[{i}/{len(pending)}] {src['id']} ({orig_w}w) — ${cost['total_usd']}")
            rewrite, it, ot = call_claude(client, src["text"])
            if rewrite == "BUDGET_EXHAUSTED":
                log("Account empty. Stop.")
                break
            if rewrite == "AUTH_ERROR":
                log("AUTH ERROR — key invalid/revoked mid-run. Stopping immediately.")
                break
            if it or ot:
                cost = update_cost(cost, it, ot)
            if rewrite is None:
                time.sleep(SLEEP_BETWEEN)
                continue
            new_w = len(rewrite.split())
            ratios.append(new_w / orig_w)
            rec = {
                "id":                f"axr_{next_idx:04d}",
                "text":              rewrite,
                "is_ai":             True,
                "label":             "ai",
                "source":            "arxiv_rewrite",
                "domain":            "arxiv_cs",
                "author_id":         src["author_id"],
                "model":             MODEL,
                "exam_type":         None,
                "prompt":            INSTRUCTION,
                "source_text_id":    src["id"],
                "text_length_words": new_w,
                "split":             "test",
            }
            append_atomic(OUTPUT_PATH, rec)
            next_idx += 1
            session += 1
            log(f"  OK {rec['id']} ({new_w}w, ratio {new_w/orig_w:.2f}, ${cost['total_usd']})")
            time.sleep(SLEEP_BETWEEN)
    except KeyboardInterrupt:
        log("Interrupted.")
        sys.exit(0)

    if ratios:
        log(f"Avg length ratio (rewrite/orig): {sum(ratios)/len(ratios):.2f}")
    log(f"Session done. Wrote {session}. Total cost: ${cost['total_usd']}")


if __name__ == "__main__":
    main()
