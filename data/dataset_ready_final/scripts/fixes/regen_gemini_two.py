"""Re-generate the 2 missing Gemini rewrites for h_0110 and h_0831."""
import json
import os
import sys
import time
from pathlib import Path

from google import genai

ROOT   = Path(__file__).resolve().parent.parent.parent  # dataset_ready_final/
OUTPUT = ROOT / "intermediate" / "rewritten_texts_real.jsonl"
INSTRUCTION = (
    "Rewrite the following text. Preserve all factual content and meaning, "
    "but rephrase it entirely in your own style — vary sentence structure, "
    "vocabulary, and phrasing. Keep approximately the same length. "
    "Do not add any preamble, explanation, or framing — output only the rewritten text."
)

client = genai.Client()
human = [json.loads(l) for l in open(ROOT / "human_texts.jsonl")]
src_lookup = {r["id"]: r for r in human}

# Pending finden
existing_done = {json.loads(l).get("source_text_id") for l in open(OUTPUT)}
pending_ids = [sid for sid in ("h_0110", "h_0831") if sid not in existing_done]
print(f"Pending: {pending_ids}")

# Höchste rg_-Index in der bestehenden Datei
existing = [json.loads(l) for l in open(OUTPUT)]
next_idx = 1 + max(
    (int(r["id"].split("_")[1]) for r in existing if r["id"].startswith("r_")),
    default=0,
)

for sid in pending_ids:
    src = src_lookup[sid]
    print(f"\n{sid}: {len(src['text'].split())} words ...")
    content = INSTRUCTION + "\n\nTEXT:\n" + src["text"]
    try:
        resp = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=content,
        )
        out = (resp.text or "").strip()
        if not out or len(out.split()) < 30:
            print(f"  REJECTED: {len(out.split())} words")
            continue
        print(f"  OK: {len(out.split())} words")

        rec = {
            "id":                f"r_{next_idx:04d}",
            "text":              out,
            "is_ai":             True,
            "label":             "ai",
            "source":            "use_rewrite",
            "domain":            src["domain"],
            "author_id":         src["author_id"],
            "model":             "gemini-3.1-flash-lite",
            "exam_type":         None,
            "prompt":            INSTRUCTION,
            "source_text_id":    sid,
            "text_length_words": len(out.split()),
            "split":             src["split"],
        }
        with open(OUTPUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        next_idx += 1
        time.sleep(2)
    except Exception as e:
        print(f"  ERROR: {str(e)[:200]}")

print(f"\nFinal count: {sum(1 for _ in open(OUTPUT))}")
