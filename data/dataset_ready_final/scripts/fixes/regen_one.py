"""Re-generate the one bad Claude rewrite for h_0054."""
import json
import os
from pathlib import Path
import anthropic

ROOT = Path(__file__).resolve().parent.parent.parent  # dataset_ready_final/

src = next(json.loads(l) for l in open(ROOT / "human_texts.jsonl")
           if json.loads(l)["id"] == "h_0054")
print(f"Source h_0054: {len(src['text'].split())} words")

all_a = [json.loads(l) for l in open(ROOT / "intermediate" / "rewritten_texts_anthropic.jsonl")]
old = next(r for r in all_a if r["source_text_id"] == "h_0054")
print(f"Old rewrite {old['id']}: {old['text_length_words']} words (too short)")

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
INSTRUCTION = (
    "Rewrite the following text. Preserve all factual content and meaning, "
    "but rephrase it entirely in your own style — vary sentence structure, "
    "vocabulary, and phrasing. Keep approximately the same length. "
    "Do not add any preamble, explanation, or framing — output only the rewritten text."
)
resp = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=2000,
    messages=[{"role": "user", "content": INSTRUCTION + "\n\nTEXT:\n" + src["text"]}],
)
new_text = resp.content[0].text.strip()
print(f"New rewrite: {len(new_text.split())} words")
print(f"Tokens: input={resp.usage.input_tokens}, output={resp.usage.output_tokens}")

for r in all_a:
    if r["source_text_id"] == "h_0054":
        r["text"] = new_text
        r["text_length_words"] = len(new_text.split())
        break

with open(ROOT / "intermediate" / "rewritten_texts_anthropic.jsonl", "w") as f:
    for r in all_a:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("Done.")
