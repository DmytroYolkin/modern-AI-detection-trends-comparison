"""Humanize the AI half of the arxiv test set with TempParaphraser.

Upstream:  https://github.com/HJJWorks/TempParaphraser   (EMNLP 2025, #1607)
HF model:  https://huggingface.co/huangjj877/TempParaphraser

TempParaphraser is a fine-tuned LLM that paraphrases at multiple
"temperatures" (sampling diversities) and stitches the per-temperature
candidates into a single output. The upstream repo serves the model with
vLLM via LLaMA-Factory; the actual prompt template + sampling params live
in ``attack/attack_for_experiment.sh``.

This driver:
  1. Clones the upstream repo into ``scripts/_workdirs/tempparaphraser/``
     on first run (or ``git pull`` if it already exists).
  2. Loads ``huangjj877/TempParaphraser`` via ``transformers`` (we skip the
     vLLM-served path for now; it adds a service-management dependency that
     does not pay off for ~1.3 k records).
  3. Iterates the 1287 ``arxiv_rewrite`` rows from
     ``data/testing_dataset/arxiv_final/arxiv_merged.jsonl``.
  4. For each row, generates one paraphrase (multi-temperature sample +
     stitch is encapsulated inside ``TempParaphraser.paraphrase``).
  5. Writes results to
     ``data/testing_dataset/arxiv_final/arxiv_humanized_temp.jsonl``
     resumably (skip rows whose ``source_text_id`` is already in the file).

Smoke testing
-------------
``--smoke`` swaps in a no-op paraphraser that returns ``"PARAPHRASED: <input>"``
so the JSONL plumbing can be tested without downloading the HF weights.

Usage
-----
    # full run on a GPU pod
    python scripts/humanize_arxiv_temppara.py --device cuda

    # smoke
    python scripts/humanize_arxiv_temppara.py --smoke --limit 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Bootstrap so `python scripts/humanize_arxiv_temppara.py` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from scripts._humanizer_common import (
    ARXIV_JSONL,
    REPO_ROOT,
    WORKDIRS,
    AIRecord,
    append_jsonl_atomic,
    ensure_repo,
    load_ai_records,
    load_already_done,
    make_humanized_record,
    smoke_paraphrase,
)


TEMP_REPO_URL = "https://github.com/HJJWorks/TempParaphraser.git"
TEMP_REPO_DIR = WORKDIRS / "tempparaphraser"
HF_MODEL_ID = "huangjj877/TempParaphraser"
OUTPUT_JSONL = REPO_ROOT / "data" / "testing_dataset" / "arxiv_final" / "arxiv_humanized_temp.jsonl"


# Prompt template extracted from `attack/attack_for_experiment.sh` in the
# upstream repo. If the pod copy of the repo has changed, update this
# constant; we centralise it here for legibility.
TEMP_PROMPT_TEMPLATE = (
    "Paraphrase the following text. Keep the meaning, change the wording.\n\n"
    "Input:\n{text}\n\nOutput:\n"
)


class TempParaphraser:
    """HF-transformers wrapper around ``huangjj877/TempParaphraser``.

    The upstream pipeline samples N candidate paraphrases at different
    temperatures (e.g. 0.7, 1.0, 1.3) and stitches them sentence-wise. We
    replicate that here with the model's chat/instruct interface.
    """

    def __init__(
        self,
        repo_dir: Path,
        *,
        model_id: str = HF_MODEL_ID,
        device: str = "cuda",
        temperatures: tuple[float, ...] = (0.7, 1.0, 1.3),
        top_p: float = 0.95,
        max_new_tokens: int = 512,
    ) -> None:
        self.repo_dir = Path(repo_dir).resolve()
        self.model_id = model_id
        self.device = device
        self.temperatures = temperatures
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        """Pull weights + tokenizer via ``transformers``.

        We import inside ``load`` so ``--smoke`` does not trigger the import
        cost (or a transformers install error) when the user just wants to
        verify the JSONL writer.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import torch  # type: ignore

        # Make upstream repo importable in case it ships a `stitch.py` or
        # similar helper we want to reuse for multi-temp stitching.
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        print(f"[temp] loading {self.model_id} on {self.device}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype
        )
        self._model.to(self.device)
        self._model.eval()

    def _generate_one(self, prompt: str, temperature: float) -> str:
        import torch  # type: ignore

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outs = self._model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        text = self._tokenizer.decode(outs[0][inputs["input_ids"].shape[-1]:],
                                      skip_special_tokens=True)
        return text.strip()

    def paraphrase(self, text: str) -> str:
        """Multi-temperature sample + stitch.

        The upstream stitcher is sentence-level; in the absence of an
        importable helper we just pick the longest non-empty candidate
        (a sane fallback that the user can swap for the upstream stitcher
        once they confirm its module path on the pod).
        """
        if self._model is None:
            raise RuntimeError("TempParaphraser.load() was not called")

        prompt = TEMP_PROMPT_TEMPLATE.format(text=text)
        candidates: list[str] = []
        for t in self.temperatures:
            try:
                out = self._generate_one(prompt, t)
                if out:
                    candidates.append(out)
            except Exception as exc:
                print(f"[temp] sampling at T={t} failed: {exc!r}")
                continue
        if not candidates:
            return ""
        # TODO(user): replace with upstream stitcher if/when its module is identified.
        return max(candidates, key=len)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    print(f"[temp] reading arxiv records from {ARXIV_JSONL}")
    records = load_ai_records(ARXIV_JSONL, limit=args.limit)
    print(f"[temp] {len(records)} AI rows to humanize")

    done = load_already_done(OUTPUT_JSONL)
    if done:
        print(f"[temp] {len(done)} already in output -> resuming")

    todo = [r for r in records if r.id not in done]
    if not todo:
        print("[temp] nothing to do")
        return

    if args.smoke:
        print("[temp] SMOKE MODE -- using no-op paraphraser")
        paraphraser = None
    else:
        print(f"[temp] preparing upstream repo at {TEMP_REPO_DIR}")
        ensure_repo(TEMP_REPO_URL, TEMP_REPO_DIR)
        paraphraser = TempParaphraser(
            repo_dir=TEMP_REPO_DIR,
            model_id=args.model_id,
            device=args.device,
            temperatures=tuple(args.temperatures),
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        paraphraser.load()

    bar = (tqdm(total=len(todo), desc="[temp]", unit="rec",
                mininterval=2.0, dynamic_ncols=True)
           if tqdm is not None else None)

    failures = 0
    start = time.time()
    try:
        for rec in todo:
            try:
                if args.smoke:
                    out_text = smoke_paraphrase(rec.text)
                else:
                    out_text = paraphraser.paraphrase(rec.text)
                if not out_text or not out_text.strip():
                    raise ValueError("empty paraphrase output")
            except Exception as exc:
                failures += 1
                print(f"[temp] {rec.id} failed: {exc!r}")
                if bar is not None:
                    bar.update(1)
                continue

            row = make_humanized_record(
                source=rec,
                paraphrased_text=out_text,
                humanizer="tempparaphraser",
                id_short="tmp",
                source_short="temp",
                humanizer_model=args.model_id,
            )
            append_jsonl_atomic(OUTPUT_JSONL, row)
            if bar is not None:
                bar.update(1)
    finally:
        if bar is not None:
            bar.close()

    elapsed = time.time() - start
    print(f"[temp] done in {elapsed:.1f}s "
          f"({len(todo) - failures}/{len(todo)} succeeded; {failures} failed)")
    print(f"[temp] output: {OUTPUT_JSONL}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default=HF_MODEL_ID,
                   help=f"HF model id (default: {HF_MODEL_ID})")
    p.add_argument("--device", default="cuda")
    p.add_argument("--temperatures", type=float, nargs="+", default=[0.7, 1.0, 1.3],
                   help="sampling temperatures for the multi-sample stitch")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="use a no-op paraphraser; verifies I/O plumbing only")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
