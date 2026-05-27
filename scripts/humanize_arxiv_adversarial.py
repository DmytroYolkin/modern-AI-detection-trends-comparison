"""Humanize the AI half of the arxiv test set with Adversarial Paraphrasing.

Upstream:  https://github.com/chengez/Adversarial-Paraphrasing  (arXiv 2506.07001)

The Adv-P method is a training-free, detector-guided beam search that
rewrites a text token-by-token: at each step it generates candidate
continuations with a LLaMA-3-8B-Instruct rewriter and re-scores them with
a guidance detector (OpenAI-RoBERTa-large by default per their paper §5).
The result is text that semantically matches the input but maximally
defeats the guidance detector.

This driver:
  1. Clones the upstream repo into ``scripts/_workdirs/adversarial_paraphrasing/``
     on first run (or ``git pull`` if it already exists).
  2. Loads the 1287 ``arxiv_rewrite`` rows from
     ``data/testing_dataset/arxiv_final/arxiv_merged.jsonl``.
  3. For each row, invokes the Adv-P paraphraser.
  4. Writes the humanized rows to
     ``data/testing_dataset/arxiv_final/arxiv_humanized_adv.jsonl``,
     resumably (existing ``source_text_id`` values are skipped).

Smoke testing
-------------
``--smoke`` substitutes a no-op paraphraser so the JSONL plumbing can be
exercised without pulling the 8 B-parameter rewriter. Combine with
``--limit N`` to bound the wall-clock.

Usage
-----
    # full run on a GPU pod
    python scripts/humanize_arxiv_adversarial.py --device cuda

    # smoke (no GPU, no model download)
    python scripts/humanize_arxiv_adversarial.py --smoke --limit 5

    # try a different guidance classifier
    python scripts/humanize_arxiv_adversarial.py \
        --guidance-classifier roberta-large-openai-detector
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Bootstrap so `python scripts/humanize_arxiv_adversarial.py` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from tqdm import tqdm
except ImportError:  # tqdm is nice-to-have, not required
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


ADV_REPO_URL = "https://github.com/chengez/Adversarial-Paraphrasing.git"
ADV_REPO_DIR = WORKDIRS / "adversarial_paraphrasing"
OUTPUT_JSONL = REPO_ROOT / "data" / "testing_dataset" / "arxiv_final" / "arxiv_humanized_adv.jsonl"


# ---------------------------------------------------------------------------
# Upstream-paraphraser loader
# ---------------------------------------------------------------------------

class AdvParaphraser:
    """Thin wrapper around the upstream Adversarial-Paraphrasing code path.

    The upstream entry point is exposed via their ``transfer_test.sbatch`` /
    ``main.py`` style scripts -- which hardcode SLURM paths and CLI args.
    We import their library directly when possible (``from adversarial_paraphrasing
    import paraphrase`` or similar). If the upstream layout changes such that
    no clean Python entry point is exposed, we fall back to ``subprocess.run``
    against ``main.py`` (documented below). The wrapper's ``paraphrase()`` is
    a single function so the driver loop stays linear.
    """

    def __init__(
        self,
        repo_dir: Path,
        *,
        rewriter_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        guidance_classifier: str = "roberta-large-openai-detector",
        device: str = "cuda",
        max_new_tokens: int = 512,
        beam_size: int = 4,
    ) -> None:
        self.repo_dir = Path(repo_dir).resolve()
        self.rewriter_model = rewriter_model
        self.guidance_classifier = guidance_classifier
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.beam_size = beam_size
        self._impl = None

    def load(self) -> None:
        """Import the upstream module and instantiate the underlying pipeline."""
        # Make the cloned repo importable.
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        # The exact symbol name depends on the upstream layout. Adv-P's repo
        # at commit time of writing exposes its core in `paraphrase.py` /
        # `attack/paraphrase.py`. We try the most plausible import paths and
        # raise a clear error if none work so the user can debug on the pod.
        candidates = [
            ("paraphrase", "AdversarialParaphraser"),
            ("attack.paraphrase", "AdversarialParaphraser"),
            ("src.paraphrase", "AdversarialParaphraser"),
            ("adversarial_paraphrasing", "AdversarialParaphraser"),
        ]
        last_err: Exception | None = None
        for mod_name, cls_name in candidates:
            try:
                mod = __import__(mod_name, fromlist=[cls_name])
                cls = getattr(mod, cls_name)
                self._impl = cls(
                    rewriter_model=self.rewriter_model,
                    guidance_classifier=self.guidance_classifier,
                    device=self.device,
                    max_new_tokens=self.max_new_tokens,
                    beam_size=self.beam_size,
                )
                print(f"[adv] loaded {mod_name}.{cls_name}")
                return
            except (ImportError, AttributeError) as e:
                last_err = e
                continue

        raise ImportError(
            "Could not locate Adversarial-Paraphrasing's Python entry point. "
            "Tried: " + ", ".join(f"{m}.{c}" for m, c in candidates) + ". "
            f"Last error: {last_err!r}. "
            "Inspect the cloned repo at "
            f"{self.repo_dir} and update the `candidates` list in "
            "scripts/humanize_arxiv_adversarial.py to point at the correct "
            "module + class. If the upstream code is purely SLURM-driven, "
            "shell out via subprocess.run on transfer_test.sbatch's underlying "
            "python invocation instead."
        )

    def paraphrase(self, text: str) -> str:
        if self._impl is None:
            raise RuntimeError("AdvParaphraser.load() was not called")
        # We assume the upstream class exposes a `.paraphrase(text) -> str`
        # method. If it instead returns a dict, unwrap the relevant field.
        out = self._impl.paraphrase(text)
        if isinstance(out, dict):
            return out.get("text") or out.get("output") or out.get("paraphrased") or ""
        return str(out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    print(f"[adv] reading arxiv records from {ARXIV_JSONL}")
    records = load_ai_records(ARXIV_JSONL, limit=args.limit)
    print(f"[adv] {len(records)} AI rows to humanize")

    done = load_already_done(OUTPUT_JSONL)
    if done:
        print(f"[adv] {len(done)} already in output -> resuming")

    todo = [r for r in records if r.id not in done]
    if not todo:
        print("[adv] nothing to do")
        return

    if args.smoke:
        print("[adv] SMOKE MODE -- using no-op paraphraser")
        paraphraser = None
    else:
        print(f"[adv] preparing upstream repo at {ADV_REPO_DIR}")
        ensure_repo(ADV_REPO_URL, ADV_REPO_DIR)
        paraphraser = AdvParaphraser(
            repo_dir=ADV_REPO_DIR,
            rewriter_model=args.rewriter_model,
            guidance_classifier=args.guidance_classifier,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            beam_size=args.beam_size,
        )
        paraphraser.load()

    bar = (tqdm(total=len(todo), desc="[adv]", unit="rec",
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
                print(f"[adv] {rec.id} failed: {exc!r}")
                if bar is not None:
                    bar.update(1)
                continue

            row = make_humanized_record(
                source=rec,
                paraphrased_text=out_text,
                humanizer="adversarial_paraphrasing",
                id_short="adv",
                source_short="adv",
                humanizer_model=args.rewriter_model,
            )
            append_jsonl_atomic(OUTPUT_JSONL, row)
            if bar is not None:
                bar.update(1)
    finally:
        if bar is not None:
            bar.close()

    elapsed = time.time() - start
    print(f"[adv] done in {elapsed:.1f}s "
          f"({len(todo) - failures}/{len(todo)} succeeded; {failures} failed)")
    print(f"[adv] output: {OUTPUT_JSONL}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rewriter-model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                   help="HF id of the rewriter LM (default: Meta-Llama-3-8B-Instruct)")
    p.add_argument("--guidance-classifier", default="roberta-large-openai-detector",
                   help="HF id of the guidance detector "
                        "(default: roberta-large-openai-detector, the Adv-P paper's best)")
    p.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--beam-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=None,
                   help="cap number of AI rows processed (smoke testing)")
    p.add_argument("--smoke", action="store_true",
                   help="use a no-op paraphraser; verifies I/O plumbing only")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
