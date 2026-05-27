"""Shared helpers for the arxiv-humanizer scripts.

The two humanizer drivers (``humanize_arxiv_adversarial.py`` and
``humanize_arxiv_temppara.py``) read the same arxiv JSONL, emit the same
output schema, and use the same resumable-checkpoint discipline as
``training/build_dataset.py``. The shared plumbing lives here so the
two driver scripts only contain the upstream-specific glue.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


REPO_ROOT = Path(__file__).resolve().parent.parent
ARXIV_JSONL = REPO_ROOT / "data" / "testing_dataset" / "arxiv_final" / "arxiv_merged.jsonl"
WORKDIRS = Path(__file__).resolve().parent / "_workdirs"


# ---------------------------------------------------------------------------
# Upstream-repo management
# ---------------------------------------------------------------------------

def ensure_repo(url: str, dest: Path, *, branch: str | None = None) -> Path:
    """Clone ``url`` into ``dest`` if missing, otherwise ``git pull``.

    Returns the absolute path of the working copy so callers can ``sys.path``-
    insert it.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"[humanizer] cloning {url} -> {dest}")
        cmd = ["git", "clone", "--depth", "1"]
        if branch:
            cmd += ["--branch", branch]
        cmd += [url, str(dest)]
        subprocess.run(cmd, check=True)
    else:
        print(f"[humanizer] repo already present at {dest}; pulling latest")
        try:
            subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
        except Exception as e:  # network can be flaky on pods; non-fatal
            print(f"[humanizer] git pull failed (continuing with cached copy): {e}")
    return dest.resolve()


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

@dataclass
class AIRecord:
    """A single AI (arxiv_rewrite) row that needs humanizing."""
    id: str
    text: str
    author_id: str
    model: str | None
    domain: str
    raw: dict


def load_ai_records(path: Path = ARXIV_JSONL, limit: int | None = None) -> list[AIRecord]:
    """Read ``arxiv_merged.jsonl`` and filter to AI (``arxiv_rewrite``) rows."""
    rows: list[AIRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("source") != "arxiv_rewrite":
                continue
            rows.append(AIRecord(
                id=str(obj["id"]),
                text=str(obj["text"]),
                author_id=str(obj["author_id"]),
                model=obj.get("model"),
                domain=str(obj.get("domain", "arxiv_cs")),
                raw=obj,
            ))
            if limit is not None and len(rows) >= limit:
                break
    return rows


# ---------------------------------------------------------------------------
# Resumable JSONL output
# ---------------------------------------------------------------------------

def load_already_done(out_path: Path) -> set[str]:
    """Set of ``source_text_id`` values already in ``out_path`` (resume key)."""
    if not out_path.exists():
        return set()
    done: set[str] = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate the last (partial) line
            sid = obj.get("source_text_id")
            if sid:
                done.add(str(sid))
    return done


def append_jsonl_atomic(out_path: Path, record: dict) -> None:
    """Append one record, using a copy+rename to avoid mid-write corruption.

    Mirrors the ``tmp``+``os.replace`` pattern in
    ``training/build_dataset.py::_save_features`` so a SIGKILL during the
    append never leaves a partially-written line in the canonical file.

    NOTE: this rewrites the entire file each call. That is O(N) per row;
    fine for ~1.3 k records of paragraph-length output (~MBs), and a tiny
    price for crash-safety. If a humanizer produced ten of thousands of
    rows we would switch to ``open(..., "a")`` + ``fsync``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".tmp.jsonl")
    # Copy existing content into tmp first.
    if out_path.exists():
        with out_path.open("rb") as src, tmp.open("wb") as dst:
            for chunk in iter(lambda: src.read(65536), b""):
                dst.write(chunk)
    else:
        tmp.write_bytes(b"")
    # Append the new line to tmp.
    with tmp.open("ab") as dst:
        dst.write(json.dumps(record, ensure_ascii=False).encode("utf-8"))
        dst.write(b"\n")
    os.replace(tmp, out_path)


# ---------------------------------------------------------------------------
# Output schema construction
# ---------------------------------------------------------------------------

def make_humanized_record(
    *,
    source: AIRecord,
    paraphrased_text: str,
    humanizer: str,         # "adversarial_paraphrasing" | "tempparaphraser"
    id_short: str,          # "adv" | "tmp" -- used in the id prefix
    source_short: str,      # "adv" | "temp" -- used in the `source` field
    humanizer_model: str,
) -> dict:
    """Construct one humanized JSONL row.

    The new ``id`` mirrors the source ``axr_NNNN`` index (``axr_0123`` ->
    ``axhm_<id_short>_0123``). ``author_id`` is inherited from the source
    AI record per the methodology note in the task brief.

    Note: ``id_short`` and ``source_short`` differ for TempParaphraser --
    the id prefix is ``axhm_tmp_*`` (per the task brief's ID scheme) but
    the ``source`` value is ``arxiv_humanized_temp`` (per the task brief's
    schema example). For Adv-P both are ``adv``.
    """
    src_id = source.id  # e.g. "axr_0123"
    # robust suffix extraction: take whatever comes after the last "_"
    suffix = src_id.rsplit("_", 1)[-1] if "_" in src_id else src_id
    new_id = f"axhm_{id_short}_{suffix}"

    word_count = len(paraphrased_text.split())
    original_model = source.model or "unknown"

    return {
        "id": new_id,
        "text": paraphrased_text,
        "is_ai": True,
        "label": "ai",
        "source": f"arxiv_humanized_{source_short}",
        "domain": source.domain,
        "author_id": source.author_id,
        "model": f"{original_model}+humanized_by_{source_short}",
        "exam_type": None,
        "prompt": None,
        "source_text_id": src_id,
        "humanizer": humanizer,
        "humanizer_model": humanizer_model,
        "text_length_words": word_count,
        "split": "test",
    }


# ---------------------------------------------------------------------------
# Smoke-test paraphraser (no GPU, no network) -- used by `--smoke`
# ---------------------------------------------------------------------------

def smoke_paraphrase(text: str) -> str:
    """No-op paraphraser used to verify the I/O plumbing.

    Returns ``"PARAPHRASED: <input>"``. Intentionally deterministic so the
    smoke run is reproducible. Real paraphrasers replace this with their
    model call.
    """
    return f"PARAPHRASED: {text}"
