"""Pinned BEAM production-granularity coverage + CE benchmark for Kaggle T4.

Measures the live-shaped s200 chunking path under a 4000-word budget with the
same bge-reranker-v2-m3 CE used by the LongMemEval champion. This replaces the
older kaggle_beam_ce.py job, which embedded a stale whole-turn probe.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "mnemonics"
RESULTS = WORK / "beam-cover-ce"
RESULTS.mkdir(parents=True, exist_ok=True)

REPO_URL = "https://github.com/nakata-app/mnemonics.git"
BRANCH = "work/memory-sota"
PINNED_SHA = "e0fa00610ae8ccbf05b35d768851928e92e8e3f6"
CE_MODEL = "BAAI/bge-reranker-v2-m3"


def run(cmd: list[str], *, cwd: Path | None = None, env=None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def checkout() -> None:
    run(["rm", "-rf", str(REPO)])
    run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO)])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if head != PINNED_SHA:
        run(["git", "fetch", "--depth", "1", "origin", PINNED_SHA], cwd=REPO)
        run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=REPO)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if head != PINNED_SHA:
        raise RuntimeError(f"checkout drift: expected {PINNED_SHA}, got {head}")
    print("pinned head", head, flush=True)


def stage(size: str, *, limit: int = 0, tag: str) -> dict:
    out_dir = RESULTS / tag
    cmd = [
        sys.executable,
        "-u",
        "benchmarks/beam_cover_diversity.py",
        "--sizes",
        size,
        "--windows",
        "s200",
        "--arms",
        "baseline",
        "--candidate-k",
        "50",
        "--budget",
        "4000",
        "--rerank",
        "--out-dir",
        str(out_dir),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    run(cmd, cwd=REPO, env=ENV)
    result_file = out_dir / f"diversity_{size}_ce_k50.json"
    if not result_file.exists():
        raise RuntimeError(f"missing result file {result_file}")
    result = json.loads(result_file.read_text())
    print(
        size,
        json.dumps(result.get("variants", {}).get("ws200:baseline", {}), indent=2),
        flush=True,
    )
    return result


checkout()
run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "-e",
    str(REPO),
    "sentence-transformers",
    "numpy",
    "pyarrow",
    "huggingface_hub",
])

ENV = os.environ.copy()
ENV.update(
    {
        "PYTHONUNBUFFERED": "1",
        "MNEMONICS_DETERMINISTIC": "1",
        "MNEMONICS_EMBED_BACKEND": "sentence-transformers",
        "MNEMONICS_RERANK_MODEL": CE_MODEL,
        "MNEMONICS_RERANK_MAX_LENGTH": "512",
        "MNEMONICS_RERANK_BATCH_SIZE": "8",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }
)

summary: dict[str, object] = {
    "pinned_sha": PINNED_SHA,
    "ce_model": CE_MODEL,
    "chunking": "s200",
    "candidate_k": 50,
    "budget_words": 4000,
}

# Fail-fast smoke before expensive tiers.
summary["smoke"] = stage("100K", limit=2, tag="smoke")

for size in ("100K", "500K", "1M"):
    try:
        summary[size] = stage(size, tag=f"ce-{size}")
    except Exception as exc:
        summary[size] = {"error": str(exc)}
        print(f"[{size} failed] {exc}", flush=True)

(RESULTS / "beam_cover_ce_summary.json").write_text(json.dumps(summary, indent=2))
print("SUMMARY", json.dumps(summary, indent=2), flush=True)
