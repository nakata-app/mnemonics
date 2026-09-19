"""Kaggle A/B: historical LongMemEval champion path vs planned retrieval.

This script pins the exact feature commit and the historical sentence-transformers
embedding backend. It changes only the retrieval policy (--planned), so the
comparison is attributable and reproducible.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "mnemonics-memory-sota"
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

PINNED_SHA = "e624a2f4db729f5fd7f50c0121259b7a1ba954f9"
BRANCH = "work/memory-sota"
REPO_URL = "https://github.com/nakata-app/mnemonics.git"
DATASET = Path("/kaggle/input/mnemonics-lme/longmemeval_s_cleaned.json")
CE_MODEL = "BAAI/bge-reranker-v2-m3"


def run(cmd: list[str], *, cwd: Path | None = None, env=None, check=True):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env, check=check)


def clone_pinned() -> None:
    run(["rm", "-rf", str(REPO)], check=False)
    run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO)])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if head != PINNED_SHA:
        run(["git", "fetch", "--depth", "1", "origin", PINNED_SHA], cwd=REPO)
        run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=REPO)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if head != PINNED_SHA:
        raise RuntimeError(f"benchmark checkout drift: expected {PINNED_SHA}, got {head}")
    print(f"PINNED HEAD {head}", flush=True)


def install() -> None:
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-e",
        str(REPO),
        "sentence-transformers",
    ])


COMMON = [
    "--mode", "rerank",
    "--chunk-mode", "turn",
    "--temporal-aware",
    "--augment-preferences",
    "--candidate-k", "50",
    "--seed", "42",
]


def stage(n: int, *, planned: bool, tag: str, env: dict[str, str]) -> dict:
    suffix = "planned" if planned else "baseline"
    out = RESULTS / f"lme{n}_{suffix}.json"
    perq = RESULTS / f"lme{n}_{suffix}_perq.json"
    cmd = [
        sys.executable,
        "-u",
        "benchmarks/longmemeval_eval.py",
        "--n", str(n),
        *COMMON,
        "--out", str(out),
        "--per-q-out", str(perq),
    ]
    if planned:
        cmd.append("--planned")
    print(f"\n=== {tag}: n={n} planned={planned} ===", flush=True)
    started = time.time()
    run(cmd, cwd=REPO, env=env)
    result = json.loads(out.read_text())["mnemonics_rerank"]
    print(
        f"{tag}: R@1={result['R@1']:.4f} R@5={result['R@5']:.4f} "
        f"R@10={result['R@10']:.4f} runtime={time.time()-started:.1f}s",
        flush=True,
    )
    return result


def main() -> None:
    if not DATASET.exists() or DATASET.stat().st_size < 100_000_000:
        raise RuntimeError(f"LongMemEval dataset missing or too small: {DATASET}")

    clone_pinned()
    install()

    env = os.environ.copy()
    env.update({
        "LME_DATA": str(DATASET),
        "PYTHONUNBUFFERED": "1",
        "MNEMONICS_DETERMINISTIC": "1",
        "MNEMONICS_EMBED_BACKEND": "sentence-transformers",
        "MNEMONICS_RERANK_MODEL": CE_MODEL,
    })

    smoke_base = stage(5, planned=False, tag="SMOKE baseline", env=env)
    smoke_plan = stage(5, planned=True, tag="SMOKE planned", env=env)

    r100_base = stage(100, planned=False, tag="100q baseline", env=env)
    r100_plan = stage(100, planned=True, tag="100q planned", env=env)

    # A severe 100q regression is enough to avoid burning a full T4 run.
    severe_regression = r100_plan["R@1"] + 0.05 < r100_base["R@1"]
    r500_plan = None
    if not severe_regression:
        r500_plan = stage(500, planned=True, tag="500q planned", env=env)
    else:
        print("planned retrieval regressed >5pp at 100q; skipping 500q", flush=True)

    summary = {
        "pinned_sha": PINNED_SHA,
        "branch": BRANCH,
        "dataset": str(DATASET),
        "config": COMMON,
        "encoder_backend": "sentence-transformers",
        "ce_model": CE_MODEL,
        "smoke": {"baseline": smoke_base, "planned": smoke_plan},
        "q100": {"baseline": r100_base, "planned": r100_plan},
        "q500_planned": r500_plan,
        "historical_champion": {
            "n": 500,
            "R@1": 0.958,
            "R@5": 1.0,
            "R@10": 1.0,
        },
    }
    (RESULTS / "memory_sota_ab.json").write_text(json.dumps(summary, indent=2))
    print("\n=== MEMORY SOTA A/B SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()