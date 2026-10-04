"""Faithful historical LongMemEval-S champion reproduction on Kaggle GPU.

Recovered 2026-10-04 from the archived June kernels ``krun-det1``,
``krun-det2`` and ``krun-confirm``.  This script intentionally pins the exact
source commit and configuration that produced the recorded 0.958 score.

Required Kaggle datasets:
  atakanakbaba/mnemonics-lme
  atakanakbaba/mnemonics-champion-ce

Expected 500q result:
  R@1=0.958, R@5=1.0, R@10=1.0, trust gate fired=86.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "mnemonics-champion"
RESULTS = WORK / "results"
RESULTS.mkdir(exist_ok=True)
PINNED_SHA = "ca5359417c9a4692d86e58bc36253e00b54172f8"


def retry(fn, what: str, tries: int = 3, wait: int = 8):
    for i in range(1, tries + 1):
        try:
            return fn()
        except Exception as exc:
            print(f"[retry {i}/{tries}] {what}: {exc}", flush=True)
            if i == tries:
                raise
            time.sleep(wait * i)


def resolve(name: str, is_dir: bool) -> str:
    hits = [
        x for x in glob.glob(f"/kaggle/input/**/{name}", recursive=True)
        if (os.path.isdir(x) if is_dir else os.path.isfile(x))
    ]
    if not hits:
        raise RuntimeError(f"required Kaggle input missing: {name}")
    return sorted(hits, key=len)[0]


def clone_exact() -> None:
    subprocess.run(["rm", "-rf", str(REPO)], check=False)
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "https://github.com/nakata-app/mnemonics.git", str(REPO)],
        check=True,
    )
    subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", PINNED_SHA], check=True)
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if head != PINNED_SHA:
        raise RuntimeError(f"checkout drift: expected {PINNED_SHA}, got {head}")
    print(f"SOURCE_SHA={head}", flush=True)


retry(clone_exact, "git clone")
retry(
    lambda: subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO),
         "sentence-transformers==5.4.0", "numpy", "adaptmem"],
        check=True,
    ),
    "pip install",
)

DATA = resolve("longmemeval_s_cleaned.json", False)
GATE = resolve("chat-ce-v3-20260516", True)
RERANK = resolve("mn-ce-v1-20260604", True)

env = os.environ.copy()
env["LME_DATA"] = DATA
env["PYTHONUNBUFFERED"] = "1"
env["MNEMONICS_RERANK_MODEL"] = RERANK
env["MNEMONICS_DETERMINISTIC"] = "1"
env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

COMMON = [
    "--mode", "rerank",
    "--chunk-mode", "turn",
    "--temporal-aware",
    "--temporal-v2",
    "--candidate-k", "50",
    "--seed", "42",
    "--trust-gate-ce", GATE,
    "--trust-gate-margin", "0.2",
    "--trust-gate-pin-margin", "0.5",
]


def stage(n: int, tag: str):
    out = RESULTS / f"lme{n}_champion.json"
    perq = RESULTS / f"lme{n}_champion_perq.json"
    print(f"\n=== {tag} ({n}q) ===", flush=True)
    run = subprocess.run(
        [sys.executable, "-u", "benchmarks/longmemeval_eval.py", "--n", str(n),
         *COMMON, "--out", str(out), "--per-q-out", str(perq)],
        cwd=REPO,
        env=env,
    )
    if run.returncode != 0 or not out.exists():
        raise RuntimeError(f"{tag} failed rc={run.returncode}")
    result = json.loads(out.read_text())["mnemonics_rerank"]
    print(
        f"{tag}: R@1={result['R@1']} R@5={result['R@5']} R@10={result['R@10']} "
        f"gate_fired={(result.get('trust_gate') or {}).get('fired')}",
        flush=True,
    )
    return result


smoke = stage(5, "SMOKE")
full = stage(500, "500q")
summary = {
    "source_sha": PINNED_SHA,
    "config": COMMON,
    "rerank_checkpoint": Path(RERANK).name,
    "gate_checkpoint": Path(GATE).name,
    "encoder": "all-MiniLM-L6-v2",
    "dataset": Path(DATA).name,
    "smoke": smoke,
    "r500": full,
}
(RESULTS / "champion_summary.json").write_text(json.dumps(summary, indent=2))

if full["R@1"] != 0.958 or full["R@5"] != 1.0 or full["R@10"] != 1.0:
    raise RuntimeError(f"historical champion mismatch: {full}")
if (full.get("trust_gate") or {}).get("fired") != 86:
    raise RuntimeError(f"trust-gate count mismatch: {full.get('trust_gate')}")
print("HISTORICAL_CHAMPION_REPRO_OK", flush=True)
