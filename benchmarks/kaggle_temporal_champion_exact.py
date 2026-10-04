"""Reproduce and instrument the historical deterministic LongMemEval champion.

This intentionally checks out ca53594 and uses the exact configuration recovered
from archived Kaggle kernels krun-det1/krun-det2/krun-confirm.  The only source
change is instrumentation: emit a post-trust-gate/pre-temporal candidate dump
with enough metadata to replay temporal-v2 + the later gate pin offline.

Expected historical result: R@1=0.958, R@5=R@10=1.0, gate fired 86/500.
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
REPO = WORK / "mnemonics-ca53594"
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
        raise RuntimeError(f"checkout drift: {head}")
    print(f"SOURCE_SHA={head}", flush=True)


def instrument_harness() -> None:
    """Add dump-only probes without changing ranking semantics."""
    path = REPO / "benchmarks" / "longmemeval_eval.py"
    src = path.read_text()

    old = "    cand_dump: list[dict] = []\n    llm_fired = 0"
    new = (
        "    cand_dump: list[dict] = []\n"
        "    temporal_dump: list[dict] = []\n"
        "    temporal_dump_path = os.environ.get('LME_TEMPORAL_DUMP')\n"
        "    llm_fired = 0"
    )
    if src.count(old) != 1:
        raise RuntimeError("instrumentation anchor 1 drift")
    src = src.replace(old, new, 1)

    old = '''                if gate_info["fired"]:\n                    gate_fired += 1\n                    gate_top_id = result["results"][0].get("id")\n\n            # Temporal-aware post-rerank.'''
    new = '''                if gate_info["fired"]:\n                    gate_fired += 1\n                    gate_top_id = result["results"][0].get("id")\n\n            if temporal_dump_path:\n                temporal_dump.append({\n                    "schema": "temporal_ablation_v2",\n                    "stage": "post_gate_pre_temporal",\n                    "qid": q.get("question_id"),\n                    "qtype": q.get("question_type"),\n                    "question": q.get("question"),\n                    "answer_sids": list(q.get("answer_session_ids") or []),\n                    "rows": [\n                        {"id": r.get("id"),\n                         "sid": _session_id_of(r.get("text")),\n                         "text": r.get("text")}\n                        for r in result["results"]\n                    ],\n                    "gate_top_id": gate_top_id,\n                    "gate_info": dict(gate_info) if gate_info is not None else None,\n                    "trust_gate_pin_margin": trust_gate_pin_margin,\n                    "temporal_v2": temporal_v2,\n                    "temporal_v3": temporal_v3,\n                })\n\n            # Temporal-aware post-rerank.'''
    if src.count(old) != 1:
        raise RuntimeError("instrumentation anchor 2 drift")
    src = src.replace(old, new, 1)

    old = '''    if dump_candidates is not None:\n        dump_candidates.write_text(json.dumps(cand_dump))\n        print(f"  candidate dump: {len(cand_dump)} q -> {dump_candidates}", flush=True)\n    return out'''
    new = '''    if dump_candidates is not None:\n        dump_candidates.write_text(json.dumps(cand_dump))\n        print(f"  candidate dump: {len(cand_dump)} q -> {dump_candidates}", flush=True)\n    if temporal_dump_path:\n        Path(temporal_dump_path).write_text(json.dumps(temporal_dump))\n        print(f"  temporal candidate dump: {len(temporal_dump)} q -> {temporal_dump_path}", flush=True)\n    return out'''
    if src.count(old) != 1:
        raise RuntimeError("instrumentation anchor 3 drift")
    path.write_text(src.replace(old, new, 1))
    print("INSTRUMENTATION_ONLY=post_gate_pre_temporal_dump", flush=True)


retry(clone_exact, "clone exact source")
instrument_harness()
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
DUMP = RESULTS / "temporal_post_gate_pre_temporal.json"
RAW = RESULTS / "raw_pre_gate.json"
OUT = RESULTS / "historical_champion.json"
PERQ = RESULTS / "historical_champion_perq.json"

common = [
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

env = os.environ.copy()
env["LME_DATA"] = DATA
env["LME_TEMPORAL_DUMP"] = str(DUMP)
env["MNEMONICS_RERANK_MODEL"] = RERANK
env["MNEMONICS_DETERMINISTIC"] = "1"
env["PYTHONUNBUFFERED"] = "1"
env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

print("EXACT_CONFIG=" + json.dumps(common), flush=True)
print(f"RERANK_CHECKPOINT={Path(RERANK).name}", flush=True)
print(f"GATE_CHECKPOINT={Path(GATE).name}", flush=True)
run = subprocess.run(
    [sys.executable, "-u", "benchmarks/longmemeval_eval.py", "--n", "500", *common,
     "--dump-candidates", str(RAW),
     "--out", str(OUT), "--per-q-out", str(PERQ)],
    cwd=REPO,
    env=env,
)
if run.returncode != 0:
    raise SystemExit(run.returncode)

result = json.loads(OUT.read_text())["mnemonics_rerank"]
dump = json.loads(DUMP.read_text())
summary = {
    "source_sha": PINNED_SHA,
    "instrumentation_only": True,
    "n_dump": len(dump),
    "n_raw_dump": len(json.loads(RAW.read_text())),
    "R@1": result["R@1"],
    "R@5": result["R@5"],
    "R@10": result["R@10"],
    "trust_gate": result.get("trust_gate"),
    "config": common,
    "rerank_checkpoint": Path(RERANK).name,
    "gate_checkpoint": Path(GATE).name,
    "dataset": Path(DATA).name,
}
(RESULTS / "historical_champion_summary.json").write_text(json.dumps(summary, indent=2))
print("HISTORICAL_CHAMPION=" + json.dumps(summary, sort_keys=True), flush=True)
if len(dump) != 500:
    raise RuntimeError(f"dump incomplete: {len(dump)}/500")
raw_dump = json.loads(RAW.read_text())
if len(raw_dump) != 500:
    raise RuntimeError(f"raw dump incomplete: {len(raw_dump)}/500")
if result["R@1"] != 0.958 or result["R@5"] != 1.0 or result["R@10"] != 1.0:
    raise RuntimeError(f"historical champion mismatch: {result}")
if (result.get("trust_gate") or {}).get("fired") != 86:
    raise RuntimeError(f"gate count mismatch: {result.get('trust_gate')}")
print("HISTORICAL_CHAMPION_REPRO_OK", flush=True)
