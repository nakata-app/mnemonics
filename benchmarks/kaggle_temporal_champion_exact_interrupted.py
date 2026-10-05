"""Exact historical champion temporal ablation on Kaggle T4.

Candidate generation is pinned to ca53594, the commit recorded in CHAMPION.json.
When Kaggle exposes two T4s, the 500 independent questions are split 250/250
across the GPUs and merged back in original dataset order. With one GPU the
runner falls back to one 500-question process.

Offline temporal analysis is pinned to 120210a, where the historical inline
temporal block was extracted without changing its default semantics.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
CHAMP = WORK / "mnemonics-champion"
ABL = WORK / "mnemonics-ablation"
RESULTS = WORK / "results"
RESULTS.mkdir(exist_ok=True)

CHAMP_SHA = "ca53594"
ABL_SHA = "120210a"
CE_MODEL = "BAAI/bge-reranker-v2-m3"


def checked(cmd, **kwargs):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


for repo in (CHAMP, ABL):
    subprocess.run(["rm", "-rf", str(repo)], check=False)

checked(["git", "clone", "https://github.com/nakata-app/mnemonics.git", str(CHAMP)])
checked(["git", "-C", str(CHAMP), "checkout", "--detach", CHAMP_SHA])
champ_head = subprocess.check_output(
    ["git", "-C", str(CHAMP), "rev-parse", "--short", "HEAD"], text=True
).strip()
if champ_head != CHAMP_SHA:
    raise RuntimeError(f"wrong champion checkout: {champ_head}")

checked(["git", "clone", "https://github.com/nakata-app/mnemonics.git", str(ABL)])
checked(["git", "-C", str(ABL), "checkout", "--detach", ABL_SHA])
abl_head = subprocess.check_output(
    ["git", "-C", str(ABL), "rev-parse", "--short", "HEAD"], text=True
).strip()
if abl_head != ABL_SHA:
    raise RuntimeError(f"wrong ablation checkout: {abl_head}")

checked(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-e",
        str(CHAMP),
        "sentence-transformers",
        "numpy",
    ]
)

data = None
for root, _, files in os.walk("/kaggle/input"):
    if "longmemeval_s_cleaned.json" in files:
        data = Path(root) / "longmemeval_s_cleaned.json"
        break
if data is None:
    raise RuntimeError("longmemeval_s_cleaned.json not found in mounted dataset")

questions = json.loads(data.read_text())
if len(questions) != 500:
    raise RuntimeError(f"expected 500 LongMemEval-S questions, got {len(questions)}")

env = os.environ.copy()
env["MNEMONICS_DETERMINISTIC"] = "1"
env["MNEMONICS_RERANK_MODEL"] = CE_MODEL
env["MNEMONICS_EMBED_BACKEND"] = "sentence-transformers"
env["PYTHONUNBUFFERED"] = "1"
env["LME_DATA"] = str(data)
env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

gpu_count = int(
    subprocess.check_output(
        [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
        text=True,
    ).strip()
)
print(f"CUDA_DEVICE_COUNT={gpu_count}", flush=True)
if gpu_count < 1:
    raise RuntimeError("Kaggle run has no CUDA GPU")

# Warm the shared Hugging Face cache once before parallel workers start.
checked(
    [
        sys.executable,
        "-c",
        (
            "from sentence_transformers import SentenceTransformer, CrossEncoder; "
            "SentenceTransformer('all-MiniLM-L6-v2'); "
            f"CrossEncoder('{CE_MODEL}'); "
            "print('MODEL_CACHE_READY')"
        ),
    ],
    env=env,
)

combined_cands = RESULTS / "cands_ca53594.json"


def eval_cmd(split_file: Path | None, cands: Path, out: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "benchmarks/longmemeval_eval.py",
        "--mode",
        "rerank",
        "--chunk-mode",
        "turn",
        "--augment-preferences",
        "--candidate-k",
        "50",
        "--seed",
        "42",
        "--dump-candidates",
        str(cands),
        "--out",
        str(out),
    ]
    if split_file is None:
        cmd[3:3] = ["--n", "500"]
    else:
        cmd[3:3] = ["--split-file", str(split_file)]
    return cmd


if gpu_count >= 2:
    midpoint = len(questions) // 2
    shard_specs = []
    for idx, subset in enumerate((questions[:midpoint], questions[midpoint:])):
        split = RESULTS / f"split_{idx}.json"
        split.write_text(json.dumps({"dev": [q["question_id"] for q in subset]}))
        cands = RESULTS / f"cands_shard_{idx}.json"
        out = RESULTS / f"run_shard_{idx}.json"
        shard_specs.append((idx, split, cands, out))

    procs = []
    for idx, split, cands, out in shard_specs:
        worker_env = env.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = str(idx)
        cmd = eval_cmd(split, cands, out)
        print(f"START SHARD {idx}: {' '.join(cmd)}", flush=True)
        procs.append((idx, subprocess.Popen(cmd, cwd=CHAMP, env=worker_env)))

    failed = []
    for idx, proc in procs:
        rc = proc.wait()
        print(f"SHARD {idx} EXIT={rc}", flush=True)
        if rc != 0:
            failed.append((idx, rc))
    if failed:
        raise RuntimeError(f"historical shard failure(s): {failed}")

    merged = []
    for idx, _split, cands, _out in shard_specs:
        rows = json.loads(cands.read_text())
        expected = midpoint if idx == 0 else len(questions) - midpoint
        if len(rows) != expected:
            raise RuntimeError(f"shard {idx}: expected {expected} candidates, got {len(rows)}")
        merged.extend(rows)
    combined_cands.write_text(json.dumps(merged))
else:
    run_out = RESULTS / "run_ca53594_pre_temporal.json"
    cmd = eval_cmd(None, combined_cands, run_out)
    print("START MONOLITHIC:", " ".join(cmd), flush=True)
    checked(cmd, cwd=CHAMP, env=env)

rows = json.loads(combined_cands.read_text())
if len(rows) != 500:
    raise RuntimeError(f"historical candidate dump incomplete: {len(rows)} != 500")

dataset_order = [q["question_id"] for q in questions]
candidate_order = [row["qid"] for row in rows]
if candidate_order != dataset_order:
    raise RuntimeError("combined candidate qid order differs from dataset order")

ablation = RESULTS / "ablation_ca53594.json"
checked(
    [
        sys.executable,
        "-u",
        str(ABL / "benchmarks" / "temporal_ablation_offline.py"),
        "--candidates",
        str(combined_cands),
        "--data",
        str(data),
        "--out",
        str(ablation),
        "--expected-n",
        "500",
    ],
    cwd=ABL,
    env=env,
)

res = json.loads(ablation.read_text())
provenance = {
    "candidate_source_commit": CHAMP_SHA,
    "ablation_code_commit": ABL_SHA,
    "candidate_sha256": hashlib.sha256(combined_cands.read_bytes()).hexdigest(),
    "candidate_count": len(rows),
    "config": "--n 500 --mode rerank --chunk-mode turn --augment-preferences --candidate-k 50 --seed 42",
    "ce_model": CE_MODEL,
    "deterministic": True,
    "cuda_device_count": gpu_count,
}
(RESULTS / "provenance_ca53594.json").write_text(json.dumps(provenance, indent=2))
res["provenance"] = provenance
ablation.write_text(json.dumps(res, indent=2))

r = res["variants"]["on_labeled"]["R@1"]
print(f"HISTORICAL_ON_LABELED_R1={r}", flush=True)
if abs(r - 0.958) > 1e-12:
    raise RuntimeError(
        f"historical champion reproduction failed: expected 0.958, got {r}"
    )

print("HISTORICAL_CHAMPION_REPRO_OK", flush=True)
print(json.dumps(res, indent=2), flush=True)
