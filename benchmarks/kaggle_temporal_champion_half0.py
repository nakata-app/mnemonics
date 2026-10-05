"""Generate one exact half of the ca53594 champion candidate dump on Kaggle.

Set OUTER_HALF to 0 or 1 before upload. Each half contains exact dataset-order
QIDs 0:250 or 250:500. If two GPUs are visible, the half is internally split
again so each GPU handles 125 independent questions.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

OUTER_HALF = 0
CHAMP_SHA = "ca53594"
CE_MODEL = "BAAI/bge-reranker-v2-m3"

WORK = Path("/kaggle/working")
CHAMP = WORK / "mnemonics-champion"
RESULTS = WORK / "results"
RESULTS.mkdir(exist_ok=True)


def checked(cmd, **kwargs):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


subprocess.run(["rm", "-rf", str(CHAMP)], check=False)
checked(["git", "clone", "https://github.com/nakata-app/mnemonics.git", str(CHAMP)])
checked(["git", "-C", str(CHAMP), "checkout", "--detach", CHAMP_SHA])
head = subprocess.check_output(
    ["git", "-C", str(CHAMP), "rev-parse", "--short", "HEAD"], text=True
).strip()
if head != CHAMP_SHA:
    raise RuntimeError(f"wrong champion checkout: {head}")

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
    raise RuntimeError("longmemeval_s_cleaned.json not found")

questions = json.loads(data.read_text())
if len(questions) != 500:
    raise RuntimeError(f"expected 500 questions, got {len(questions)}")
if OUTER_HALF not in (0, 1):
    raise RuntimeError(f"OUTER_HALF must be 0 or 1, got {OUTER_HALF}")

selected = questions[:250] if OUTER_HALF == 0 else questions[250:]
selected_ids = [q["question_id"] for q in selected]

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
print(f"OUTER_HALF={OUTER_HALF} CUDA_DEVICE_COUNT={gpu_count}", flush=True)
if gpu_count < 1:
    raise RuntimeError("no CUDA GPU")

# Prime HF cache once so parallel workers do not race on model downloads.
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


def make_cmd(split: Path, cands: Path, out: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        "benchmarks/longmemeval_eval.py",
        "--split-file",
        str(split),
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


worker_count = min(gpu_count, 2)
parts = [selected]
if worker_count == 2:
    parts = [selected[:125], selected[125:]]

specs = []
for worker, subset in enumerate(parts):
    split = RESULTS / f"half{OUTER_HALF}_worker{worker}_split.json"
    split.write_text(json.dumps({"dev": [q["question_id"] for q in subset]}))
    cands = RESULTS / f"half{OUTER_HALF}_worker{worker}_cands.json"
    out = RESULTS / f"half{OUTER_HALF}_worker{worker}_run.json"
    specs.append((worker, subset, split, cands, out))

procs = []
for worker, _subset, split, cands, out in specs:
    worker_env = env.copy()
    worker_env["CUDA_VISIBLE_DEVICES"] = str(worker)
    cmd = make_cmd(split, cands, out)
    print(f"START HALF={OUTER_HALF} WORKER={worker}: {' '.join(cmd)}", flush=True)
    procs.append((worker, subprocess.Popen(cmd, cwd=CHAMP, env=worker_env)))

failures = []
for worker, proc in procs:
    rc = proc.wait()
    print(f"HALF={OUTER_HALF} WORKER={worker} EXIT={rc}", flush=True)
    if rc:
        failures.append((worker, rc))
if failures:
    raise RuntimeError(f"worker failure(s): {failures}")

merged = []
for worker, subset, _split, cands, _out in specs:
    rows = json.loads(cands.read_text())
    if len(rows) != len(subset):
        raise RuntimeError(
            f"half {OUTER_HALF} worker {worker}: expected {len(subset)}, got {len(rows)}"
        )
    merged.extend(rows)

merged_path = RESULTS / f"cands_ca53594_half{OUTER_HALF}.json"
merged_path.write_text(json.dumps(merged))
candidate_ids = [row["qid"] for row in merged]
if candidate_ids != selected_ids:
    raise RuntimeError(f"half {OUTER_HALF}: candidate QID order mismatch")

provenance = {
    "candidate_source_commit": CHAMP_SHA,
    "outer_half": OUTER_HALF,
    "question_count": len(merged),
    "first_qid": candidate_ids[0],
    "last_qid": candidate_ids[-1],
    "candidate_sha256": hashlib.sha256(merged_path.read_bytes()).hexdigest(),
    "ce_model": CE_MODEL,
    "deterministic": True,
    "cuda_device_count": gpu_count,
    "internal_worker_count": worker_count,
    "config": "--split-file exact-QIDs --mode rerank --chunk-mode turn --augment-preferences --candidate-k 50 --seed 42",
}
(RESULTS / f"provenance_half{OUTER_HALF}.json").write_text(
    json.dumps(provenance, indent=2)
)
print("HALF_COMPLETE", json.dumps(provenance, sort_keys=True), flush=True)
