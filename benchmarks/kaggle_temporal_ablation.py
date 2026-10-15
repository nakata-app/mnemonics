"""Kaggle T4 runner for honest temporal ablation at mnemonics 33628f1."""
import json
import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "mnemonics"
RESULTS = WORK / "results"
RESULTS.mkdir(exist_ok=True)
SHA = "5510154"
FULL_SHA = "5510154"

subprocess.run(["rm", "-rf", str(REPO)], check=False)
subprocess.run(["git", "clone", "https://github.com/nakata-app/mnemonics.git", str(REPO)], check=True)
subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", SHA], check=True)
head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
if head != FULL_SHA:
    raise RuntimeError(f"wrong checkout: {head}")

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO),
    "sentence-transformers", "numpy", "adaptmem"
], check=True)

data = None
for root, _, files in os.walk("/kaggle/input"):
    if "longmemeval_s_cleaned.json" in files:
        data = Path(root) / "longmemeval_s_cleaned.json"
        break
if data is None:
    raise RuntimeError("longmemeval_s_cleaned.json not found in mounted Kaggle dataset")

env = os.environ.copy()
env["MNEMONICS_DETERMINISTIC"] = "1"
env["MNEMONICS_RERANK_MODEL"] = "BAAI/bge-reranker-v2-m3"
env["MNEMONICS_RERANK_BACKEND"] = "sentence-transformers"
env["MNEMONICS_EMBED_BACKEND"] = "sentence-transformers"
env["PYTHONUNBUFFERED"] = "1"
env["LME_DATA"] = str(data)

cands = RESULTS / "cands.json"
run = RESULTS / "run.json"
ablation = RESULTS / "ablation.json"

cmd = [
    sys.executable, "-u", "benchmarks/longmemeval_eval.py",
    "--n", "500",
    "--mode", "rerank",
    "--chunk-mode", "turn",
    "--augment-preferences",
    "--candidate-k", "50",
    "--seed", "42",
    "--dump-candidates", str(cands),
    "--out", str(run),
]
print("RUN:", " ".join(cmd), flush=True)
subprocess.run(cmd, cwd=REPO, env=env, check=True)

subprocess.run([
    sys.executable, "-u", "benchmarks/temporal_ablation_offline.py",
    "--candidates", str(cands),
    "--data", str(data),
    "--out", str(ablation),
], cwd=REPO, env=env, check=True)

res = json.loads(ablation.read_text())
r = res["variants"]["on_labeled"]["R@1"]
print(f"ON_LABELED_R1={r}", flush=True)
if abs(r - 0.958) > 1e-12:
    raise RuntimeError(f"champion reproduction failed: expected 0.958, got {r}")
print("CHAMPION_REPRO_OK", flush=True)
print(json.dumps(res, indent=2), flush=True)
