"""Kaggle GPU kernel: score make_rerank_docs.py documents with Qwen3-VL-Reranker (the model evren's
qwen3-reranker-8b endpoint routes to, per its usage field), deterministically, via sentence-transformers.

Weights pinned by HF commit, 8-bit so the 8B model fits one T4, fixed batch order, deterministic torch flags,
the model's default prompt (the cloud runs sent no instruction). The first REPEAT_CHECK questions are scored
twice and the max |diff| is printed. Output: scores.jsonl, {"qid", "scores"} aligned with the input docs.
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "sentence-transformers>=5.1", "transformers>=4.57.0",
                "qwen-vl-utils>=0.0.14", "bitsandbytes"], check=True)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from sentence_transformers import CrossEncoder
from transformers import BitsAndBytesConfig

MODEL = os.environ.get("RR_MODEL", "Qwen/Qwen3-VL-Reranker-8B")
REVISION = os.environ.get("RR_REVISION", "b212dc8c91a8164aef1ea2de9c1a867611e75c04")
DOCS_NAME = os.environ.get("RR_DOCS", "rerank_docs.jsonl")
BATCH, REPEAT_CHECK = 4, 20

torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
IN = Path(sorted(glob.glob(f"/kaggle/input/**/{DOCS_NAME}", recursive=True), key=len)[0])
OUT = Path("/kaggle/working")
ce = CrossEncoder(MODEL, revision=REVISION, device="cuda",
                  model_kwargs={"torch_dtype": torch.float16,
                                "quantization_config": BitsAndBytesConfig(load_in_8bit=True)})


def score(question, texts):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vals = ce.predict([(question, texts[i]) for i in order], batch_size=BATCH, show_progress_bar=False)
    out = [0.0] * len(texts)
    for i, v in zip(order, vals):
        out[i] = float(v)
    return out


rows = [json.loads(l) for l in open(IN) if l.strip()]
print("model", MODEL, REVISION, "| questions", len(rows), "| docs", sum(len(r["docs"]) for r in rows), flush=True)
t0, maxdiff, nan = time.time(), 0.0, 0
with open(OUT / "scores.jsonl", "w") as f:
    for qi, r in enumerate(rows):
        s = score(r["question"], [d["text"] for d in r["docs"]]) if r["docs"] else []
        nan += sum(v != v for v in s)
        if qi < REPEAT_CHECK and s:
            s2 = score(r["question"], [d["text"] for d in r["docs"]])
            maxdiff = max(maxdiff, max(abs(a - b) for a, b in zip(s, s2)))
        f.write(json.dumps({"qid": r["qid"], "scores": s}) + "\n")
        if qi % 25 == 0:
            print(qi, f"{time.time() - t0:.0f}s", flush=True)
print(f"DONE {len(rows)} q in {time.time() - t0:.0f}s | repeat-check max |diff| over first {REPEAT_CHECK} q = "
      f"{maxdiff:.3g} | NaN scores {nan}", flush=True)
