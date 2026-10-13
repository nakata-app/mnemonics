"""Kaggle GPU kernel: score make_rerank_docs.py documents with an open-weights Qwen3-Reranker, deterministically.

Model weights are pinned by HF revision. fp16, fixed batch order (docs sorted by length within a question),
deterministic torch flags. The first `REPEAT_CHECK` questions are scored twice and the max absolute score
difference is printed, so determinism is measured, not assumed. Output: scores.jsonl, {"qid", "scores"} aligned
with the input docs.
"""
import glob
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("RR_MODEL", "Qwen/Qwen3-Reranker-4B")
REVISION = os.environ.get("RR_REVISION", "22e683669bc0f0bd69640a1354a6d0aebcfeede5")
DOCS_NAME = os.environ.get("RR_DOCS", "rerank_docs.jsonl")
INSTRUCT = ("Given a question a user asks about their own past conversations, judge whether this conversation "
            "excerpt contains information needed to answer the question")
BATCH, MAX_LEN, REPEAT_CHECK = 8, 2048, 20

torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
IN = Path(sorted(glob.glob(f"/kaggle/input/**/{DOCS_NAME}", recursive=True), key=len)[0])
OUT = Path("/kaggle/working")

tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
if os.environ.get("RR_QUANT") == "int8":
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=True)
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, device_map="cuda:0", torch_dtype=torch.float16,
                                                 quantization_config=BitsAndBytesConfig(load_in_8bit=True)).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float16).cuda().eval()
yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
prefix = tok.encode('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
                    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
                    add_special_tokens=False)
suffix = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)


@torch.no_grad()
def score(question, texts):
    pairs = [f"<Instruct>: {INSTRUCT}\n<Query>: {question}\n<Document>: {t}" for t in texts]
    enc = tok(pairs, padding=False, truncation="longest_first", return_attention_mask=False,
              max_length=MAX_LEN - len(prefix) - len(suffix))
    ids = [prefix + e + suffix for e in enc["input_ids"]]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out = [0.0] * len(ids)
    for b0 in range(0, len(order), BATCH):
        idx = order[b0:b0 + BATCH]
        batch = tok.pad({"input_ids": [ids[i] for i in idx]}, padding=True, return_tensors="pt").to("cuda")
        logits = model(**batch, logits_to_keep=1).logits[:, -1, :]
        lp = torch.log_softmax(torch.stack([logits[:, no], logits[:, yes]], dim=1).float(), dim=1)[:, 1].exp()
        for i, v in zip(idx, lp.tolist()):
            out[i] = v
    return out


rows = [json.loads(l) for l in open(IN) if l.strip()]
print("model", MODEL, REVISION, os.environ.get("RR_QUANT", "fp16"), "| questions", len(rows), "| docs", sum(len(r["docs"]) for r in rows), flush=True)
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
