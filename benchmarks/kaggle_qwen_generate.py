"""Kaggle GPU kernel: deterministic greedy generation for judge prompts with an open-weights Qwen3 LLM.

Input: prompts.jsonl ({qid, prompt}) from llm_judge_rerank.py --dump-prompts. Output: raw.jsonl ({qid, raw}) for
llm_judge_rerank.py --replay-raw. Weights pinned by HF commit, 4-bit NF4 (fits one T4), greedy decoding, thinking
off, deterministic torch flags. The first REPEAT_CHECK prompts are generated twice; the count of mismatching
outputs is printed, so determinism is measured, not assumed.
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=True)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = os.environ.get("GEN_MODEL", "Qwen/Qwen3-14B")
REVISION = os.environ.get("GEN_REVISION", "40c069824f4251a91eefaf281ebe4c544efd3e18")
PROMPTS = os.environ.get("GEN_PROMPTS", "prompts.jsonl")
MAX_NEW, REPEAT_CHECK, PREFILL_CHUNK = int(os.environ.get("GEN_MAX_NEW", "48")), 10, 1024
THINK = os.environ.get("GEN_THINK") == "1"

torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
IN = Path(sorted(glob.glob(f"/kaggle/input/**/{PROMPTS}", recursive=True), key=len)[0])
OUT = Path("/kaggle/working")

tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, revision=REVISION, device_map="cuda:0", attn_implementation="sdpa",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.float16)).eval()


eos = model.generation_config.eos_token_id
EOS = set(eos if isinstance(eos, list) else [eos])


@torch.no_grad()
def generate(prompt):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=THINK)
    ids = tok(text, return_tensors="pt").input_ids.to("cuda:0")
    out = None
    for c0 in range(0, ids.shape[1], PREFILL_CHUNK):
        out = model(input_ids=ids[:, c0:c0 + PREFILL_CHUNK], past_key_values=out.past_key_values if out else None,
                    use_cache=True, logits_to_keep=1)
    toks = []
    for _ in range(MAX_NEW):
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        if nxt.item() in EOS:
            break
        toks.append(nxt.item())
        out = model(input_ids=nxt, past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
    del out
    torch.cuda.empty_cache()
    text = tok.decode(toks, skip_special_tokens=True)
    if THINK:
        text = text.split("</think>", 1)[1] if "</think>" in text else ""
    return text.strip()


rows = [json.loads(l) for l in open(IN) if l.strip()]
print("model", MODEL, REVISION, "| prompts", len(rows), flush=True)
t0, mismatch = time.time(), 0
with open(OUT / "raw.jsonl", "w") as f:
    for i, r in enumerate(rows):
        raw = generate(r["prompt"])
        if i < REPEAT_CHECK:
            mismatch += generate(r["prompt"]) != raw
        f.write(json.dumps({"qid": r["qid"], "raw": raw}) + "\n")
        f.flush()
        if i % 25 == 0:
            print(i, f"{time.time() - t0:.0f}s", repr(raw[:40]), flush=True)
print(f"DONE {len(rows)} in {time.time() - t0:.0f}s | repeat-check mismatches {mismatch}/{min(REPEAT_CHECK, len(rows))}",
      flush=True)
