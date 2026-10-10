import glob, json, re, time, os
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

IN = Path(sorted(glob.glob("/kaggle/input/**/locomo10.json", recursive=True), key=len)[0]).parent
OUT = Path("/kaggle/working/results"); OUT.mkdir(exist_ok=True)
torch.set_num_threads(os.cpu_count() or 4)
data = json.load(open(IN / "locomo10.json"))

chunks, owner = [], []
sess_conv = {}
for c in data:
    cv = c["conversation"]
    for k in cv:
        m = re.fullmatch(r"session_(\d+)", k)
        if m and cv[k]:
            sid = f'{c["sample_id"]}_s{m.group(1)}'
            sess_conv[sid] = c["sample_id"]
            lines = [f'{t["speaker"]}: {t["text"]}' for t in cv[k]]
            for j in range(0, len(lines), 2):
                chunks.append("\n".join(lines[j:j + 2])); owner.append(sid)
sids = sorted(sess_conv)
qs = []
for c in data:
    for qa in c["qa"]:
        if qa.get("category") == 5:
            continue
        nums = {int(m.group(1)) for e in qa.get("evidence", []) for m in [re.match(r"D(\d+):", e)] if m}
        gold = {f'{c["sample_id"]}_s{n}' for n in nums} & set(sids)
        if gold:
            qs.append((c["sample_id"], qa["question"], gold))
print("chunks", len(chunks), "sessions", len(sids), "questions", len(qs), flush=True)

MODELS = {"minilm": ("sentence-transformers/all-MiniLM-L6-v2", None),
          "qwen3-emb-0.6b": ("Qwen/Qwen3-Embedding-0.6B", "query")}
owner_arr = np.array([sids.index(o) for o in owner])
summary = {"chunks": len(chunks), "sessions": len(sids), "questions": len(qs)}
for name, (mid, qprompt) in MODELS.items():
    t0 = time.time()
    m = SentenceTransformer(mid, device="cpu")
    E = m.encode(chunks, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    Q = m.encode([q for _, q, _ in qs], batch_size=32, normalize_embeddings=True, convert_to_numpy=True,
                 show_progress_bar=False, **({"prompt_name": qprompt} if qprompt else {}))
    print(name, "encoded", f"{time.time() - t0:.0f}s", flush=True)
    sims = Q @ E.T
    res = {}
    for mode in ("own_conv", "all_convs"):
        hits = {k: 0 for k in (1, 5, 10, 30)}
        for qi, (conv, _, gold) in enumerate(qs):
            best = np.full(len(sids), -1e9)
            np.maximum.at(best, owner_arr, sims[qi])
            if mode == "own_conv":
                mask = np.array([sess_conv[s] == conv for s in sids]); best = np.where(mask, best, -1e9)
            order = np.argsort(-best)
            for k in hits:
                hits[k] += any(sids[i] in gold for i in order[:k])
        res[mode] = {f"R@{k}": round(v / len(qs), 4) for k, v in hits.items()}
    summary[name] = res
    print(name, json.dumps(res), flush=True)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print("RESULT=" + json.dumps(summary), flush=True)
