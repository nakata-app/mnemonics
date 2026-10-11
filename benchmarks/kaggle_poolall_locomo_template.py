from __future__ import annotations

import glob
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

SHARD, NSHARD, NSAMPLE, SEED = __SHARD__, __NSHARD__, 200, 11
WORK = Path("/kaggle/working")
REPO = WORK / "repo"
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
SHA = "ca5359417c9a4692d86e58bc36253e00b54172f8"

subprocess.run(["git", "clone", "--filter=blob:none", "https://github.com/nakata-app/mnemonics.git", str(REPO)], check=True)
subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", SHA], check=True)
assert subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip() == SHA

p = REPO / "benchmarks" / "longmemeval_eval.py"
s = p.read_text()
old = 'elif sid_to_date and q.get("question_type") == "temporal-reasoning":'
assert s.count(old) == 1
s = s.replace(old, 'elif False and sid_to_date and q.get("question_type") == "temporal-reasoning":', 1)
loop_anchor = "    for i, q in enumerate(questions):\n"
assert s.count(loop_anchor) == 1
s = s.replace(loop_anchor, loop_anchor + """        import hashlib as _hl
        _anon = lambda _x: "s" + _hl.sha1(_x.encode()).hexdigest()[:12]
        q = dict(q)
        q["haystack_session_ids"] = [_anon(_x) for _x in (q.get("haystack_session_ids") or [])]
        q["answer_session_ids"] = [_anon(_x) for _x in (q.get("answer_session_ids") or [])]
""", 1)
tk = "def evaluate_mnemonics(questions: list[dict], rerank: bool, top_k: int = 10,"
assert s.count(tk) == 1
s = s.replace(tk, "def evaluate_mnemonics(questions: list[dict], rerank: bool, top_k: int = 50,", 1)
pq = '                    "retrieved_top10_sids": retrieved_sids[:10],\n'
assert s.count(pq) == 1
s = s.replace(pq, pq + '                    "retrieved_all_sids": list(retrieved_sids),\n', 1)
p.write_text(s)

r = REPO / "mnemonics" / "retrieve.py"
rs = r.read_text()
qa = "    qvec = enc.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]\n"
assert rs.count(qa) == 1
rs = rs.replace(qa, "    _qp = {\"prompt_name\": \"query\"} if \"query\" in (getattr(enc, \"prompts\", None) or {}) else {}\n"
                "    qvec = enc.encode([query], normalize_embeddings=True, convert_to_numpy=True, **_qp)[0]\n", 1)
r.write_text(rs)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO), "sentence-transformers==5.4.0", "numpy", "adaptmem"], check=True)

src = sorted(glob.glob("/kaggle/input/**/locomo10.json", recursive=True), key=len)[0]
data = json.load(open(src))


def lme_date(x):
    return datetime.strptime(x.strip(), "%I:%M %p on %d %B, %Y").strftime("%Y/%m/%d (%a) %H:%M")


qs = []
ALL = []
for smp in data:
    for k in smp["conversation"]:
        m = re.fullmatch(r"session_(\d+)", k)
        if m and smp["conversation"][k]:
            n = int(m.group(1))
            ALL.append((f'{smp["sample_id"]}_s{n}', n, lme_date(smp["conversation"][f"session_{n}_date_time"]),
                        [{"role": "user", "content": f'{t["speaker"]}: {t["text"]}'} for t in smp["conversation"][k]]))
for smp in data:
    conv = smp["conversation"]
    own = []
    for k in conv:
        m = re.fullmatch(r"session_(\d+)", k)
        if m and conv[k]:
            n = int(m.group(1))
            own.append((f'{smp["sample_id"]}_s{n}', n, lme_date(conv[f"session_{n}_date_time"]),
                        [{"role": "user", "content": f'{t["speaker"]}: {t["text"]}'} for t in conv[k]]))
    own.sort(key=lambda x: x[1])
    last = max(datetime.strptime(d, "%Y/%m/%d (%a) %H:%M") for _, _, d, _ in own)
    qdate = (last + timedelta(days=1)).strftime("%Y/%m/%d (%a) %H:%M")
    for i, qa_ in enumerate(smp["qa"]):
        if qa_.get("category") == 5:
            continue
        nums = {int(mm.group(1)) for e in qa_.get("evidence", []) for mm in [re.match(r"D(\d+):", e)] if mm}
        gold = [sid for sid, n, _, _ in own if n in nums]
        if gold:
            qs.append({"question_id": f'{smp["sample_id"]}_q{i}', "question_type": f'locomo-cat{qa_["category"]}',
                       "question": qa_["question"], "answer": str(qa_.get("answer", "")), "question_date": qdate,
                       "haystack_session_ids": [x[0] for x in ALL], "haystack_dates": [x[2] for x in ALL],
                       "haystack_sessions": [x[3] for x in ALL], "answer_session_ids": gold})
sample = random.Random(SEED).sample(qs, NSAMPLE)[SHARD::NSHARD]
DATA = WORK / "locomo_shard.json"
DATA.write_text(json.dumps(sample))
print("QUESTIONS", len(qs), "SHARD", SHARD, len(sample), flush=True)

ARMS = {"minilm": {}}
summary = {"shard": SHARD, "n": len(sample), "source_sha": SHA,
           "policy": "ORIG-V2 (turn chunks, cand 50, top_k 50, no lanes, ordinal off, anon SIDs, ms-marco CE) on LoCoMo all-10-conv haystack (272 sessions); query prompt when encoder defines one"}
for arm, extra in ARMS.items():
    env = os.environ.copy()
    env.update({"LME_DATA": str(DATA), "PYTHONUNBUFFERED": "1", "MNEMONICS_RERANK_MODEL": "cross-encoder/ms-marco-MiniLM-L-12-v2",
                "MNEMONICS_DETERMINISTIC": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", **extra})
    out, perq = RESULTS / f"{arm}_run.json", RESULTS / f"{arm}_perq.json"
    rc = subprocess.run([sys.executable, "-u", "benchmarks/longmemeval_eval.py", "--n", str(len(sample)), "--mode", "rerank",
                         "--chunk-mode", "turn", "--candidate-k", "50", "--seed", "42", "--out", str(out), "--per-q-out", str(perq)],
                        cwd=REPO, env=env)
    if rc.returncode:
        raise SystemExit(rc.returncode)
    d = json.loads(out.read_text())["mnemonics_rerank"]
    summary[arm] = {k: d[k] for k in ("R@1", "R@5", "R@10")}
    print("ARM", arm, json.dumps(summary[arm]), flush=True)
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
print("RESULT=" + json.dumps(summary), flush=True)
