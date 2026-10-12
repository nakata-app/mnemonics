from __future__ import annotations

import glob
import hashlib
import shutil
import tarfile
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "m_origv2_m"
RESULTS = WORK / "results_origv2pool_m"
RESULTS.mkdir(parents=True, exist_ok=True)
SHA = "ca5359417c9a4692d86e58bc36253e00b54172f8"


def retry(fn, w, tries=3):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            print("retry", i + 1, w, e, flush=True)
            if i == tries - 1:
                raise
            time.sleep(8 * (i + 1))


def resolve(name, isdir):
    xs = [
        x
        for x in glob.glob(f"/kaggle/input/**/{name}", recursive=True)
        if (os.path.isdir(x) if isdir else os.path.isfile(x))
    ]
    if not xs:
        raise RuntimeError(name)
    return sorted(xs, key=len)[0]


def clone():
    subprocess.run(["rm", "-rf", str(REPO)], check=False)
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "https://github.com/nakata-app/mnemonics.git",
            str(REPO),
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", SHA], check=True)
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if head != SHA:
        raise RuntimeError("sha drift " + head)


retry(clone, "clone")
p = REPO / "benchmarks" / "longmemeval_eval.py"
s = p.read_text()

# Gold-label ordinal branch OFF: productionizable label-free temporal path only.
old = 'elif sid_to_date and q.get("question_type") == "temporal-reasoning":'
if s.count(old) != 1:
    raise RuntimeError("ordinal anchor")
s = s.replace(
    old, 'elif False and sid_to_date and q.get("question_type") == "temporal-reasoning":', 1
)

# Stream the 2.5 GB JSON array so Colab RAM holds only one question at a time.
loader_old = '    all_q = json.load(open(DATA))'
loader_new = r'''    class _StreamingQuestions:
        def __init__(self, path, expected=500):
            self.path = path
            self.expected = expected

        def __len__(self):
            return self.expected

        def __iter__(self):
            decoder = json.JSONDecoder()
            count = 0
            with open(self.path, "r", encoding="utf-8") as source:
                buffer = ""
                started = False
                eof = False
                while True:
                    buffer = buffer.lstrip()
                    if not buffer and not eof:
                        part = source.read(1024 * 1024)
                        eof = not part
                        buffer += part
                        continue
                    if not started:
                        if not buffer.startswith("["):
                            raise RuntimeError("expected JSON array")
                        buffer = buffer[1:]
                        started = True
                        continue
                    if buffer.startswith(","):
                        buffer = buffer[1:]
                        continue
                    if buffer.startswith("]"):
                        if count != self.expected:
                            raise RuntimeError(f"expected {self.expected} questions, got {count}")
                        return
                    try:
                        question, used = decoder.raw_decode(buffer)
                    except json.JSONDecodeError:
                        if eof:
                            raise
                        part = source.read(1024 * 1024)
                        eof = not part
                        buffer += part
                        continue
                    count += 1
                    buffer = buffer[used:]
                    yield question

    all_q = _StreamingQuestions(DATA)'''
if s.count(loader_old) != 1:
    raise RuntimeError("streaming loader anchor mismatch")
s = s.replace(loader_old, loader_new, 1)


# CLEAN2: anonymize session ids so "answer_" never appears in indexed/scored text.
loop_anchor = "    for i, q in enumerate(questions):\n"
if s.count(loop_anchor) != 1:
    raise RuntimeError("loop anchor")
s = s.replace(loop_anchor, loop_anchor + """        import hashlib as _hl
        _anon = lambda _x: "s" + _hl.sha1(_x.encode()).hexdigest()[:12]
        q = dict(q)
        q["haystack_session_ids"] = [_anon(_x) for _x in (q.get("haystack_session_ids") or [])]
        q["answer_session_ids"] = [_anon(_x) for _x in (q.get("answer_session_ids") or [])]
""", 1)
print("SID_ANONYMIZED", flush=True)


# ORIG-V2: return enough chunks that >=10 distinct sessions exist, so R@10 is actually measured.
_tk_old = "def evaluate_mnemonics(questions: list[dict], rerank: bool, top_k: int = 10,"
if s.count(_tk_old) != 1:
    raise RuntimeError("top_k anchor")
s = s.replace(_tk_old, "def evaluate_mnemonics(questions: list[dict], rerank: bool, top_k: int = 50,", 1)
pq = '                    "retrieved_top10_sids": retrieved_sids[:10],\n'
if s.count(pq) != 1:
    raise RuntimeError("perq anchor")
s = s.replace(pq, pq + '                    "retrieved_all_sids": list(retrieved_sids),\n', 1)
p.write_text(s)
print("POLICY=ORIG-V2: generic engineering choices (turn chunks, candidate_k 50, top_k 50); no lanes, no temporal flags, no gate (BM25+MiniLM+RRF, off-the-shelf ms-marco CE); no lanes, no temporal flags, no gate, anonymized SIDs", flush=True)

retry(
    lambda: subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-e",
            str(REPO),
            "sentence-transformers==5.4.0",
            "numpy",
            "adaptmem",
        ],
        check=True,
    ),
    "pip",
)
DATA_PATH = Path("/tmp/longmemeval_m_cleaned.json")
if not DATA_PATH.exists():
    url = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json"
    print("DOWNLOADING_LONGMEMEVAL_M", url, flush=True)
    for _try in range(6):  # resume + retries; parallel shards each pull the file
        _r = subprocess.run(["curl", "-L", "--fail", "-C", "-", "--retry", "5", "--retry-delay", "10", "-o", str(DATA_PATH), url])
        if _r.returncode == 0 and DATA_PATH.stat().st_size == 2737100077:
            break
        print("download retry", _try, _r.returncode, flush=True)
        time.sleep(15)
    else:
        raise RuntimeError("dataset download failed")
DATA = str(DATA_PATH)
RERANK = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # CLEAN2: off-the-shelf, never trained on LongMemEval
env = os.environ.copy()
env.update(
    {
        "LME_DATA": DATA,
        "PYTHONUNBUFFERED": "1",
        "MNEMONICS_RERANK_MODEL": RERANK,
        "MNEMONICS_DETERMINISTIC": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    }
)
args = [
    "--mode",
    "rerank",
    "--chunk-mode",
    "turn",
    "--candidate-k",
    "50",
    "--seed",
    "42",
]

SPLIT = WORK / "split_shard2.json"
SPLIT.write_text(json.dumps({"dev": ["0100672e", "07b6f563", "0a34ad58", "0ea62687", "15745da0_abs", "1a1907b4", "1e043500", "2318644b", "29f2956b_abs", "32260d93", "3a704032", "41275add", "4bc144e2", "51a45a95", "5831f84d", "60036106", "6222b6eb", "6a27ffc2", "6e984301", "720133ac", "76d63226", "81507db6", "87f22b4a", "8b9d4367", "945e3d21", "9bbe84a2", "a3045048", "a9f6b44c", "b29f3365", "b9cfe692", "bf659f65", "c7cf7dfd", "cc06de0d", "d24813b1", "dad224aa", "e3fc4d6e", "e831120c", "edced276", "f4f1d8a4", "fea54f57", "gpt4_1e4a8aeb", "gpt4_2f56ae70", "gpt4_468eb064", "gpt4_59149c77", "gpt4_70e84552", "gpt4_7ddcf75f", "gpt4_93159ced", "gpt4_b0863698", "gpt4_d6585ce9", "gpt4_f2262a51"]}))
out = RESULTS / "run.json"
perq = RESULTS / "perq.json"
rc = subprocess.run(
    [
        sys.executable,
        "-u",
        "benchmarks/longmemeval_eval.py",
        "--n",
        "500",
        "--split-file",
        str(SPLIT),
        *args,
        "--out",
        str(out),
        "--per-q-out",
        str(perq),
    ],
    cwd=REPO,
    env=env,
)
if rc.returncode:
    raise SystemExit(rc.returncode)
d = json.loads(out.read_text())["mnemonics_rerank"]
summary = {
    "shard": 2,
    "n": d.get("n"),
    "source_sha": SHA,
    "R@1": d["R@1"],
    "R@5": d["R@5"],
    "R@10": d["R@10"],
    "trust_gate": None,
    "reranker": RERANK,
    "policy": "ORIG-V2: generic engineering choices (turn chunks, candidate_k 50, top_k 50); no lanes, no temporal flags, no gate; no lanes, no temporal flags, no gate, anonymized SIDs",
}
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
print("RESULT=" + json.dumps(summary, sort_keys=True), flush=True)
print("M_QUERY_LANES_COMPLETE", flush=True)
