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
RESULTS = WORK / "results_origv2deep_m"
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
s = s.replace(_tk_old, "def evaluate_mnemonics(questions: list[dict], rerank: bool, top_k: int = 300,", 1)
pq = '                    "retrieved_top10_sids": retrieved_sids[:10],\n'
if s.count(pq) != 1:
    raise RuntimeError("perq anchor")
s = s.replace(pq, pq + '                    "retrieved_all_sids": list(retrieved_sids),\n', 1)
p.write_text(s)
print("POLICY=ORIG-V2: generic engineering choices (turn chunks, candidate_k 300, top_k 300); no lanes, no temporal flags, no gate (BM25+MiniLM+RRF, off-the-shelf ms-marco CE); no lanes, no temporal flags, no gate, anonymized SIDs", flush=True)

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
    "300",
    "--seed",
    "42",
]

SPLIT = WORK / "split_deep0.json"
SPLIT.write_text(json.dumps({"dev": ["001be529", "0100672e", "031748ae", "06878be2", "06f04340", "07741c45", "07b6f563", "0862e8bf_abs", "08f4fc43", "099778bb", "09ba9854_abs", "0a34ad58", "0bb5a684", "0bc8ad93", "0ddfec37", "0e4e4c46", "0ea62687", "0f05491a", "10e09553", "1192316e", "1568498a", "15745da0_abs", "16c90bf4", "18bc8abd", "1903aded", "19b5f2b3", "1a1907b4", "1b9b7252", "1c549ce4", "1d4da289", "1da05512", "1e043500", "1faac195", "2133c1b5_abs", "21d02d0d", "2311e44b", "2318644b", "2698e78f", "26bdc477", "2788b940", "28dc39ac", "29f2956b_abs", "2b8f3739", "2c63a862", "2e6d26dc", "2ebe6c92", "32260d93", "352ab8bd", "36580ce8", "370a8ff4", "37f165cf", "3a704032", "3ba21379", "3d86fd0a", "3f1e9474", "3fe836c9", "41275add", "42ec0761", "45dc21b6", "488d3006", "4b24c848", "4bc144e2", "4d6b87c8", "4dfccbf8", "4fd1909e", "505af2f5", "51a45a95", "51c32626", "545bd2b5", "561fabcd", "57f827a0", "5831f84d", "58bf7951", "59524333", "5a7937c8", "5d3d2817", "60036106", "603deb26", "6071bd76", "60bf93ed_abs", "618f13b2", "6222b6eb", "6456829e_abs", "6613b389", "67e0d0f2", "69fee5aa", "6a27ffc2", "6ae235be", "6aeb4375_abs", "6b7dfb22", "6cb6f249", "6e984301", "6f9b354f", "70b3e69b", "71017277", "7161e7e2", "720133ac", "72e3ee87", "7401057b", "7527f7e2", "75832dbd", "76d63226", "77eafa52", "7a8d0b71", "7e974930", "80ec1f4f", "81507db6", "8464fc84", "853b0a1d", "85fa3a3f", "86f00804", "87f22b4a", "88432d0a_abs", "8979f9ec", "89941a94", "8a2466db", "8b9d4367", "8cf4d046", "8e91e7d9", "8ebdbe50", "91b15a6e", "945e3d21", "95228167", "982b5123", "993da5e2", "9a707b82", "9bbe84a2", "9ea5eabc", "a06e4cfe", "a11281a2", "a1eacc2a", "a3045048", "a346bb18", "a40e080f", "a82c026e", "a96c20ee", "a9f6b44c", "ac031881", "af082822", "afdc33df", "b01defab", "b29f3365", "b3c15d39", "b46e15ee", "b6019101", "b759caee", "b9cfe692", "ba358f49_abs", "bb7c3b45", "bc149d6b", "bc8a6e93_abs", "bf659f65", "c18a7dc8", "c2ac3c61", "c4ea545c", "c5e8278d", "c7cf7dfd", "c8090214", "c8c3f81d", "c960da58", "caf03d32", "cc06de0d", "cc5ded98", "ccb36322", "ceb54acb", "d01c6aa8", "d24813b1", "d52b4f67", "d6062bb9", "d682f1a2", "d851d5ba", "dad224aa", "dc439ea3", "dcfa8644", "dfde3500", "e25c3b8d", "e3fc4d6e", "e48988bc", "e4e14d04", "e5ba910e_abs", "e61a7584", "e831120c", "e9327a54", "eac54adc", "eaca4986", "ec81a493", "edced276", "eeda8a6d", "ef66a6e5", "efc3f7c2", "f0e564bc", "f4f1d8a4", "f523d9fe", "f685340e_abs", "f9e8c073", "fca70973", "fea54f57", "gpt4_0b2f1d21", "gpt4_18c2b244", "gpt4_194be4b3", "gpt4_1d4ab0c9", "gpt4_1e4a8aeb", "gpt4_213fd887", "gpt4_2312f94c", "gpt4_2655b836", "gpt4_2c50253f", "gpt4_2f56ae70", "gpt4_2f8be40d", "gpt4_31ff4165", "gpt4_372c3eed_abs", "gpt4_45189cb4", "gpt4_468eb064", "gpt4_4929293a", "gpt4_4cd9eba1", "gpt4_4ef30696", "gpt4_5438fa52", "gpt4_59149c77", "gpt4_59c863d7", "gpt4_61e13b3c", "gpt4_68e94287", "gpt4_6dc9b45b", "gpt4_70e84552", "gpt4_731e37d7", "gpt4_76048e76", "gpt4_7a0daae1", "gpt4_7bc6cf22", "gpt4_7ddcf75f", "gpt4_7f6b06db", "gpt4_8279ba02", "gpt4_85da3956", "gpt4_8c8961ae", "gpt4_93159ced", "gpt4_93f6379c", "gpt4_9a159967", "gpt4_a2d1d1f6", "gpt4_ab202e7f", "gpt4_b0863698", "gpt4_b5700ca0", "gpt4_c27434e8", "gpt4_cd90e484", "gpt4_d31cdae3", "gpt4_d6585ce9", "gpt4_d9af6064", "gpt4_e061b84f", "gpt4_e072b769", "gpt4_e414231f", "gpt4_f2262a51", "gpt4_f420262d", "gpt4_fa19884c", "gpt4_fe651585"]}))
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
    "shard": "deep0",
    "n": d.get("n"),
    "source_sha": SHA,
    "R@1": d["R@1"],
    "R@5": d["R@5"],
    "R@10": d["R@10"],
    "trust_gate": None,
    "reranker": RERANK,
    "policy": "ORIG-V2: generic engineering choices (turn chunks, candidate_k 300, top_k 300); no lanes, no temporal flags, no gate; no lanes, no temporal flags, no gate, anonymized SIDs",
}
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
print("RESULT=" + json.dumps(summary, sort_keys=True), flush=True)
print("M_QUERY_LANES_COMPLETE", flush=True)
