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

SPLIT = WORK / "split_deep1.json"
SPLIT.write_text(json.dumps({"dev": ["00ca467f", "01493427", "031748ae_abs", "06db6396", "07741c44", "078150f1", "0862e8bf", "08e075c7", "0977f2af", "09ba9854", "09d032c9", "0a995998", "0bc8ad92", "0db4c65d", "0ddfec37_abs", "0e5e2d1a", "0edc2aef", "10d9b85a", "118b2229", "129d1232", "15745da0", "157a136e", "184da446", "18dcd5a5", "195a1a1b", "19b5f2b3_abs", "1a8a66a6", "1c0ddc50", "1cea1afa", "1d4e3b97", "1de5cff2", "1f2b8d4f", "2133c1b5", "21436231", "22d2cb42", "2311e44b_abs", "25e5aa4f", "2698e78f_abs", "27016adc", "28bcfaac", "29f2956b", "2a1811e2", "2bf43736", "2ce6a0f2", "2ebe6c90", "311778f1", "3249768e", "35a27287", "36b9f61e", "37d43f65", "38146c39", "3b6f954b", "3c1045c8", "3e321797", "3fdac837", "4100d0a0", "41698283", "4388e9dd", "46a3abf7", "4adc0475", "4baee567", "4c36ccef", "4dfccbf7", "4f54b7c9", "5025383b", "50635ada", "51b23612", "54026fce", "55241a1f", "577d4d32", "5809eb10", "58470ed2", "58ef2f1c", "5a4f22c0", "5c40ec5b", "5e1b23de", "60159905", "60472f9c", "60bf93ed", "60d45044", "61f8c8f8", "6456829e", "65240037", "66f24dbb", "681a1674", "6a1eabeb", "6ade9755", "6aeb4375", "6b168ec8", "6c49646a", "6d550036", "6e984302", "7024f17c", "71017276", "71315a70", "71a3fd6b", "726462e0", "73d42213", "7405e8b1", "75499fd8", "75f70248", "778164c6", "7a87bd0c", "7e00a6cb", "8077ef71", "80ec1f4f_abs", "830ce83f", "852ce960", "8550ddae", "86b68151", "8752c811", "88432d0a", "89527b6b", "89941a93", "8a137a7f", "8aef76bc", "8c18457d", "8cf51dda", "8e9d538c", "8fb83627", "92a0aa75", "94f70d80", "95bcc1c8", "982b5123_abs", "9a707b81", "9aaed6a3", "9d25d4e0", "9ee3ecd6", "a08a253f", "a1cc6108", "a2f3aa27", "a3332713", "a3838d2b", "a4996e51", "a89d7624", "a96c20ee_abs", "aae3761f", "ad7109d1", "af8d2e46", "affe2881", "b0479f84", "b320f3f8", "b46e15ed", "b5ef892d", "b6025781", "b86304ba", "ba358f49", "ba61f0b9", "bbf86515", "bc8a6e93", "bcbe585f", "c14c00dd", "c19f7a0b", "c4a1ceb8", "c4f10528", "c6853660", "c7dc5443", "c8090214_abs", "c8f1aeed", "c9f37c46", "caf9ead2", "cc539528", "cc6d1ec1", "ce6d2d27", "cf22b7bf", "d23cf73b", "d3ab962e", "d596882b", "d6233ab6", "d7c942c3", "d905b33f", "db467c8c", "dccbc061", "dd2973ad", "e01b8e2f", "e3038f8c", "e47becba", "e493bb7c", "e56a43b9", "e6041065", "e66b632c", "e8a79c70", "e982271f", "eac54add", "eace081b", "ed4ddc30", "edced276_abs", "eeda8a6d_abs", "ef9cf60a", "f0853d11", "f35224e0", "f4f1d8a4_abs", "f685340e", "f8c5f88b", "faba32e5", "fca762bc", "gpt4_0a05b494", "gpt4_15e38248", "gpt4_1916e0ea", "gpt4_1a1dc16d", "gpt4_1d80365e", "gpt4_1e4a8aec", "gpt4_21adecb5", "gpt4_2487a7cb", "gpt4_2ba83207", "gpt4_2d58bcd6", "gpt4_2f584639", "gpt4_2f91af09", "gpt4_372c3eed", "gpt4_385a5000", "gpt4_468eb063", "gpt4_483dd43c", "gpt4_4929293b", "gpt4_4edbafa2", "gpt4_4fc4f797", "gpt4_5501fe77", "gpt4_59149c78", "gpt4_5dcc0aab", "gpt4_65aabe59", "gpt4_68e94288", "gpt4_6ed717ea", "gpt4_70e84552_abs", "gpt4_74aed68e", "gpt4_78cf46a3", "gpt4_7abb270c", "gpt4_7ca326fa", "gpt4_7de946e7", "gpt4_7fce9456", "gpt4_8279ba03", "gpt4_88806d6e", "gpt4_8e165409", "gpt4_93159ced_abs", "gpt4_98f46fc6", "gpt4_a1b77f9c", "gpt4_a56e767c", "gpt4_af6db32f", "gpt4_b4a80587", "gpt4_b5700ca9", "gpt4_c27434e8_abs", "gpt4_d12ceb0e", "gpt4_d6585ce8", "gpt4_d84a3211", "gpt4_e05b82a6", "gpt4_e061b84g", "gpt4_e414231e", "gpt4_ec93e27f", "gpt4_f420262c", "gpt4_f49edff3", "gpt4_fa19884d", "gpt4_fe651585_abs"]}))
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
    "shard": "deep1",
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
