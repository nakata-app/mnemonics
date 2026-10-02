"""krun job: BEAM END-TO-END eval on GPU — the Mem0-comparable metric.

Flow: ingest conversation -> retrieve top_k memories -> LLM answerer -> LLM judge
scores each rubric nugget (0/0.5/1.0) -> accuracy per question type.

This is NOT the retrieval-only probe (hit@k / cover@k). It mirrors what Mem0's
published BEAM numbers measure, so the result can be compared directly.

Run:
    krun benchmarks/kaggle_beam_e2e.py --acc NvidiaTeslaT4 --detach

Auth: OPENROUTER_API_KEY must be present. krun does not forward env vars, so the
key is injected into the pushed script at generation time (private Kaggle kernel),
or set as a Kaggle secret for a cleaner route.

Env knobs (set in the local shell before krun; they are baked into the script):
    BEAM_SIZES="100K"        comma list: 100K,500K,1M
    BEAM_LIMIT=0             0 = all conversations (smoke uses 1)
    BEAM_QUESTIONS=0         0 = all 20 probing questions per conversation
    BEAM_TOP_K=200
"""

import ast
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

WORK = "/kaggle/working"
REPO = f"{WORK}/mnemonics"
RESULTS = f"{WORK}/results"
OUT_DIR = os.path.join(RESULTS, "beam_e2e")

# Provenance: the local commit whose package tree shipped inline below. It is
# ahead of origin/main and NOT fetchable from GitHub, so the tree itself travels
# with this script and is extractall()'d over the clone.
PINNED_SHA = os.environ.get("PINNED_SHA", "0c9414ec24f92b88a9e69370400f496682f1f6d4")
PKG_B64 = "__MNEMONICS_PKG_B64__"
SMOKE = os.environ.get("BEAM_SMOKE", "1") == "1"
SIZES = [s.strip() for s in os.environ.get("BEAM_SIZES", "100K").split(",") if s.strip()]
LIMIT = int(os.environ.get("BEAM_LIMIT", "1" if SMOKE else "0"))
Q_LIMIT = int(os.environ.get("BEAM_QUESTIONS", "2" if SMOKE else "0"))
TOP_K = int(os.environ.get("BEAM_TOP_K", "200"))
LLM_WORKERS = max(1, int(os.environ.get("BEAM_LLM_WORKERS", "8")))
SHARD_COUNT = max(1, int(os.environ.get("BEAM_SHARD_COUNT", "1")))
SHARD_INDEX = int(os.environ.get("BEAM_SHARD_INDEX", "0"))
if not 0 <= SHARD_INDEX < SHARD_COUNT:
    raise ValueError("BEAM_SHARD_INDEX must satisfy 0 <= index < count")

API_BASE = os.environ.get("BEAM_API_BASE", "https://openrouter.ai/api/v1").rstrip("/")
ANSWERER_MODEL = os.environ.get("BEAM_ANSWERER", "deepseek/deepseek-chat")
JUDGE_MODEL = os.environ.get("BEAM_JUDGE", "deepseek/deepseek-chat")
API_KEY = os.environ.get("BEAM_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")

HF_URL = "https://huggingface.co/datasets/Mohammadta/BEAM/resolve/main/data/{size}-00000-of-00001.parquet"
DATA_DIR = f"{WORK}/beam_data"

# ─── Prompts (verbatim from memory-benchmarks/benchmarks/beam/prompts.py) ─────

ANSWER_GENERATION_PROMPT = """You are an AI assistant with access to stored memories from prior conversations with a user.
Use these memories to answer the following question as accurately and completely as possible.

IMPORTANT RULES:
1. Scan ALL provided memories before answering — do not stop after the first relevant one.
2. If multiple memories contain relevant information, combine and cross-reference them.
3. If the memories contain contradictory information, prefer the more recent one.
4. If the memories don't contain enough information to answer, say exactly: "I don't have enough information to answer this question."
5. For temporal questions: pay attention to dates and relative time references.
6. For ordering questions: present events in chronological order.
7. For preference questions: use the most recently stated preference.
8. Be specific and direct — include exact names, dates, numbers, and details from the memories.
9. Do NOT invent or assume information that isn't in the memories.

QUESTION: {question}

RETRIEVED MEMORIES:
{memories}

ANSWER:"""

JUDGE_PROMPT = """Evaluate whether the following LLM response demonstrates compliance with the specified RUBRIC CRITERION.

QUESTION:
{question}

LLM RESPONSE:
{response}

RUBRIC CRITERION:
{answer}

SCORING GUIDELINES:

First, determine whether the rubric criterion is a POSITIVE requirement (the response SHOULD include something) or a NEGATIVE constraint (the response SHOULD NOT include something).

**For POSITIVE requirements** (response should contain, mention, or demonstrate something):
- **1.0 (Complete Compliance)**: The required element is present, accurate, and complete. The response fully and clearly satisfies the rubric criterion.
- **0.5 (Partial Compliance)**: The required element is partially present, has minor inaccuracies, or is incomplete. The core intent is present but not fully realized.
- **0.0 (No Compliance)**: The required element is missing, incorrect, or the response is entirely off-topic / non-responsive.

**For NEGATIVE constraints** (response should NOT contain or should avoid something):
- **1.0 (Complete Compliance)**: The response is responsive to the question AND the prohibited element is absent.
- **0.5 (Partial Compliance)**: The response is responsive but contains a borderline or ambiguous reference to the prohibited element.
- **0.0 (No Compliance)**: The prohibited element is present in the response, OR the response is non-responsive (off-topic, refusal, empty).

**Compound statement handling**: If the rubric criterion contains "and" or commas connecting multiple required elements:
- All elements present and correct = 1.0
- Some (but not all) elements present and correct = 0.5
- No elements present or correct = 0.0

EVALUATION RULES:
1. **Semantic tolerance**: Paraphrases and synonyms are acceptable. The response does not need to use the exact same words as the rubric.
2. **Numeric and date equivalence**: Treat equivalent representations as identical. "$68,000" = "68k" = "sixty-eight thousand dollars". "2 years" = "24 months". Prefer normalized comparison for numbers, currencies, dates, and durations.
3. **Case / punctuation / whitespace tolerance**: Differences in capitalization, punctuation, and whitespace must be ignored when comparing content.
4. **Hedging tolerance**: Do not penalize hedging language ("I think", "probably", "it seems"), passive voice, or verbosity if the substantive content satisfies the rubric criterion.
5. **Style neutrality**: Do not penalize for tone, formatting, or length unless the rubric criterion specifically requires a particular format.
6. **Responsiveness**: If the LLM response is completely off-topic or refuses to answer, score 0.0.
7. **Independence**: Evaluate this criterion in isolation — do not consider other rubric items.
8. **Specificity matters**: Vague or generic answers that could apply to any question score lower than specific, detailed answers.

STEP-BY-STEP EVALUATION:
Follow these steps in order:
1. **Understand the Requirement**: Read the rubric criterion and classify it as a positive requirement or a negative constraint.
2. **Parse Compound Statements**: If the criterion contains multiple sub-requirements joined by "and" or commas, identify each element separately.
3. **Check Compliance**: Compare the LLM response against each element, applying the tolerance rules above (semantic, numeric, case, hedging).
4. **Assign Score**: Use the appropriate scoring table (positive or negative) and compound-statement rule to determine the score.
5. **Provide Reasoning**: Write a concise explanation referencing which elements were or were not satisfied.

Return your evaluation as a JSON object with exactly two fields:
{{"score": <0.0 or 0.5 or 1.0>, "reason": "<one concise sentence explaining your score>"}}"""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def say(*a):
    print(*a, flush=True)


def _non_retryable_http(exc):
    return isinstance(exc, urllib.error.HTTPError) and exc.code in {401, 402, 403}


def retry(fn, *args, attempts=4, base=5, **kwargs):
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if _non_retryable_http(exc) or i == attempts - 1:
                raise
            wait = base * (2 ** i)
            say(f"  retry {i + 1}/{attempts - 1} after {wait}s: {type(exc).__name__}: {exc}")
            time.sleep(wait)
    raise RuntimeError("retry exhausted")


def run(cmd, **kw):
    kw.setdefault("check", True)
    return subprocess.run(cmd, **kw)


def http_json(url, payload=None, headers=None, timeout=180):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class EmptyCompletion(RuntimeError):
    """The provider returned no usable text for a completion."""


def llm(messages, model, temperature=0.0, max_tokens=1024):
    """One chat completion, escalating max_tokens until real content comes back.

    Reasoning models can burn the whole budget on thinking tokens and return an
    empty `content`. Returning that silently poisons the metric (the judge then
    scores 0.0 for every nugget), so an empty completion is a hard error here and
    is retried with a larger budget before giving up.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://nakata.app",
        "X-Title": "mnemonics-beam-e2e",
    }

    def _call(budget):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": budget,
        }
        out = http_json(f"{API_BASE}/chat/completions", payload, headers)
        choice = out["choices"][0]
        content = (choice.get("message", {}).get("content") or "").strip()
        if not content:
            raise EmptyCompletion(
                f"empty content (finish_reason={choice.get('finish_reason')}, budget={budget})"
            )
        return content

    last_err = None
    for budget in (max_tokens, max_tokens * 3, max_tokens * 8):
        try:
            return retry(lambda: _call(budget), attempts=3, base=4)
        except EmptyCompletion as exc:
            last_err = exc
            say(f"    empty completion, escalating -> {budget * 3}: {exc}")
        except Exception as exc:  # noqa: BLE001
            if _non_retryable_http(exc):
                raise
            last_err = exc
            say(f"    llm failed at budget {budget}: {type(exc).__name__}: {exc}")
    raise EmptyCompletion(f"no content after escalation: {last_err}")


def parse_score(text):
    """Pull {'score':..} out of a judge response; tolerate fences and prose."""
    if not text:
        return 0.0, "empty judge response"
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned[3:]
        cleaned = cleaned.lstrip("json").strip()
    try:
        obj = json.loads(cleaned)
        return float(obj.get("score", 0.0)), str(obj.get("reason", ""))[:200]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(cleaned[start : end + 1])
            return float(obj.get("score", 0.0)), str(obj.get("reason", ""))[:200]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return 0.0, f"unparseable: {cleaned[:120]}"


# ─── Bootstrap ───────────────────────────────────────────────────────────────

def clone_and_install():
    """Clone for scaffolding, then overlay the EXACT local package tree.

    The local commits are not on GitHub yet, so the package is shipped inline
    (same approach kaggle_beam_ce.py uses) instead of checking out a SHA.
    """
    if os.path.isdir(os.path.join(REPO, ".git")):
        run(["git", "-C", REPO, "fetch", "--quiet", "origin"], check=False)
        run(["git", "-C", REPO, "reset", "--quiet", "--hard", "origin/HEAD"], check=False)
    else:
        retry(lambda: run(["git", "clone", "--quiet", "https://github.com/nakata-app/mnemonics.git", REPO]))
    head = run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    say(f"  scaffolding at {head}")

    shipped = base64.b64decode(PKG_B64)
    with tarfile.open(fileobj=io.BytesIO(shipped), mode="r:gz") as tf:
        tf.extractall(REPO)
    say(f"  overlaid local package tree: {len(shipped)} bytes (from {PINNED_SHA[:8]})")

    retry(lambda: run([sys.executable, "-m", "pip", "install", "-q", "-e", REPO]))
    retry(lambda: run([sys.executable, "-m", "pip", "install", "-q", "fastembed", "pyarrow", "hnswlib"]))
    sys.path.insert(0, REPO)


def download_beam(size):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{size}.parquet")
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        say(f"  cached {size}: {os.path.getsize(path) / 1e6:.1f} MB")
        return path
    url = HF_URL.format(size=size)
    say(f"  downloading {size} <- {url}")

    def _get():
        req = urllib.request.Request(url, headers={"User-Agent": "mnemonics-beam-eval"})
        with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)

    retry(_get, attempts=4, base=10)
    got = os.path.getsize(path)
    if got < 1_000_000:
        raise RuntimeError(f"{size} parquet too small: {got} bytes")
    say(f"  downloaded {size}: {got / 1e6:.1f} MB")
    return path


def load_conversations(size):
    import pyarrow.parquet as pq

    rows = pq.read_table(download_beam(size)).to_pylist()
    convs = []
    for i, item in enumerate(rows):
        raw = item.get("probing_questions", "{}")
        if isinstance(raw, str):
            try:
                probes = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                probes = {}
        else:
            probes = raw if isinstance(raw, dict) else {}
        convs.append(
            {
                "id": item.get("conversation_id", f"{size}_{i}"),
                "chat": item.get("chat", []),
                "probes": probes,
            }
        )
    return convs


# ─── Memory operations ───────────────────────────────────────────────────────

def ingest_conversation(store, conv_id, chat):
    """Ingest one conversation's chat into its own namespace."""
    from mnemonics.ingest import ingest

    texts = []
    for batch in chat:
        lines = []
        for m in batch:
            content = (m.get("content") or "").strip()
            if not content:
                continue
            role = m.get("role", "user")
            prefix = "USER" if role in ("user", "human") else "ASSISTANT"
            anchor = m.get("time_anchor") or ""
            stamp = f" ({anchor})" if anchor else ""
            lines.append(f"[{prefix}]{stamp} {content}")
        if lines:
            texts.append("\n".join(lines))
    if not texts:
        return 0
    ingest(
        texts=texts,
        store=store,
        ns=conv_id,
        model="all-MiniLM-L6-v2",
        chunk_size=200,
        chunk_overlap=40,
    )
    return len(texts)


def retrieve_memories(store, conv_id, question, top_k, query_vector=None):
    from mnemonics.retrieve import retrieve

    res = retrieve(
        question,
        store,
        ns=conv_id,
        top_k=top_k,
        model="all-MiniLM-L6-v2",
        decay=True,
        hybrid=True,
        query_vector=query_vector,
    )
    return res.get("results", [])


# ─── Evaluation ──────────────────────────────────────────────────────────────

def _evaluate_prepared_probe(prepared):
    conv_id = prepared["conversation_id"]
    qtype = prepared["question_type"]
    question = prepared["question"]
    rubric = prepared["rubric"]
    mem_text = prepared["mem_text"]
    n_hits = prepared["n_hits"]

    try:
        answer = llm(
            [{"role": "user", "content": ANSWER_GENERATION_PROMPT.format(question=question, memories=mem_text)}],
            ANSWERER_MODEL,
            max_tokens=2000,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "conversation_id": conv_id,
            "question_type": qtype,
            "question": question,
            "answer": None,
            "answer_valid": False,
            "error": f"answerer {type(exc).__name__}: {exc}"[:200],
            "rubric_scores": [],
            "accuracy": None,
            "n_hits": n_hits,
        }

    scores = []
    for nugget in rubric:
        try:
            raw = llm(
                [{"role": "user", "content": JUDGE_PROMPT.format(question=question, response=answer, answer=nugget)}],
                JUDGE_MODEL,
                max_tokens=400,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "conversation_id": conv_id,
                "question_type": qtype,
                "question": question,
                "answer": answer,
                "answer_valid": True,
                "judge_valid": False,
                "error": f"judge {type(exc).__name__}: {exc}"[:200],
                "rubric_scores": scores,
                "accuracy": None,
                "n_hits": n_hits,
            }
        sc, _reason = parse_score(raw)
        scores.append(sc)

    return {
        "conversation_id": conv_id,
        "question_type": qtype,
        "question": question,
        "answer": answer,
        "answer_valid": True,
        "judge_valid": True,
        "rubric_scores": scores,
        "accuracy": sum(scores) / len(scores) if scores else 0.0,
        "n_hits": n_hits,
    }


def evaluate_conversation(store, conv, q_limit):
    conv_id = conv["id"]
    n_chunks = ingest_conversation(store, conv_id, conv["chat"])
    say(f"    ingested ns={conv_id}: {n_chunks} chunks")

    probe_items = []
    for qtype, items in conv["probes"].items():
        if not isinstance(items, list):
            continue
        for item in items[: q_limit or None]:
            question = item.get("question") or ""
            rubric = item.get("rubric") or []
            if isinstance(rubric, str):
                rubric = [rubric]
            if question and rubric:
                probe_items.append((qtype, question, rubric))

    prepared = []
    if probe_items:
        import numpy as np
        from mnemonics.ingest import _get_encoder, _resolve_model_for_store

        resolved_model = _resolve_model_for_store("all-MiniLM-L6-v2", store)
        encoder = _get_encoder(resolved_model)
        questions = [question for _, question, _ in probe_items]
        qvecs = encoder.encode(
            questions,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        single = encoder.encode(
            [questions[0]],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        max_delta = float(np.max(np.abs(single - qvecs[0])))
        if max_delta > 1e-5:
            raise RuntimeError(f"batch embedding parity failed: max_delta={max_delta}")
        say(f"    batch query parity max_delta={max_delta:.2e}")

        for (qtype, question, rubric), qvec in zip(probe_items, qvecs):
            hits = retrieve_memories(store, conv_id, question, TOP_K, query_vector=qvec)
            mem_text = "\n".join(f"- {h.get('text', '')}" for h in hits) or "(No memories available)"
            prepared.append(
                {
                    "conversation_id": conv_id,
                    "question_type": qtype,
                    "question": question,
                    "rubric": rubric,
                    "mem_text": mem_text,
                    "n_hits": len(hits),
                }
            )

    if not prepared:
        return []

    records = [None] * len(prepared)
    workers = min(LLM_WORKERS, len(prepared))
    say(f"    evaluating {len(prepared)} probes with {workers} LLM workers")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_evaluate_prepared_probe, item): i for i, item in enumerate(prepared)}
        for future in as_completed(futures):
            i = futures[future]
            item = prepared[i]
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001
                record = {
                    "conversation_id": conv_id,
                    "question_type": item["question_type"],
                    "question": item["question"],
                    "answer": None,
                    "answer_valid": False,
                    "error": f"worker {type(exc).__name__}: {exc}"[:200],
                    "rubric_scores": [],
                    "accuracy": None,
                    "n_hits": item["n_hits"],
                }
            records[i] = record
            acc = record.get("accuracy")
            if acc is None:
                say(f"    {record['question_type']:<26} FAILED: {record.get('error', 'unknown')}")
            else:
                say(
                    f"    {record['question_type']:<26} "
                    f"nuggets={len(record.get('rubric_scores', []))} acc={acc:.3f}"
                )

    return [record for record in records if record is not None]


def summarise(records, size):
    """Report accuracy over VALID answers only, and surface how many were lost.

    A missing answer is not a zero score. Counting it as zero silently deflates
    the metric, so the valid rate is printed next to every number and a broken
    answerer can never masquerade as a low result again.
    """
    valid = [r for r in records if r.get("answer_valid") and r.get("accuracy") is not None]
    lost = len(records) - len(valid)

    by_type = {}
    for r in valid:
        by_type.setdefault(r["question_type"], []).append(r["accuracy"])
    overall = sum(r["accuracy"] for r in valid) / len(valid) if valid else 0.0

    all_types = sorted({r["question_type"] for r in records})
    say("")
    say(f"===== BEAM end-to-end accuracy ({size}) =====")
    say(f"  valid answers: {len(valid)}/{len(records)} ({len(valid) / max(len(records), 1):.1%})  lost: {lost}")
    if lost:
        say("  !! accuracy is computed over VALID answers ONLY; a missing answer is NOT zero.")
        say("  !! investigate the answerer before trusting this number.")
    say(f"{'question_type':<26} {'n':>4} {'acc':>9} {'dropped':>8}")
    for qt in all_types:
        n = len(by_type.get(qt, []))
        acc = sum(by_type[qt]) / n if n else float("nan")
        dropped = sum(1 for r in records if r["question_type"] == qt) - n
        flag = "  <- NO VALID ANSWERS" if n == 0 else ""
        say(f"{qt:<26} {n:>4} {acc:>9.3f} {dropped:>8}{flag}")
    say(f"{'ALL (valid only)':<26} {len(valid):>4} {overall:>9.3f}")
    say("")
    return {
        "size": size,
        "overall": overall,
        "n_valid": len(valid),
        "n_total": len(records),
        "valid_rate": len(valid) / max(len(records), 1),
        "by_type": {
            qt: {
                "n": len(by_type.get(qt, [])),
                "acc": (sum(by_type[qt]) / len(by_type[qt])) if by_type.get(qt) else None,
            }
            for qt in all_types
        },
    }


def main():
    if not API_KEY:
        raise SystemExit("BEAM API key missing — cannot run answerer/judge")

    say(f"=== BEAM e2e === sizes={SIZES} limit={LIMIT or 'all'} q_limit={Q_LIMIT or 'all'} top_k={TOP_K}")
    say(f"answerer={ANSWERER_MODEL} judge={JUDGE_MODEL}")
    os.makedirs(OUT_DIR, exist_ok=True)

    clone_and_install()
    from mnemonics.store import Store

    import torch

    say("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    store_path = f"{WORK}/beam_store"
    store = Store(path=store_path)

    all_summary = {}
    for size in SIZES:
        convs = load_conversations(size)
        convs = [conv for idx, conv in enumerate(convs) if idx % SHARD_COUNT == SHARD_INDEX]
        say(f"  shard={SHARD_INDEX}/{SHARD_COUNT} selected={len(convs)}")
        if LIMIT:
            convs = convs[:LIMIT]
        say(f"\n--- {size}: {len(convs)} conversations ---")

        records = []
        started = time.time()
        for idx, conv in enumerate(convs):
            say(f"  [{idx + 1}/{len(convs)}] {conv['id']}")
            try:
                records.extend(evaluate_conversation(store, conv, Q_LIMIT))
            except Exception as exc:  # noqa: BLE001
                say(f"    FAILED {conv['id']}: {type(exc).__name__}: {exc}")
            # checkpoint after every conversation so a timeout still leaves data
            with open(os.path.join(OUT_DIR, f"records_{size}.json"), "w") as f:
                json.dump(records, f)
            say(f"    elapsed {time.time() - started:.0f}s, records={len(records)}")

        summary = summarise(records, size)
        all_summary[size] = summary
        with open(os.path.join(OUT_DIR, f"summary_{size}.json"), "w") as f:
            json.dump(summary, f, indent=2)

    with open(os.path.join(OUT_DIR, "summary_all.json"), "w") as f:
        json.dump(all_summary, f, indent=2)
    say("=== BITTI ===", json.dumps({k: True for k in SIZES}))
    say("results in", OUT_DIR)


if __name__ == "__main__":
    main()
