"""LOCOMO benchmark adapter for Mnemonics.

Apples-to-apples against Mem0's evaluation harness:
  https://github.com/mem0ai/mem0/tree/main/evaluation

Pipeline per conversation:
  1. Fresh Mnemonics store, namespaces per speaker.
  2. Ingest every session turn with timestamp metadata.
  3. For each QA (skip category==5 per Mem0 convention), retrieve top-k from
     both speakers, build ANSWER_PROMPT, ask NIM (llama-3.3-70b), record.
  4. Emit JSON in Mem0's evaluation results shape so mem0/evaluation/evals.py
     can score it.

Usage:
  NVIDIA_API_KEY=... python benchmarks/locomo_eval.py \
      --dataset /tmp/locomo/dataset/locomo10.json \
      --out eval/results/locomo_mnemonics_<ts>.json \
      --top-k 30 --candidate-k 50 --augment-preferences \
      --convs 1   # cap conversations for smoke; omit for full
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm

from mnemonics.ingest import ingest
from mnemonics.retrieve import retrieve
from mnemonics.store import Store


ANSWER_PROMPT_TEMPLATE = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the memory timestamp.
6. Always convert relative time references to specific dates, months, or years.
7. Focus only on the content of the memories from both speakers. Do not confuse character names mentioned in memories with the actual users who created those memories.
8. The answer should be less than 5-6 words.

Memories for user {{speaker_1_user_id}}:

{{speaker_1_memories}}

Memories for user {{speaker_2_user_id}}:

{{speaker_2_memories}}

Question: {{question}}

Answer:"""


def ingest_conversation(conv: dict, sample_id: str, store: Store, augment_preferences: bool) -> tuple[str, str]:
    """Ingest all sessions for both speakers. Returns (ns_a, ns_b)."""
    speaker_a = conv["conversation"]["speaker_a"]
    speaker_b = conv["conversation"]["speaker_b"]
    ns_a = f"locomo_{sample_id}_{speaker_a}"
    ns_b = f"locomo_{sample_id}_{speaker_b}"

    texts_a: list[str] = []
    metas_a: list[dict] = []
    texts_b: list[str] = []
    metas_b: list[dict] = []

    for key in conv["conversation"]:
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        turns = conv["conversation"][key]
        if not isinstance(turns, list):
            continue
        date_time = conv["conversation"].get(f"{key}_date_time", "")
        for turn in turns:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            if not text:
                continue
            stamped = f"[{date_time}] {speaker}: {text}"
            meta = {"timestamp": date_time, "speaker": speaker, "session": key, "dia_id": turn.get("dia_id", "")}
            if speaker == speaker_a:
                texts_a.append(stamped)
                metas_a.append(meta)
            elif speaker == speaker_b:
                texts_b.append(stamped)
                metas_b.append(meta)

    if texts_a:
        ingest(texts=texts_a, store=store, ns=ns_a, meta=metas_a, augment_preferences=augment_preferences)
    if texts_b:
        ingest(texts=texts_b, store=store, ns=ns_b, meta=metas_b, augment_preferences=augment_preferences)
    return ns_a, ns_b


def format_memories(out) -> str:
    """Render retrieve() output as bullet list. retrieve() returns {'results': [...]}"""
    results = out.get("results", []) if isinstance(out, dict) else out
    if not results:
        return "(no memories)"
    return "\n".join(f"- {r.get('text','')}" for r in results)


def answer_question(
    client: OpenAI,
    model: str,
    store: Store,
    ns_a: str,
    ns_b: str,
    speaker_a: str,
    speaker_b: str,
    question: str,
    top_k: int,
    candidate_k: int,
    rerank: bool,
) -> tuple[str, float]:
    """Retrieve top-k from both nss, ask LLM, return (response, latency_s)."""
    t0 = time.time()
    hits_a = retrieve(query=question, store=store, ns=ns_a, top_k=top_k, candidate_k=candidate_k, rerank=rerank)
    hits_b = retrieve(query=question, store=store, ns=ns_b, top_k=top_k, candidate_k=candidate_k, rerank=rerank)
    prompt = Template(ANSWER_PROMPT_TEMPLATE).render(
        speaker_1_user_id=speaker_a,
        speaker_2_user_id=speaker_b,
        speaker_1_memories=format_memories(hits_a),
        speaker_2_memories=format_memories(hits_b),
        question=question,
    )
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0,
            )
            return resp.choices[0].message.content.strip(), time.time() - t0
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                import time as _t
                _t.sleep(15 * (attempt + 1))
            else:
                raise
    raise RuntimeError("max retries exceeded")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to locomo10.json")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--candidate-k", type=int, default=50)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--augment-preferences", action="store_true")
    ap.add_argument("--convs", type=int, default=0, help="Cap conversations for smoke (0=all)")
    ap.add_argument("--model", default="meta/llama-3.3-70b-instruct")
    ap.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    args = ap.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("NVIDIA_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    data = json.load(open(args.dataset))
    if args.convs:
        data = data[: args.convs]

    rerank = not args.no_rerank
    results: dict[str, list[dict]] = {"mnemonics": []}

    for conv in tqdm(data, desc="conversations"):
        sample_id = conv["sample_id"]
        speaker_a = conv["conversation"]["speaker_a"]
        speaker_b = conv["conversation"]["speaker_b"]

        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "mnemonics.db")
            store = Store(path=db_path)
            try:
                ns_a, ns_b = ingest_conversation(conv, sample_id, store, args.augment_preferences)

                for qa in tqdm(conv["qa"], desc=f"qa[{sample_id}]", leave=False):
                    category = qa.get("category")
                    if str(category) == "5":
                        continue
                    question = qa["question"]
                    gt = qa["answer"]
                    try:
                        response, latency = answer_question(
                            client, args.model, store, ns_a, ns_b,
                            speaker_a, speaker_b, question, args.top_k, args.candidate_k, rerank,
                        )
                    except Exception as e:
                        response, latency = f"ERROR: {e}", 0.0
                    results["mnemonics"].append({
                        "sample_id": sample_id,
                        "question": question,
                        "answer": gt,
                        "response": response,
                        "category": str(category),
                        "latency_s": latency,
                    })
            finally:
                pass  # Store closes via tempdir cleanup; no explicit close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}  ({len(results['mnemonics'])} answers)")


if __name__ == "__main__":
    main()
