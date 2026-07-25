"""Encoder A/B probe — does a modern encoder change what the CE ever sees?

The champion pipeline is: encoder top-`candidate_k` + BM25 top-`candidate_k`
-> RRF fuse -> cross-encoder rerank -> top_k. The cross-encoder is the same in
every arm, so the ONLY thing a different encoder can change is the candidate
band handed to the CE. If the gold session is already in the fused top-50 for
essentially every question with all-MiniLM-L6-v2, a stronger encoder cannot
raise final R@1 no matter how good it is, and the swap is dead on arrival.

This probe measures that directly and cheaply (no CE, no answering LLM):

  vec R@1/@5/@50   encoder alone, no BM25, no CE
  bm25 R@50        lexical alone (arm-independent sanity constant)
  fused R@50       what the CE actually receives  <-- the deciding number
  fused R@1        pre-CE ordering quality

Protocol matches benchmarks/CHAMPION.json where it is observable without a CE:
--chunk-mode turn --augment-preferences --candidate-k 50 --seed 42.
temporal-aware is NOT applied: it reorders the post-retrieval top_k, it never
adds a session to the candidate band, so it cannot move any number here.

Usage:
  python benchmarks/encoder_probe.py --data longmemeval_s_cleaned.json -n 25
  python benchmarks/encoder_probe.py --data ... -n 500 --arms minilm,bge-large,qwen3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Arms. `query_prefix` matters: bge/e5/qwen3 are trained with an asymmetric
# query instruction and lose several points of retrieval without it. Omitting
# it would rig the A/B against the challengers.
ARMS: dict[str, dict] = {
    "minilm": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "doc_prefix": "",
    },
    "bge-base": {
        "model": "BAAI/bge-base-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "",
    },
    "bge-large": {
        "model": "BAAI/bge-large-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "",
    },
    "e5-large": {
        "model": "intfloat/e5-large-v2",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
    },
    "qwen3": {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "query_prefix": (
            "Instruct: Given a question, retrieve the conversation turn that "
            "answers it\nQuery: "
        ),
        "doc_prefix": "",
    },
}


def _hit(rows: list[dict], gold: set[str], k: int) -> bool:
    from longmemeval_eval import _session_id_of

    for r in rows[:k]:
        if _session_id_of(r.get("text")) in gold:
            return True
    return False


def run_arm(name: str, questions: list[dict], candidate_k: int,
            out_path: Path | None) -> dict:
    arm = ARMS[name]
    os.environ["MNEMONICS_ENCODER_MODEL"] = arm["model"]
    os.environ["MNEMONICS_DETERMINISTIC"] = "1"

    # Import after the env is set; _get_encoder resolves the model by name and
    # reloads when the resolved name changes, so arms stay isolated.
    from mnemonics.ingest import _get_encoder, ingest
    from mnemonics.retrieve import retrieve
    from mnemonics.store import Store
    from longmemeval_eval import _session_turn_chunks

    enc = _get_encoder()
    dim = enc.get_sentence_embedding_dimension()
    print(f"[{name}] {arm['model']} dim={dim}", flush=True)

    agg = {k: 0 for k in ("vec@1", "vec@5", f"vec@{candidate_k}",
                          f"bm25@{candidate_k}",
                          "fused@1", f"fused@{candidate_k}")}
    per_q: list[dict] = []
    t0 = time.time()
    n = 0

    for i, q in enumerate(questions):
        gold = set(q.get("answer_session_ids") or [])
        if not gold:
            continue
        with tempfile.TemporaryDirectory() as td:
            store = Store(td, dim=dim)
            texts: list[str] = []
            for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"]):
                texts.extend(_session_turn_chunks(sid, sess))
            if not texts:
                continue
            if arm["doc_prefix"]:
                texts = [_prefix_after_sid(t, arm["doc_prefix"]) for t in texts]
            ingest(texts=texts, store=store, ns="lme",
                   augment_preferences=True, chunk_size=99999, chunk_overlap=0)

            query = arm["query_prefix"] + q["question"]
            vec = retrieve(query=query, store=store, ns="lme",
                           top_k=candidate_k, candidate_k=candidate_k,
                           hybrid=False, rerank=False)["results"]
            fused = retrieve(query=query, store=store, ns="lme",
                             top_k=candidate_k, candidate_k=candidate_k,
                             hybrid=True, rerank=False)["results"]
            bm25 = store.search_bm25(q["question"], ns="lme", top_k=candidate_k)

        n += 1
        row = {
            "qid": q.get("question_id"),
            "qtype": q.get("question_type"),
            "n_chunks": len(texts),
            "vec@1": _hit(vec, gold, 1),
            "vec@5": _hit(vec, gold, 5),
            f"vec@{candidate_k}": _hit(vec, gold, candidate_k),
            f"bm25@{candidate_k}": _hit(bm25, gold, candidate_k),
            "fused@1": _hit(fused, gold, 1),
            f"fused@{candidate_k}": _hit(fused, gold, candidate_k),
        }
        for k in agg:
            agg[k] += int(row[k])
        per_q.append(row)

        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  [{name}] {i+1}/{len(questions)} "
                  f"fused@{candidate_k}={agg[f'fused@{candidate_k}']/max(n,1):.3f} "
                  f"({el:.0f}s)", flush=True)

    res = {
        "arm": name, "model": arm["model"], "dim": dim, "n": n,
        "candidate_k": candidate_k,
        "seconds": round(time.time() - t0, 1),
        "metrics": {k: round(v / n, 4) if n else None for k, v in agg.items()},
        "counts": agg,
    }
    if out_path:
        out_path.write_text(json.dumps({**res, "per_q": per_q}, indent=2))
        print(f"  [{name}] -> {out_path}", flush=True)
    return res


def _prefix_after_sid(text: str, prefix: str) -> str:
    """Insert the doc instruction after the SID= tag so sid parsing survives."""
    if text.startswith("SID=") and "|" in text:
        head, rest = text.split("|", 1)
        return f"{head}|{prefix}{rest}"
    return prefix + text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("-n", type=int, default=0, help="0 = all questions")
    ap.add_argument("--candidate-k", type=int, default=50)
    ap.add_argument("--arms", default="minilm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="benchmarks/results/encoder_probe")
    args = ap.parse_args()

    unknown = [a for a in args.arms.split(",") if a not in ARMS]
    if unknown:
        print(f"unknown arms: {unknown}; known: {list(ARMS)}", file=sys.stderr)
        return 2

    questions = json.loads(Path(args.data).read_text())
    if args.n:
        random.Random(args.seed).shuffle(questions)
        questions = questions[: args.n]
    print(f"questions={len(questions)} candidate_k={args.candidate_k}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for name in args.arms.split(","):
        try:
            res = run_arm(name, questions, args.candidate_k,
                          out_dir / f"{name}_n{len(questions)}.json")
        except Exception as e:  # one bad arm must not kill the rest
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            res = {"arm": name, "error": f"{type(e).__name__}: {e}"}
        summary.append(res)
        (out_dir / f"summary_n{len(questions)}.json").write_text(
            json.dumps(summary, indent=2))

    print("\n=== SUMMARY ===")
    ck = args.candidate_k
    hdr = f"{'arm':<12}{'dim':>5}{'vec@1':>8}{'vec@5':>8}{f'vec@{ck}':>9}" \
          f"{f'bm25@{ck}':>10}{'fused@1':>9}{f'fused@{ck}':>11}{'sec':>8}"
    print(hdr)
    for r in summary:
        if "error" in r:
            print(f"{r['arm']:<12} ERROR {r['error']}")
            continue
        m = r["metrics"]
        print(f"{r['arm']:<12}{r['dim']:>5}{m['vec@1']:>8.3f}{m['vec@5']:>8.3f}"
              f"{m[f'vec@{ck}']:>9.3f}{m[f'bm25@{ck}']:>10.3f}"
              f"{m['fused@1']:>9.3f}{m[f'fused@{ck}']:>11.3f}{r['seconds']:>8.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
