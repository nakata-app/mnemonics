"""BEAM retrieval probe — where does mnemonics stand on the unsaturated benchmark?

BEAM (Beyond a Million Tokens, ICLR 2026) ships 100 synthetic conversations at
100K / 500K / 1M token scale, each with 20 probing questions across 10 memory
abilities. The official eval generates an answer with a backbone LLM and scores
it with an LLM judge — that costs money and mixes the retriever's quality with
the answerer's.

This probe measures ONLY the part mnemonics owns: given the probing question,
does the store surface the turns the answer actually depends on? BEAM hands us
`source_chat_ids` (message ids in the flattened chat) for 9 of the 10
abilities, so this is ground truth, not a proxy. `abstention` has no source by
construction and is excluded — a retriever cannot be scored on a question whose
correct answer is "the chat does not say".

Two metrics, because they answer different questions:
  hit@k    at least one gold turn in the top-k. Generous; what single-hop needs.
  cover@k  EVERY gold turn in the top-k. What multi-hop actually needs, and the
           number that separates a real memory system from a lucky one.

No cross-encoder and no LLM: retrieval only, so the result is a floor, not a
BEAM score. Do not compare it to published BEAM accuracy.

Usage:
  python benchmarks/beam_probe.py --sizes 100K
  python benchmarks/beam_probe.py --sizes 100K,500K,1M --top-ks 5,10,20,50
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

REPO = "Mohammadta/BEAM"
SIZES = {"100K": "data/100K-00000-of-00001.parquet",
         "500K": "data/500K-00000-of-00001.parquet",
         "1M": "data/1M-00000-of-00001.parquet"}
SKIP_ABILITIES = {"abstention"}  # no source turns by design


def load_size(size: str) -> list[dict]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(REPO, SIZES[size], repo_type="dataset")
    return pq.ParquetFile(path).read().to_pylist()


def flat_messages(chat) -> list[dict]:
    """BEAM stores chat as a list of batches; flatten to one id-ordered list."""
    msgs = [m for batch in chat for m in batch]
    return sorted(msgs, key=lambda m: m["id"])


def turn_chunks(msgs: list[dict]) -> tuple[list[str], list[set[int]]]:
    """Pair each user turn with the assistant reply that follows it.

    Returns the chunk texts and, per chunk, the set of message ids it covers,
    so a gold id can be resolved to the chunk that contains it. Mirrors the
    turn-pair chunking the LongMemEval champion config uses.
    """
    texts: list[str] = []
    covers: list[set[int]] = []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        ids = {m["id"]}
        body = f"[{m.get('role','?')}] {m.get('content','')}"
        if (m.get("role") == "user" and i + 1 < len(msgs)
                and msgs[i + 1].get("role") == "assistant"):
            nxt = msgs[i + 1]
            ids.add(nxt["id"])
            body += f"\n[assistant] {nxt.get('content','')}"
            i += 2
        else:
            i += 1
        cid = len(texts)
        texts.append(f"SID=c{cid}|{body}")
        covers.append(ids)
    return texts, covers


def gold_ids(item: dict) -> set[int]:
    """source_chat_ids is a list, or a dict of event->list for temporal."""
    raw = item.get("source_chat_ids")
    out: set[int] = set()

    def walk(v):
        if isinstance(v, int):
            out.add(v)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)

    walk(raw)
    return out


def _chunk_of(text: str | None) -> int | None:
    if text and text.startswith("SID=c") and "|" in text:
        try:
            return int(text.split("|", 1)[0][5:])
        except ValueError:
            return None
    return None


def run_size(size: str, top_ks: list[int], candidate_k: int, limit: int,
             out_dir: Path, rerank: bool = False) -> dict:
    from mnemonics.ingest import _get_encoder, ingest
    from mnemonics.retrieve import retrieve
    from mnemonics.store import Store

    enc = _get_encoder()
    dim = enc.get_sentence_embedding_dimension()
    convs = load_size(size)
    if limit:
        convs = convs[:limit]
    print(f"[{size}] {len(convs)} conversations, encoder dim={dim}", flush=True)

    agg = defaultdict(lambda: {"n": 0, **{f"hit@{k}": 0 for k in top_ks},
                               **{f"cover@{k}": 0 for k in top_ks}})
    per_q: list[dict] = []
    t0 = time.time()
    max_k = max(top_ks)

    for ci, conv in enumerate(convs):
        msgs = flat_messages(conv["chat"])
        texts, covers = turn_chunks(msgs)
        probes = conv["probing_questions"]
        if isinstance(probes, str):
            probes = ast.literal_eval(probes)

        with tempfile.TemporaryDirectory() as td:
            store = Store(td, dim=dim)
            ingest(texts=texts, store=store, ns="beam",
                   augment_preferences=True, chunk_size=99999, chunk_overlap=0)

            for ability, items in probes.items():
                if ability in SKIP_ABILITIES:
                    continue
                for item in items:
                    gold = gold_ids(item)
                    if not gold:
                        continue
                    gold_chunks = {ci_ for ci_, cov in enumerate(covers)
                                   if cov & gold}
                    if not gold_chunks:
                        continue  # gold id outside the chat; not our miss
                    rows = retrieve(query=item["question"], store=store,
                                    ns="beam", top_k=max_k,
                                    candidate_k=max(candidate_k, max_k),
                                    hybrid=True, rerank=rerank)["results"]
                    ranked = [_chunk_of(r.get("text")) for r in rows]

                    rec = {"conv": conv["conversation_id"], "ability": ability,
                           "n_gold_chunks": len(gold_chunks),
                           "n_chunks": len(texts)}
                    a = agg[ability]
                    a["n"] += 1
                    agg["ALL"]["n"] += 1
                    for k in top_ks:
                        top = set(ranked[:k])
                        hit = bool(top & gold_chunks)
                        cover = gold_chunks.issubset(top)
                        rec[f"hit@{k}"] = hit
                        rec[f"cover@{k}"] = cover
                        a[f"hit@{k}"] += int(hit)
                        a[f"cover@{k}"] += int(cover)
                        agg["ALL"][f"hit@{k}"] += int(hit)
                        agg["ALL"][f"cover@{k}"] += int(cover)
                    per_q.append(rec)

        print(f"  [{size}] conv {ci+1}/{len(convs)} chunks={len(texts)} "
              f"q={agg['ALL']['n']} cover@{max_k}="
              f"{agg['ALL'][f'cover@{max_k}']/max(agg['ALL']['n'],1):.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    res = {
        "size": size, "n_conversations": len(convs), "dim": dim,
        "encoder": os.environ.get("MNEMONICS_ENCODER_MODEL", "all-MiniLM-L6-v2"),
        "rerank": rerank,
        "ce_model": (os.environ.get("MNEMONICS_RERANK_MODEL",
                                    "BAAI/bge-reranker-v2-m3")
                     if rerank else None),
        "top_ks": top_ks,
        "seconds": round(time.time() - t0, 1),
        "by_ability": {
            ab: {"n": v["n"],
                 **{m: round(v[m] / v["n"], 4) for m in v if m != "n"}}
            for ab, v in agg.items() if v["n"]
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_ce" if rerank else ""
    (out_dir / f"beam_{size}{tag}.json").write_text(
        json.dumps({**res, "per_q": per_q}, indent=2))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100K")
    ap.add_argument("--top-ks", default="5,10,20,50")
    ap.add_argument("--candidate-k", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="0 = all conversations")
    ap.add_argument("--rerank", action="store_true",
                    help="cross-encoder rerank the candidate band (GPU job)")
    ap.add_argument("--out-dir", default="benchmarks/results/beam")
    args = ap.parse_args()

    os.environ.setdefault("MNEMONICS_DETERMINISTIC", "1")
    top_ks = sorted(int(x) for x in args.top_ks.split(","))
    out_dir = Path(args.out_dir)

    results = []
    for size in args.sizes.split(","):
        if size not in SIZES:
            print(f"unknown size {size}; known {list(SIZES)}", file=sys.stderr)
            return 2
        results.append(run_size(size, top_ks, args.candidate_k, args.limit,
                                out_dir, args.rerank))

    label = ("BEAM RETRIEVAL + CE RERANK (no answerer)" if args.rerank
             else "BEAM RETRIEVAL FLOOR (no CE, no answerer)")
    print(f"\n=== {label} ===")
    for r in results:
        print(f"\n[{r['size']}] {r['n_conversations']} convs, {r['seconds']}s"
              f"{' ce=' + r['ce_model'] if r.get('ce_model') else ''}")
        cols = "".join(f"{f'hit@{k}':>9}{f'cov@{k}':>9}" for k in r["top_ks"])
        print(f"{'ability':<26}{'n':>5}{cols}")
        for ab, v in sorted(r["by_ability"].items(),
                            key=lambda kv: (kv[0] != "ALL", kv[0])):
            cells = "".join(f"{v[f'hit@{k}']:>9.3f}{v[f'cover@{k}']:>9.3f}"
                            for k in r["top_ks"])
            print(f"{ab:<26}{v['n']:>5}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
