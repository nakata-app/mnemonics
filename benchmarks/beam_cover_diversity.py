"""Close the BEAM cover gap: chunk granularity x ranking strategy, budget-fair.

`beam_probe.py` measured the problem: on BEAM, hit@10 barely moves from 100K to
1M (0.786 -> 0.706) while cover@10 halves (0.451 -> 0.237). The system keeps
finding *a* gold turn as the haystack grows and stops finding *all* of them.

Two independent causes, so two independent axes:

CHUNKING. Gold sets span several turns. With one turn-pair per chunk, three
gold turns need three of the ten slots and each must win on its own merit.
Wider chunks let one slot carry several gold turns.

RANKING. A single dense query vector ranks by similarity to the whole question,
so top slots fill with near-duplicates of the single best match. Gold turns that
are unlike the question (BEAM multi_session_reasoning gold is e.g. turns 14,
106, 162) get crowded out by neighbours of the first hit.

    mmr    Maximal Marginal Relevance over the band: penalise a row that merely
           restates an already-selected row. Attacks crowding directly.
    prf    Pseudo-relevance feedback: use the top-n rows as queries in their own
           right, RRF-fuse with the original. Reaches gold that looks like the
           gold you already found rather than like the question. No LLM.
    nbr    Adjacency expansion: promote the chunks immediately before/after each
           hit. Cheap, and the shape multi-turn questions actually have.
    mmr+nbr, prf+nbr  the combinations worth checking.

THE TRAP, and why the headline metric is not cover@10.
Widening chunks inflates cover@10 for free: ten slots simply carry more text. A
session-sized chunk would score near 1.0 while returning something no answerer
can use. Slot-based cover across different chunk sizes is not a comparison, it
is a units error.

So every arm is also scored at a fixed TEXT BUDGET: take rows until the
cumulative word count reaches `--budget` (default 4000 words, roughly the
context a memory layer is willing to spend), then ask the same question. That
number is comparable across chunk sizes because the cost is held equal. Both are
reported, plus words@10, so any slot-based gain that is really a budget increase
is visible rather than hidden.

Success criterion (fixed before running, 100K tier, 355 questions):
  cover@10  >= 0.55   (baseline 0.451) AND hit@10 >= 0.786 (no regression)
  cover@budget must move too, otherwise the slot-based gain was bought with
  context and does not count.

A negative result closes the cheap route and is worth recording: it would mean
cover needs a structural index (event segmentation, hierarchical summaries),
not better ranking over flat turns.

Usage:
  python benchmarks/beam_cover_diversity.py --sizes 100K
  python benchmarks/beam_cover_diversity.py --sizes 100K --windows 1,2,4 \
      --arms baseline,mmr,prf,nbr,mmr+nbr,prf+nbr
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from beam_probe import SKIP_ABILITIES, _chunk_of, flat_messages, gold_ids, load_size

# beam_probe.py numbers, window=1 turn-pair, plain hybrid retrieval, no CE.
BASELINE = {"100K": {"cover@10": 0.451, "hit@10": 0.786},
            "500K": {"cover@10": 0.416, "hit@10": 0.700},
            "1M": {"cover@10": 0.237, "hit@10": 0.706}}
TARGET_COVER10 = 0.55


def windowed_chunks(msgs: list[dict], spec: str) -> tuple[list[str], list[set[int]]]:
    """Chunk the conversation at the granularity named by `spec`.

    `"1"`, `"2"`, `"4"`  group that many turn-pairs per chunk. `"1"` reproduces
    beam_probe.turn_chunks.

    `"s200"`, `"s400"`   split *within* a turn-pair into word windows of that
    size (40-word overlap, the mnemonics ingest default). BEAM turn-pairs
    average ~770 words because the assistant answers at length, so ten slots of
    whole turn-pairs cost ~7700 words. Sub-turn chunks buy more distinct gold
    locations per unit of context, which is the axis the budget metric rewards.
    A sub-chunk inherits its parent pair's message ids, so gold mapping is
    unchanged.
    """
    # Each pair keeps the word span every message occupies inside its body, so
    # a sub-window can be credited with the messages it ACTUALLY overlaps.
    # Letting every sub-window inherit the whole parent's ids would hand small
    # chunks a free win: one arbitrary 200-word slice of a 770-word pair would
    # count as having surfaced both messages even when it contains none of the
    # evidence. That is the metric measuring "touched the neighbourhood",
    # not "returned the evidence".
    pairs: list[tuple[list[str], list[tuple[int, int, int]]]] = []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        words = f"[{m.get('role','?')}] {m.get('content','')}".split()
        spans = [(m["id"], 0, len(words))]
        if (m.get("role") == "user" and i + 1 < len(msgs)
                and msgs[i + 1].get("role") == "assistant"):
            nxt = msgs[i + 1]
            nwords = f"[assistant] {nxt.get('content','')}".split()
            spans.append((nxt["id"], len(words), len(words) + len(nwords)))
            words = words + nwords
            i += 2
        else:
            i += 1
        pairs.append((words, spans))

    def ids_in(spans: list[tuple[int, int, int]], st: int, en: int) -> set[int]:
        return {mid for mid, a, b in spans if a < en and st < b}

    texts: list[str] = []
    covers: list[set[int]] = []

    if spec.startswith("s"):
        size = int(spec[1:])
        overlap = 40
        for words, spans in pairs:
            step = max(size - overlap, 1)
            starts = range(0, len(words), step) if len(words) > size else [0]
            for st in starts:
                en = min(st + size, len(words))
                if en <= st:
                    continue
                cid = len(texts)
                texts.append(f"SID=c{cid}|" + " ".join(words[st:en]))
                covers.append(ids_in(spans, st, en))
                if en >= len(words):
                    break
        return texts, covers

    window = int(spec)
    for start in range(0, len(pairs), window):
        group = pairs[start:start + window]
        if not group:
            continue
        cid = len(texts)
        texts.append(f"SID=c{cid}|" + "\n\n".join(" ".join(w) for w, _ in group))
        ids: set[int] = set()
        for words, spans in group:
            ids |= ids_in(spans, 0, len(words))
        covers.append(ids)
    return texts, covers


def _rrf(rankings: list[list[int]], k: float = 60.0) -> list[int]:
    """Reciprocal Rank Fusion over lists of chunk ids, best-first."""
    score: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            if cid is not None:
                score[cid] += 1.0 / (k + rank + 1)
    return [cid for cid, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def _mmr(cand_ids: list[int], rel: list[float], vecs, lam: float, n: int
         ) -> list[int]:
    """Greedy MMR. `vecs` are L2-normalised, so similarity is a dot product."""
    import numpy as np

    if not cand_ids:
        return []
    # Relevance arrives as an RRF/decay score; min-max it onto [0,1] so it
    # shares a scale with cosine similarity, otherwise lambda is meaningless.
    r = np.asarray(rel, dtype="float32")
    span = float(r.max() - r.min())
    r = (r - r.min()) / span if span > 1e-9 else np.ones_like(r)

    sims = vecs @ vecs.T
    selected = [0]
    remaining = set(range(1, len(cand_ids)))
    while remaining and len(selected) < n:
        best, best_score = None, -1e9
        for i in remaining:
            redundancy = max(float(sims[i][j]) for j in selected)
            s = lam * float(r[i]) - (1.0 - lam) * redundancy
            if s > best_score:
                best, best_score = i, s
        selected.append(best)
        remaining.discard(best)
    return [cand_ids[i] for i in selected]


def _neighbours(ranked: list[int], n_seed: int, n_chunks: int) -> list[int]:
    """Interleave each of the top `n_seed` hits with its adjacent chunks.

    Kept adjacent to their seed rather than appended, so a neighbour competes
    for a slot only just behind the row that suggested it.
    """
    out: list[int] = []
    seen: set[int] = set()
    for pos, cid in enumerate(ranked):
        for c in ((cid, cid - 1, cid + 1) if pos < n_seed else (cid,)):
            if c is None or c < 0 or c >= n_chunks or c in seen:
                continue
            seen.add(c)
            out.append(c)
    return out


def _ids_of(ranked: list[int], covers: list[set[int]], k: int) -> set[int]:
    """Message ids reachable from the first k retrieved chunks.

    Coverage is defined over message ids, not chunk ids. With one chunk per
    turn-pair the two are equivalent, but sub-turn chunking maps many chunks
    onto the same message, and demanding that *every* such sub-chunk be
    retrieved would be an impossible bar that has nothing to do with whether
    the answer's evidence was surfaced.
    """
    out: set[int] = set()
    for cid in ranked[:k]:
        if cid is not None and 0 <= cid < len(covers):
            out |= covers[cid]
    return out


def _cover_at_budget(ranked: list[int], gold: set[int], words: list[int],
                     covers: list[set[int]], budget: int
                     ) -> tuple[bool, bool, int]:
    """Walk the ranking until `budget` words are spent. Returns (hit, cover, k)."""
    spent = 0
    taken: set[int] = set()
    n = 0
    for cid in ranked:
        w = words[cid] if cid is not None and 0 <= cid < len(words) else 0
        if spent + w > budget and n:
            break
        if cid is not None and 0 <= cid < len(covers):
            taken |= covers[cid]
        spent += w
        n += 1
    return bool(taken & gold), gold.issubset(taken), n


def run(size: str, windows: list[str], arms: list[str], lam: float,
        prf_n: int, nbr_seed: int, budget: int, limit: int, candidate_k: int,
        out_dir: Path, rerank: bool = False) -> dict:
    from mnemonics.ingest import _get_encoder, ingest
    from mnemonics.retrieve import retrieve
    from mnemonics.store import Store

    enc = _get_encoder()
    dim = enc.get_sentence_embedding_dimension()
    convs = load_size(size)
    if limit:
        convs = convs[:limit]
    print(f"[{size}] {len(convs)} convs dim={dim} windows={windows} "
          f"arms={arms} budget={budget}w", flush=True)

    keys = [f"w{w}:{a}" for w in windows for a in arms]
    agg = {k: {"n": 0, "hit@5": 0, "cover@5": 0, "hit@10": 0, "cover@10": 0,
               "cover@20": 0, "hit@B": 0, "cover@B": 0, "words@10": 0,
               "k@B": 0} for k in keys}
    by_ability = {k: defaultdict(lambda: {"n": 0, "cover@10": 0, "cover@B": 0})
                  for k in keys}
    t0 = time.time()

    for ci, conv in enumerate(convs):
        msgs = flat_messages(conv["chat"])
        probes = conv["probing_questions"]
        if isinstance(probes, str):
            probes = ast.literal_eval(probes)

        for w in windows:
            texts, covers = windowed_chunks(msgs, w)
            words = [len(t.split()) for t in texts]
            n_chunks = len(texts)
            with tempfile.TemporaryDirectory() as td:
                store = Store(td, dim=dim)
                ingest(texts=texts, store=store, ns="beam",
                       augment_preferences=True, chunk_size=99999,
                       chunk_overlap=0)

                for ability, items in probes.items():
                    if ability in SKIP_ABILITIES:
                        continue
                    for item in items:
                        gold = gold_ids(item)
                        if not gold:
                            continue
                        reachable: set[int] = set()
                        for cov in covers:
                            reachable |= cov & gold
                        if not reachable:
                            continue  # gold id outside the chat; not our miss
                        q = item["question"]

                        # With rerank the CE rescores the whole band, so
                        # top_k must stay at candidate_k or the budget walk
                        # would only ever see the CE's own top slice.
                        rows = retrieve(query=q, store=store, ns="beam",
                                        top_k=candidate_k,
                                        candidate_k=candidate_k,
                                        hybrid=True, rerank=rerank)["results"]
                        base_ids = [_chunk_of(r.get("text")) for r in rows]

                        ranked: dict[str, list[int]] = {"baseline": base_ids}

                        need_mmr = any("mmr" in a for a in arms)
                        need_prf = any("prf" in a for a in arms)

                        if need_mmr and rows:
                            vecs = enc.encode([r.get("text", "") for r in rows],
                                              normalize_embeddings=True,
                                              convert_to_numpy=True,
                                              show_progress_bar=False)
                            rel = [float(r.get("score", 0.0)) for r in rows]
                            ranked["mmr"] = _mmr(base_ids, rel, vecs, lam, n=25)

                        if need_prf and rows:
                            seeds = []
                            for r in rows[:prf_n]:
                                txt = r.get("text", "")
                                # Strip the SID tag: the expansion query should
                                # be the turn's words, not our bookkeeping.
                                body = txt.split("|", 1)[1] if "|" in txt else txt
                                sr = retrieve(query=body[:2000], store=store,
                                              ns="beam", top_k=candidate_k,
                                              candidate_k=candidate_k,
                                              hybrid=True,
                                              rerank=False)["results"]
                                seeds.append([_chunk_of(x.get("text")) for x in sr])
                            ranked["prf"] = _rrf([base_ids] + seeds)

                        if "nbr" in arms:
                            ranked["nbr"] = _neighbours(base_ids, nbr_seed, n_chunks)
                        if "mmr+nbr" in arms and "mmr" in ranked:
                            ranked["mmr+nbr"] = _neighbours(ranked["mmr"],
                                                            nbr_seed, n_chunks)
                        if "prf+nbr" in arms and "prf" in ranked:
                            ranked["prf+nbr"] = _neighbours(ranked["prf"],
                                                            nbr_seed, n_chunks)

                        for a in arms:
                            ids = ranked.get(a) or []
                            key = f"w{w}:{a}"
                            m = agg[key]
                            m["n"] += 1
                            ba = by_ability[key][ability]
                            ba["n"] += 1
                            for k in (5, 10, 20):
                                got = _ids_of(ids, covers, k)
                                if f"hit@{k}" in m:
                                    m[f"hit@{k}"] += int(bool(got & reachable))
                                if f"cover@{k}" in m:
                                    m[f"cover@{k}"] += int(
                                        reachable.issubset(got))
                            m["words@10"] += sum(
                                words[c] for c in ids[:10]
                                if c is not None and 0 <= c < len(words))
                            hb, cb, kb = _cover_at_budget(ids, reachable, words,
                                                          covers, budget)
                            m["hit@B"] += int(hb)
                            m["cover@B"] += int(cb)
                            m["k@B"] += kb
                            ba["cover@10"] += int(
                                reachable.issubset(_ids_of(ids, covers, 10)))
                            ba["cover@B"] += int(cb)

        first = keys[0]
        done = agg[first]["n"]
        best = max(keys, key=lambda k: agg[k]["cover@B"])
        print(f"  conv {ci+1}/{len(convs)} q={done} "
              f"best={best} cov@B={agg[best]['cover@B']/max(agg[best]['n'],1):.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    res = {
        "size": size, "n_conversations": len(convs), "dim": dim,
        "windows": windows, "arms": arms, "lam": lam, "prf_n": prf_n,
        "nbr_seed": nbr_seed, "budget_words": budget,
        "candidate_k": candidate_k, "rerank": rerank,
        "ce_model": (os.environ.get("MNEMONICS_RERANK_MODEL",
                                    "BAAI/bge-reranker-v2-m3") if rerank else None),
        "seconds": round(time.time() - t0, 1),
        "baseline_reference": BASELINE.get(size),
        "variants": {
            k: {"n": m["n"],
                **{x: round(m[x] / m["n"], 4) for x in m if x != "n"}}
            for k, m in agg.items() if m["n"]
        },
        "by_ability": {
            k: {ab: {"n": d["n"],
                     "cover@10": round(d["cover@10"] / d["n"], 4),
                     "cover@B": round(d["cover@B"] / d["n"], 4)}
                for ab, d in sorted(bya.items())}
            for k, bya in by_ability.items()
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_ce_k{candidate_k}" if rerank else ""
    (out_dir / f"diversity_{size}{tag}.json").write_text(json.dumps(res, indent=2))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100K")
    ap.add_argument("--windows", default="s200,s400,1,2",
                    help="chunk granularity: sN = N-word sub-turn windows, integer = that many turn-pairs")
    ap.add_argument("--arms", default="baseline,mmr,prf,nbr,mmr+nbr,prf+nbr")
    ap.add_argument("--lam", type=float, default=0.5, help="MMR relevance weight")
    ap.add_argument("--prf-n", type=int, default=3)
    ap.add_argument("--nbr-seed", type=int, default=5)
    ap.add_argument("--budget", type=int, default=4000,
                    help="word budget for the chunk-size-fair metric")
    ap.add_argument("--candidate-k", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rerank", action="store_true",
                    help="cross-encoder rerank the band before the budget walk")
    ap.add_argument("--out-dir", default="benchmarks/results/beam")
    args = ap.parse_args()

    os.environ.setdefault("MNEMONICS_DETERMINISTIC", "1")
    windows = args.windows.split(",")
    arms = args.arms.split(",")

    results = []
    for size in args.sizes.split(","):
        results.append(run(size, windows, arms, args.lam, args.prf_n,
                           args.nbr_seed, args.budget, args.limit,
                           args.candidate_k, Path(args.out_dir), args.rerank))

    print("\n=== COVER: CHUNKING x RANKING ===")
    for r in results:
        ref = r.get("baseline_reference") or {}
        print(f"\n[{r['size']}] {r['n_conversations']} convs, {r['seconds']}s, "
              f"budget={r['budget_words']}w")
        print(f"{'variant':<18}{'n':>5}{'hit@10':>9}{'cover@10':>10}"
              f"{'words@10':>10}{'hit@B':>8}{'cover@B':>9}{'k@B':>7}{'':>8}")
        rows = sorted(r["variants"].items(), key=lambda kv: -kv[1]["cover@B"])
        for k, m in rows:
            flag = ""
            if ref and m["cover@10"] >= TARGET_COVER10 and m["hit@10"] >= ref["hit@10"]:
                flag = "PASS@10"
            print(f"{k:<18}{m['n']:>5}{m['hit@10']:>9.3f}{m['cover@10']:>10.3f}"
                  f"{m['words@10']:>10.0f}{m['hit@B']:>8.3f}{m['cover@B']:>9.3f}"
                  f"{m['k@B']:>7.1f}{flag:>8}")
        if ref:
            print(f"  baseline (w1, no diversity): cover@10={ref['cover@10']} "
                  f"hit@10={ref['hit@10']}")
            print(f"  target: cover@10 >= {TARGET_COVER10}, hit@10 no regression,"
                  f" and cover@B must move too")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
