"""Where do the missed gold chunks actually rank? The diagnostic C depends on.

Established so far: the evidence fits the budget easily (oracle 0.98-1.00, gold
averages 263-706 words against 4000), yet retrieval covers about half. The
encoder is maxed, chunk size is already right in the live store, and cheap
re-ranking lost 20 cells out of 20. So the miss is in retrieval itself, and the
question is which kind of miss.

Three failure shapes, three different fixes, and they are told apart by one
number: the rank of the gold chunk that got left out.

  rank 50-500   The candidate band is simply too narrow. Gold is found, just not
                fetched. Fix: raise candidate_k. Cheap, no new machinery.
  rank 500-5000 Gold is outranked by a crowd of look-alikes. Fix: better scoring
                or diversity at depth, not structure.
  rank > 5000   Gold is effectively invisible to the query: it shares no surface
                with the question and is only reachable through some other turn
                that does. Fix: structure (entity links, event segmentation,
                hierarchical summaries). Only this shape justifies building one.
  unranked      Not retrievable at any depth, an index or chunking problem.

Reported per ability, because "summarize the project" and "when did X happen"
almost certainly fail in different shapes and a single histogram would blur them.

Retrieval depth is `--deep` (default 2000) rather than the production 50, since
the whole point is to see how far down the misses live.

THE CONFOUND, and what `--pad-to` is for.
Rank grew with corpus size between the 100K and 500K tiers, but so did question
difficulty: gold sets average 2.84 chunks at 100K and 4.02 at 500K, and missing
messages per failed question went 3.1 to 5.4. A larger gold set mechanically
pushes its worst member deeper, so part of the rank growth may be difficulty
rather than crowding, and the two imply different fixes. Widening the candidate
band is hopeless against crowding that scales with the corpus; it is merely
expensive against harder questions.

`--pad-to N` separates them. It keeps the 100K questions and their gold exactly
as they are and inflates only the haystack, with chunks borrowed from other
conversations, up to N chunks. Difficulty is held constant by construction, so
any rank growth that survives is crowding.

  ratio = padded_p50 / unpadded_p50
  >= 2.5  crowding drives it; a wider band does not survive corpus growth
  <= 1.5  difficulty drives it; the band is the constraint, not the corpus
  between mixed, and neither fix can be chosen on this evidence alone

`--strata` cuts the same run by gold-set size, which is the second, independent
read on the same question: if questions with <= 3 gold chunks rank the same at
both tiers, difficulty was the driver.

Usage:
  python benchmarks/miss_rank.py --sizes 100K --window s200
  python benchmarks/miss_rank.py --sizes 100K --window s200 --pad-to 2239
  python benchmarks/miss_rank.py --sizes 100K,500K --window s200 --deep 3000
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

from beam_cover_diversity import windowed_chunks
from beam_probe import SKIP_ABILITIES, _chunk_of, flat_messages, gold_ids, load_size

BANDS = [(0, 10, "top10"), (10, 50, "11-50"), (50, 200, "51-200"),
         (200, 1000, "201-1000"), (1000, 5000, "1001-5000"),
         (5000, 10**9, ">5000")]


def band_of(rank: int | None) -> str:
    if rank is None:
        return "unranked"
    for lo, hi, name in BANDS:
        if lo <= rank < hi:
            return name
    return ">5000"


def build_pad_pool(convs: list[dict], spec: str, seed: int = 42) -> list[str]:
    """Chunks harvested from other conversations, used as pure distractors.

    Deterministic shuffle so a padded run is reproducible. These carry no gold
    by construction: each conversation's gold ids only ever appear in its own
    chunks, and the padded chunks are registered with an empty cover set.
    """
    import random

    pool: list[str] = []
    for conv in convs:
        texts, _ = windowed_chunks(flat_messages(conv["chat"]), spec)
        # Drop the SID tag; it is re-stamped with a fresh id when padded in.
        pool.extend(t.split("|", 1)[1] if "|" in t else t for t in texts)
    random.Random(seed).shuffle(pool)
    return pool


def run(size: str, spec: str, deep: int, budget: int, limit: int,
        out_dir: Path, pad_to: int = 0, pad_pool: list[str] | None = None
        ) -> dict:
    from mnemonics.ingest import _get_encoder, ingest
    from mnemonics.retrieve import retrieve
    from mnemonics.store import Store

    enc = _get_encoder()
    dim = enc.get_sentence_embedding_dimension()
    convs = load_size(size)
    if limit:
        convs = convs[:limit]
    print(f"[{size}] {len(convs)} convs, chunk={spec}, depth={deep}", flush=True)

    # A question contributes one row per gold message it failed to surface
    # inside the production band, so a question needing three turns and missing
    # two is counted twice. The unit of failure is the missing evidence.
    bands: dict[str, int] = defaultdict(int)
    bands_by_ability: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int))
    q_total = q_failed = 0
    ranks: list[int] = []
    # Second read on the same question: if easy questions rank the same at both
    # tiers, the tier-to-tier rank growth was difficulty, not crowding.
    ranks_by_stratum: dict[str, list[int]] = {"gold<=3": [], "gold>3": []}
    gold_chunk_counts: list[int] = []
    corpus_sizes: list[int] = []
    t0 = time.time()

    pad_cursor = 0
    for ci, conv in enumerate(convs):
        msgs = flat_messages(conv["chat"])
        texts, covers = windowed_chunks(msgs, spec)

        if pad_to and pad_pool:
            # Distractors drawn from a rotating cursor so different
            # conversations get different padding, and re-stamped with ids that
            # continue this corpus's numbering. Empty cover: no gold inside.
            need = max(0, pad_to - len(texts))
            for j in range(need):
                body = pad_pool[(pad_cursor + j) % len(pad_pool)]
                texts.append(f"SID=c{len(texts)}|{body}")
                covers.append(set())
            pad_cursor = (pad_cursor + need) % max(len(pad_pool), 1)

        words = [len(t.split()) for t in texts]
        corpus_sizes.append(len(texts))
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
                    reachable: set[int] = set()
                    for cov in covers:
                        reachable |= cov & gold
                    if not reachable:
                        continue
                    q_total += 1
                    n_gold_chunks = sum(1 for cov in covers if cov & gold)
                    gold_chunk_counts.append(n_gold_chunks)
                    stratum = "gold<=3" if n_gold_chunks <= 3 else "gold>3"

                    rows = retrieve(query=item["question"], store=store,
                                    ns="beam", top_k=deep, candidate_k=deep,
                                    hybrid=True, rerank=False)["results"]
                    ranked = [_chunk_of(r.get("text")) for r in rows]

                    # What the production band would have delivered.
                    spent, got_prod = 0, set()
                    for cid in ranked:
                        w = words[cid] if cid is not None and cid < len(words) else 0
                        if spent + w > budget and got_prod:
                            break
                        if cid is not None and cid < len(covers):
                            got_prod |= covers[cid]
                        spent += w
                    missing = reachable - got_prod
                    if not missing:
                        continue
                    q_failed += 1

                    # For each missing message, the best rank any chunk
                    # containing it reached in the deep ranking.
                    first_rank: dict[int, int] = {}
                    for r, cid in enumerate(ranked):
                        if cid is None or cid >= len(covers):
                            continue
                        for mid in covers[cid] & missing:
                            first_rank.setdefault(mid, r)
                    for mid in missing:
                        b = band_of(first_rank.get(mid))
                        bands[b] += 1
                        bands_by_ability[ability][b] += 1
                        if mid in first_rank:
                            ranks.append(first_rank[mid])
                            ranks_by_stratum[stratum].append(first_rank[mid])

        print(f"  conv {ci+1}/{len(convs)} q={q_total} failed={q_failed} "
              f"({time.time()-t0:.0f}s)", flush=True)

    total_missing = sum(bands.values())
    ranks.sort()
    res = {
        "size": size, "chunk": spec, "depth": deep, "budget_words": budget,
        "n_conversations": len(convs),
        "mean_corpus_chunks": round(sum(corpus_sizes) / len(corpus_sizes), 1),
        "questions": q_total, "questions_failed": q_failed,
        "missing_messages": total_missing,
        "bands": {k: {"n": v, "pct": round(v / total_missing, 4)}
                  for k, v in sorted(bands.items())} if total_missing else {},
        "rank_percentiles": ({p: ranks[int(len(ranks) * p / 100)]
                              for p in (50, 75, 90, 95, 99)} if ranks else {}),
        "pad_to": pad_to,
        "mean_gold_chunks": (round(sum(gold_chunk_counts) / len(gold_chunk_counts), 2)
                             if gold_chunk_counts else None),
        "rank_percentiles_by_stratum": {
            s: ({"n": len(v),
                 **{f"p{p}": sorted(v)[int(len(v) * p / 100)]
                    for p in (50, 90, 95)}} if v else {"n": 0})
            for s, v in ranks_by_stratum.items()},
        "bands_by_ability": {
            ab: {k: round(v / sum(d.values()), 4) for k, v in sorted(d.items())}
            for ab, d in sorted(bands_by_ability.items())},
        "seconds": round(time.time() - t0, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_pad{pad_to}" if pad_to else ""
    (out_dir / f"miss_rank_{size}_{spec}{tag}.json").write_text(
        json.dumps(res, indent=2))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100K")
    ap.add_argument("--window", default="s200")
    ap.add_argument("--deep", type=int, default=2000)
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pad-to", type=int, default=0,
                    help="inflate each corpus to N chunks with distractors "
                         "borrowed from other conversations; questions and gold "
                         "stay identical, so surviving rank growth is crowding")
    ap.add_argument("--out-dir", default="benchmarks/results/beam")
    args = ap.parse_args()

    os.environ.setdefault("MNEMONICS_DETERMINISTIC", "1")
    results = []
    for size in args.sizes.split(","):
        pool = None
        if args.pad_to:
            convs = load_size(size)
            if args.limit:
                convs = convs[:args.limit]
            pool = build_pad_pool(convs, args.window)
            print(f"[{size}] pad pool: {len(pool)} distractor chunks", flush=True)
        results.append(run(size, args.window, args.deep, args.budget,
                           args.limit, Path(args.out_dir), args.pad_to, pool))

    print("\n=== WHERE THE MISSED EVIDENCE RANKS ===")
    for r in results:
        print(f"\n[{r['size']}] chunk={r['chunk']} corpus~{r['mean_corpus_chunks']} "
              f"chunks, {r['questions']} questions, {r['questions_failed']} failed, "
              f"{r['missing_messages']} missing messages")
        for k, v in r["bands"].items():
            print(f"  {k:<12} {v['n']:>5}  {v['pct']*100:>5.1f}%")
        if r.get("pad_to"):
            print(f"  PADDED to {r['pad_to']} chunks (difficulty held constant)")
        print(f"  mean gold chunks per question: {r.get('mean_gold_chunks')}")
        for st, v in r.get("rank_percentiles_by_stratum", {}).items():
            if v.get("n"):
                print(f"  stratum {st:<9} n={v['n']:>5}  "
                      + "  ".join(f"{k}={v[k]}" for k in ("p50", "p90", "p95")))
        if r["rank_percentiles"]:
            pcs = "  ".join(f"p{p}={v}" for p, v in r["rank_percentiles"].items())
            print(f"  rank percentiles (found ones): {pcs}")
        print("  by ability, share in each band:")
        for ab, d in r["bands_by_ability"].items():
            top = "  ".join(f"{k}={v*100:.0f}%" for k, v in d.items())
            print(f"    {ab:<26} {top}")
    print("\nreading: mass in 51-200 means widen the band; mass in >5000 or "
          "unranked means the query cannot see the evidence and structure is "
          "the only route")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
