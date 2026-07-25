"""What is the ceiling? Oracle cover@budget, with retrieval removed entirely.

Every cover number measured so far compares retrieval against retrieval. None of
them answers the prior question: can the evidence a question depends on even FIT
in the budget? If a BEAM question's gold turns total 9000 words, no retriever
covers it at a 4000-word budget and the miss is not a retrieval failure.

So: drop retrieval. For each question take the cheapest set of chunks that
covers every gold message id (greedy set cover by word cost) and ask whether it
fits the budget. That is the ceiling any retriever could reach.

The decision rule this exists to settle:
  oracle - measured < 0.10  ->  retrieval is already near its ceiling; a
                               structural memory layer has little room to win
  oracle - measured >= 0.10 ->  there is real headroom, and the size of it is
                               the honest upper bound on what such a layer buys

Usage:
  python benchmarks/oracle_cover.py --sizes 100K,500K,1M --windows s100,s200,1
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from beam_cover_diversity import windowed_chunks
from beam_probe import SKIP_ABILITIES, flat_messages, gold_ids, load_size

# Measured retrieval numbers to compare the ceiling against (cover@B, 4000w).
MEASURED = {
    ("100K", "s100"): 0.499, ("100K", "s200"): 0.462, ("100K", "1"): 0.361,
    ("500K", "s100"): 0.475, ("500K", "s200"): 0.418, ("500K", "1"): 0.334,
}


def greedy_cover(gold: set[int], covers: list[set[int]], words: list[int]
                 ) -> tuple[int, int]:
    """Cheapest chunk set covering every gold id. Returns (words, n_chunks).

    Set cover is NP-hard; greedy by words-per-newly-covered-id is the standard
    ln(n) approximation. It can overshoot the true optimum, so this is a
    slightly conservative ceiling, which is the safe direction for a bound.
    """
    need = set(gold)
    total_words = 0
    n = 0
    while need:
        best, best_ratio, best_gain = None, None, None
        for i, cov in enumerate(covers):
            gain = len(cov & need)
            if not gain:
                continue
            ratio = words[i] / gain
            if best_ratio is None or ratio < best_ratio:
                best, best_ratio, best_gain = i, ratio, cov & need
        if best is None:
            break  # unreachable gold; caller filters these out beforehand
        need -= best_gain
        total_words += words[best]
        n += 1
    return total_words, n


def run(size: str, specs: list[str], budget: int, limit: int) -> dict:
    convs = load_size(size)
    if limit:
        convs = convs[:limit]

    agg = {s: {"n": 0, "fits": 0, "words_sum": 0, "chunks_sum": 0}
           for s in specs}
    by_ability = {s: defaultdict(lambda: {"n": 0, "fits": 0, "words_sum": 0})
                  for s in specs}

    for conv in convs:
        msgs = flat_messages(conv["chat"])
        probes = conv["probing_questions"]
        if isinstance(probes, str):
            probes = ast.literal_eval(probes)

        for spec in specs:
            texts, covers = windowed_chunks(msgs, spec)
            words = [len(t.split()) for t in texts]
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
                    w, n = greedy_cover(reachable, covers, words)
                    a = agg[spec]
                    a["n"] += 1
                    a["words_sum"] += w
                    a["chunks_sum"] += n
                    fits = int(w <= budget)
                    a["fits"] += fits
                    ba = by_ability[spec][ability]
                    ba["n"] += 1
                    ba["fits"] += fits
                    ba["words_sum"] += w

    return {
        "size": size, "n_conversations": len(convs), "budget_words": budget,
        "oracle": {
            s: {"n": a["n"],
                "oracle_cover@B": round(a["fits"] / a["n"], 4),
                "mean_gold_words": round(a["words_sum"] / a["n"], 1),
                "mean_gold_chunks": round(a["chunks_sum"] / a["n"], 2),
                "measured_cover@B": MEASURED.get((size, s)),
                "headroom": (round(a["fits"] / a["n"] - MEASURED[(size, s)], 4)
                             if (size, s) in MEASURED else None)}
            for s, a in agg.items() if a["n"]
        },
        "by_ability": {
            s: {ab: {"n": d["n"],
                     "oracle_cover@B": round(d["fits"] / d["n"], 4),
                     "mean_gold_words": round(d["words_sum"] / d["n"], 1)}
                for ab, d in sorted(bya.items())}
            for s, bya in by_ability.items()
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100K,500K,1M")
    ap.add_argument("--windows", default="s100,s200,1")
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="benchmarks/results/beam")
    args = ap.parse_args()

    specs = args.windows.split(",")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for size in args.sizes.split(","):
        r = run(size, specs, args.budget, args.limit)
        results.append(r)
        (out_dir / f"oracle_{size}.json").write_text(json.dumps(r, indent=2))
        print(f"[{size}] wrote oracle_{size}.json", flush=True)

    print(f"\n=== ORACLE CEILING (budget {args.budget} words) ===")
    print(f"{'tier':<7}{'chunk':<7}{'n':>5}{'oracle':>9}{'measured':>10}"
          f"{'headroom':>10}{'gold words':>12}{'gold chunks':>13}")
    for r in results:
        for s, o in r["oracle"].items():
            meas = f"{o['measured_cover@B']:.3f}" if o["measured_cover@B"] else "-"
            head = f"{o['headroom']:+.3f}" if o["headroom"] is not None else "-"
            print(f"{r['size']:<7}{s:<7}{o['n']:>5}{o['oracle_cover@B']:>9.3f}"
                  f"{meas:>10}{head:>10}{o['mean_gold_words']:>12.0f}"
                  f"{o['mean_gold_chunks']:>13.2f}")
    print("\nrule: headroom < 0.10 -> retrieval is near its ceiling, "
          "a structural layer has little to win")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
