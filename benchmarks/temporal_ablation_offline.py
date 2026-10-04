"""Honest temporal ablation from ONE candidate dump (CPU only, no GPU, no API).

Question: how much of the champion's R@1 comes from --temporal-aware, and how
much of that comes from the ordinal branch that is gated on the dataset's gold
``question_type`` (a label that does not exist at query time)?

``--temporal-aware`` is a pure post-processing of the cross-encoder's output, so
all variants can be computed from the SAME candidate rows. That removes the
usual confound between GPU runs (different candidate sets, ties, nondeterminism):

    off          rows exactly as the cross-encoder ranked them
    on_labeled   --temporal-aware as the champion ran it (ordinal branch routed
                 by the gold question_type)
    on_no_label  --temporal-aware with the ordinal branch OFF; only the
                 label-free relative-target branch ("N weeks ago") remains

Produce the dump once (GPU), with the champion config. ``--dump-candidates``
writes the pre-temporal order whether or not --temporal-aware is passed:

    MNEMONICS_DETERMINISTIC=1 MNEMONICS_RERANK_MODEL=BAAI/bge-reranker-v2-m3 \\
    python benchmarks/longmemeval_eval.py --mode rerank --chunk-mode turn \\
        --augment-preferences --candidate-k 50 --seed 42 \\
        --dump-candidates cands.json --out run.json

then:

    python benchmarks/temporal_ablation_offline.py \\
        --candidates cands.json --data longmemeval_s_cleaned.json --out ablation.json

Sanity check: ``on_labeled`` R@1 must reproduce the champion's R@1 for the same
config (CHAMPION.json: 0.958); if it does not, the dump and the champion run
differ and no conclusion should be drawn from the ablation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("lme_harness", _HERE / "longmemeval_eval.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

KS = (1, 5, 10)
VARIANTS = ("off", "on_labeled", "on_no_label")
COMPARISONS = (
    ("on_labeled", "off"),
    ("on_no_label", "off"),
    ("on_labeled", "on_no_label"),
)


def apply_variant(variant: str, rows: list[dict], q: dict) -> list[dict]:
    rows = [dict(r) for r in rows]
    if variant == "off":
        return rows
    return H._temporal_rerank(rows, q, label_routing=(variant == "on_labeled"))


def hits_at_k(rows: list[dict], answer_sids: set[str]) -> dict[int, bool]:
    """Same definition as the harness: unique session ids in rank order."""
    seen: list[str] = []
    for r in rows:
        sid = H._session_id_of(r.get("text"))
        if sid and sid not in seen:
            seen.append(sid)
    return {k: any(s in answer_sids for s in seen[:k]) for k in KS}


def evaluate_variants(cands: list[dict], questions: dict[str, dict]) -> dict:
    """Per-variant, per-question hit vectors from one shared candidate set."""
    per_q: dict[str, dict[str, dict[int, bool]]] = {v: {} for v in VARIANTS}
    qtype: dict[str, str] = {}
    missing = 0
    for c in cands:
        qid = c["qid"]
        q = questions.get(qid)
        if q is None:
            missing += 1
            continue
        answer = set(c.get("answer_sids") or q.get("answer_session_ids") or [])
        qtype[qid] = q.get("question_type", c.get("qtype", "unknown"))
        for v in VARIANTS:
            per_q[v][qid] = hits_at_k(apply_variant(v, c["rows"], q), answer)
    return {"per_q": per_q, "qtype": qtype, "missing": missing}


def summarize(per_q: dict, qtype: dict) -> dict:
    out: dict = {}
    for v in VARIANTS:
        qids = list(per_q[v])
        n = len(qids)
        by_type: dict = defaultdict(lambda: {"n": 0, **{f"R@{k}": 0 for k in KS}})
        row = {"n": n}
        for k in KS:
            row[f"R@{k}"] = round(sum(per_q[v][q][k] for q in qids) / max(n, 1), 4)
        for q in qids:
            t = qtype[q]
            by_type[t]["n"] += 1
            for k in KS:
                by_type[t][f"R@{k}"] += per_q[v][q][k]
        row["by_type"] = {
            t: {"n": d["n"], **{f"R@{k}": round(d[f"R@{k}"] / max(d["n"], 1), 4) for k in KS}}
            for t, d in sorted(by_type.items())
        }
        out[v] = row
    return out


def paired_bootstrap(a: list[int], b: list[int], iters: int = 10000, seed: int = 0) -> dict:
    """Mean paired difference a-b with a percentile 95% CI over questions."""
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    helped = sum(d > 0 for d in diffs)
    hurt = sum(d < 0 for d in diffs)
    mean = sum(diffs) / max(n, 1)
    if n == 0 or helped + hurt == 0:
        return {"n": n, "delta": round(mean, 4), "ci95": [0.0, 0.0], "helped": helped, "hurt": hurt}
    rng = random.Random(seed)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters)
    )
    lo, hi = means[int(0.025 * iters)], means[int(0.975 * iters) - 1]
    return {"n": n, "delta": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "helped": helped, "hurt": hurt}


def compare(per_q: dict, k: int = 1, iters: int = 10000, seed: int = 0) -> dict:
    out = {}
    for a, b in COMPARISONS:
        qids = sorted(set(per_q[a]) & set(per_q[b]))
        out[f"{a}_vs_{b}"] = paired_bootstrap(
            [int(per_q[a][q][k]) for q in qids],
            [int(per_q[b][q][k]) for q in qids],
            iters=iters, seed=seed,
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidates", type=Path, required=True,
                    help="JSON written by longmemeval_eval.py --dump-candidates")
    ap.add_argument("--data", type=Path, required=True, help="longmemeval_s_cleaned.json")
    ap.add_argument("--out", type=Path, default=Path("temporal_ablation.json"))
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cands = json.loads(args.candidates.read_text())
    questions = {q["question_id"]: q for q in json.loads(args.data.read_text())}
    res = evaluate_variants(cands, questions)
    summary = summarize(res["per_q"], res["qtype"])
    result = {
        "candidates": str(args.candidates),
        "n_candidates": len(cands),
        "missing_from_dataset": res["missing"],
        "variants": summary,
        "paired_R@1": compare(res["per_q"], 1, args.iters, args.seed),
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(f"n={summary['off']['n']}  (candidates={len(cands)}, missing={res['missing']})")
    for v in VARIANTS:
        s = summary[v]
        print(f"  {v:12s} R@1={s['R@1']:.4f}  R@5={s['R@5']:.4f}  R@10={s['R@10']:.4f}")
    for name, c in result["paired_R@1"].items():
        print(f"  {name:28s} dR@1={c['delta']:+.4f}  95%CI={c['ci95']}  "
              f"helped={c['helped']} hurt={c['hurt']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
