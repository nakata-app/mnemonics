"""Honest temporal ablation from ONE post-gate/pre-temporal candidate dump.

The recorded 0.958 LongMemEval-S champion is not a plain CE+temporal run.  The
archived Kaggle kernels (krun-det1, krun-det2, krun-confirm) show this exact
stage order:

    deterministic HNSW -> mn-ce-v1 rerank -> chat-ce-v3 trust gate
    -> temporal-v2 -> high-confidence gate pin

The trust gate fired on 86/500 questions.  Therefore a raw ``--dump-candidates``
snapshot (post-CE, pre-gate) cannot reproduce the champion and MUST NOT be used
for the temporal ablation.  Use ``--dump-temporal-candidates`` instead.  That
snapshot is taken after the gate and before temporal-aware and records the gate
winner/margin needed to replay the later pin exactly.

From that one shared snapshot this tool computes:

    off          trust-gated rows, no temporal stage
    on_labeled   historical temporal-v2 behaviour, including the ordinal branch
                 routed by gold question_type, then the historical gate pin
    on_no_label  same pipeline but ordinal label routing disabled; relative-time
                 handling remains label-free, then the same gate pin

``on_labeled`` must reproduce the full run's R@1 (historically 0.958) before any
ablation conclusion is accepted.
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
SCHEMA = "temporal_ablation_v2"
STAGE = "post_gate_pre_temporal"


def validate_candidates(cands: list[dict], expected_n: int) -> None:
    if len(cands) != expected_n:
        raise ValueError(
            f"candidate count mismatch: expected {expected_n}, got {len(cands)}; "
            "regenerate the dump with the intended --n"
        )
    qids = [c.get("qid") for c in cands]
    if any(q is None for q in qids) or len(set(qids)) != len(qids):
        raise ValueError("candidate dump must contain one unique non-null qid per row")
    bad = [c.get("qid") for c in cands
           if c.get("schema") != SCHEMA or c.get("stage") != STAGE]
    if bad:
        raise ValueError(
            f"wrong candidate dump contract for {len(bad)} rows; expected "
            f"schema={SCHEMA!r}, stage={STAGE!r}. Use --dump-temporal-candidates, "
            "not --dump-candidates."
        )
    for c in cands:
        rows = c.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"{c['qid']}: rows must be a list")
        ids = [r.get("id") for r in rows]
        if any(i is None for i in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{c['qid']}: row ids must be non-null and unique")
        info = c.get("gate_info") or {}
        if info.get("fired") and c.get("gate_top_id") is None:
            raise ValueError(f"{c['qid']}: fired gate is missing gate_top_id")
        if c.get("gate_top_id") is not None and c.get("gate_top_id") not in ids:
            raise ValueError(f"{c['qid']}: gate_top_id is not present in rows")


def replay_gate_pin(rows: list[dict], cand: dict) -> list[dict]:
    """Replay the historical post-temporal high-confidence gate pin."""
    top_id = cand.get("gate_top_id")
    pin_margin = cand.get("trust_gate_pin_margin")
    info = cand.get("gate_info") or {}
    margin = float(info.get("ftce_margin") or 0.0)
    if top_id is None or pin_margin is None or margin < float(pin_margin) or not rows:
        return rows
    if rows[0].get("id") == top_id:
        return rows
    idx = next((i for i, r in enumerate(rows) if r.get("id") == top_id), None)
    if idx is None:
        return rows
    return [rows[idx]] + rows[:idx] + rows[idx + 1:]


def apply_variant(variant: str, cand: dict, q: dict) -> list[dict]:
    rows = [dict(r) for r in cand["rows"]]
    if variant == "off":
        return rows
    rows = H._temporal_rerank(
        rows,
        q,
        temporal_v2=bool(cand.get("temporal_v2")),
        temporal_v3=bool(cand.get("temporal_v3")),
        label_routing=(variant == "on_labeled"),
    )
    return replay_gate_pin(rows, cand)


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
            per_q[v][qid] = hits_at_k(apply_variant(v, c, q), answer)
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
                    help="JSON written by longmemeval_eval.py --dump-temporal-candidates")
    ap.add_argument("--data", type=Path, required=True, help="longmemeval_s_cleaned.json")
    ap.add_argument("--out", type=Path, default=Path("temporal_ablation.json"))
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expected-n", type=int, default=500,
                    help="expected candidate rows; champion ablation requires 500")
    args = ap.parse_args()

    cands = json.loads(args.candidates.read_text())
    try:
        validate_candidates(cands, args.expected_n)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    questions = {q["question_id"]: q for q in json.loads(args.data.read_text())}
    res = evaluate_variants(cands, questions)
    summary = summarize(res["per_q"], res["qtype"])
    result = {
        "candidate_schema": SCHEMA,
        "candidate_stage": STAGE,
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
