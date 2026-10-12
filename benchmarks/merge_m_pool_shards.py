"""Merge the 10 origv2pool_m Kaggle shards into one M poolall perq and verify the base reproduces.

Each shard k ran benchmarks/kaggle_origv2pool_m_shard{k}.py (split = sorted(all 500 qids)[k::10]) and
wrote results_origv2pool_m/perq.json with retrieved_all_sids. After downloading each shard's output to
POOL_DIR/s{k}/results_origv2pool_m/perq.json, this merges them and checks the deterministic base score
(R@1/R@5/R@10 from retrieved_top10_sids vs answer_sids) equals the pinned 0.800/0.924/0.952. Any drift
voids the dump (the plan's reproduction gate). Writes rerank_inputs/m_poolall_perq.json only when all 10
shards are present and the base reproduces.
"""
from __future__ import annotations

import json
from pathlib import Path

POOL_DIR = Path("/Volumes/WD-Backup/datasets/longmemeval/results/origv2pool_m")
OUT = Path("/Volumes/WD-Backup/datasets/longmemeval/rerank_inputs/m_poolall_perq.json")
BASE_PINNED = (0.800, 0.924, 0.952)


def main() -> None:
    rows: dict[str, dict] = {}
    present, missing = [], []
    for k in range(10):
        f = POOL_DIR / f"s{k}" / "results_origv2pool_m" / "perq.json"
        if not f.exists():
            missing.append(k)
            continue
        present.append(k)
        for r in json.load(open(f)):
            rows[r["qid"]] = r

    print(f"shards present: {present}  missing: {missing}  questions: {len(rows)}")
    if missing:
        print(f"INCOMPLETE: run kaggle_origv2pool_m_shard{{{','.join(map(str, missing))}}}.py on Kaggle, "
              "download each to origv2pool_m/s<k>/results_origv2pool_m/perq.json, then rerun this.")
        return

    n = len(rows)
    if n != 500:
        raise SystemExit(f"expected 500 questions, got {n}")

    hits = [0, 0, 0]
    for r in rows.values():
        gold = set(r["answer_sids"])
        top = r["retrieved_top10_sids"]
        for i, kk in enumerate((1, 5, 10)):
            hits[i] += any(s in gold for s in top[:kk])
    base = tuple(round(x / n, 3) for x in hits)
    print(f"base R@1/5/10 = {base}  (pinned {BASE_PINNED})")
    if base != BASE_PINNED:
        raise SystemExit("BASE DRIFT: dump voided, do not use for exam 5")

    OUT.write_text(json.dumps(list(rows.values())))
    print(f"OK, merged 500 -> {OUT}")


if __name__ == "__main__":
    main()
