#!/usr/bin/env bash
# Mac mini local sweep: baseline + doc_top_k ∈ {3,5,10,15}.
# Background run, outputs land in /tmp/lme50_*.json + .log.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
COMMON="--n 50 --mode no_rerank --augment-preferences --candidate-k 50 --seed 42"
OUT=/tmp

echo "=== Baseline (no doc_filter) ==="
PYTHONUNBUFFERED=1 $PY -u benchmarks/longmemeval_eval.py $COMMON \
  --out "$OUT/lme50_baseline.json" \
  --per-q-out "$OUT/lme50_baseline_perq.json" > "$OUT/lme50_baseline.log" 2>&1

for k in 3 5 10 15; do
  echo "=== Stage-2 doc_top_k=$k ==="
  PYTHONUNBUFFERED=1 $PY -u benchmarks/longmemeval_eval.py $COMMON \
    --use-doc-filter --doc-top-k $k \
    --out "$OUT/lme50_stage2_k${k}.json" \
    --per-q-out "$OUT/lme50_stage2_k${k}_perq.json" > "$OUT/lme50_stage2_k${k}.log" 2>&1
done

echo
echo "=== SWEEP SUMMARY ==="
$PY <<'PYEOF'
import json
def m(p):
    try:
        return json.load(open(p))["mnemonics_no_rerank"]
    except Exception as e:
        return {"R@1": None, "R@5": None, "R@10": None, "_err": str(e)}
rows = [("baseline", "/tmp/lme50_baseline.json")] + [
    (f"k={k}", f"/tmp/lme50_stage2_k{k}.json") for k in (3, 5, 10, 15)
]
print(f"{'config':10} {'R@1':>8} {'R@5':>8} {'R@10':>8}")
print("-" * 38)
for name, path in rows:
    r = m(path)
    print(f"{name:10} {str(r.get('R@1')):>8} {str(r.get('R@5')):>8} {str(r.get('R@10')):>8}")
PYEOF
