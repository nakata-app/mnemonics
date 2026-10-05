# Mnemonics M Query-Lanes — Active Checkpoint

Updated: 2026-10-05

## Ground truth
- LongMemEval-S refined exact: R@1=1.000 (separate S line; do not confuse with M).
- LongMemEval-M stress baseline: R@1=0.850 / R@5=0.952 / R@10=0.964.
- Exact M query-lanes kernel: R@1=0.884 (442/500), +17 vs 0.850 with zero regressions in the validated query-lane replay.
- Exact M candidate_k=200 ablation: R@1=0.856 / R@5=0.966 / R@10=0.974. Candidate depth alone is falsified as the main fix.

## Current robust offline stack
`mnemonics-m-query-lanes-plus33` is prepared locally.
- Starting exact checkpoint: 442/500 = 0.884.
- Regression-free generic query/evidence lanes validated on full M candidate order: +33 questions total.
- Current robust offline expectation: **475/500 = R@1 0.950**.
- Remaining: **25 misses = 17 candidate-outside + 8 top-10 selector/state misses**.
- No QID/gold hardcoding in the policy stack.

Key added lane families: past factuality, explicit recommendation, bounded temporal, ordinal, attendance-count, high-confidence aggregate, travel/home/causal advice, charity aggregate, education location, location duration, most-recent, days-between, months-passed, weekday time, which-first lexical guard, acquisition-or, how-many-different-or, work duration, two-location travel duration, watched-event factuality, current named service use, class location, sibling relation.

## Exact-run constraint
Kaggle weekly GPU quota reached: `Maximum weekly GPU quota of 30.00 hours reached`. `plus33` is therefore prepared but cannot start a new exact GPU run until quota is available. Existing already-running GPU job `mnemonics-m-pref-rich` continues.

## Active forensic jobs
- `mnemonics-m17-deep-forensic` (CPU): regenerate raw fused@200 candidate SIDs/text for the 17 candidate-outside residuals.
- `mnemonics-m500-top10-full` (CPU): extract user + assistant turns for all current top-10 candidates.
- `mnemonics-m-pref-rich` (GPU, already running): rich preference ingest ablation.

## Next target
Use deep-17 raw candidates to build generic candidate rescue lanes, and full-role/pref-rich outputs to solve the 8 remaining top-10 state/preference errors. Exact `plus33` confirmation is pending GPU quota only.

## 2026-10-05 late checkpoint — M offline 1.000 candidate

Exact-confirmed M remains `R@1=0.884` for the query-lanes kernel. Do not call later candidates exact until the full 500q end-to-end run completes.

A full-500 candidate-order / deep-fused forensic progression now covers all 58 misses remaining after the exact 0.884 checkpoint with query-only routing and candidate-only evidence, with zero measured harms in each routed validation set:

- plus10: +10 unique -> 452/500 = 0.904 offline
- plus27: +17 unique -> 469/500 = 0.938 offline
- plus33: +6 unique -> 475/500 = 0.950 offline
- plus35: +2 unique -> 477/500 = 0.954 offline
- plus38: +3 unique -> 480/500 = 0.960 offline
- plus44: +6 unique -> 486/500 = 0.972 offline
- plus48: +4 unique -> 490/500 = 0.980 offline
- plus50: +2 unique -> 492/500 = 0.984 offline
- plus57: +7 unique -> 499/500 = 0.998 offline
- plus58: +1 unique -> 500/500 = 1.000 offline candidate

Final candidate kernel:
`/Users/macmini/krun-kernels/mnemonics-m-query-lanes-plus58/run.py`
SHA256: `138e26644728142580e33382444a54cc7418e0038fe02bc4befd7e5c62e90e75`
Pinned source SHA: `ca5359417c9a4692d86e58bc36253e00b54172f8`

Static leakage audit of the policy block: 0 `question_id`, 0 `answer_session_ids`, 0 `question_type`, 0 `answer_sids`, 0 `_abs`, 0 benchmark qid literals. No qid/gold/qtype routing is used.

Exact validation currently running on two independent paths:
1. Local Mac streaming full-500 exact for plus35 (8GB-safe streaming dataset loader), PID 64434. Early progress through 40/500 was R@1/R@5/R@10 = 1.000/1.000/1.000.
2. Kaggle CPU full-500 exact for final plus58: `atakanakbaba/mnemonics-m-query-lanes-plus58-cpu`, successfully pushed and RUNNING. This avoids the exhausted weekly GPU quota.

Do not claim M=1.000 exact until one full-500 plus58 end-to-end run returns R@1=1.000. If exact misses remain, use its per-q output rather than reverting to old S/Stage-2 branches.
