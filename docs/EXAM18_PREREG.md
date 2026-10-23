# EXAM 18 pre-registration: temporal-routed deterministic judge over the S11 / EXAM 11 order

Status: **pre-registered, router frozen, judge not yet run** (blocked on Kaggle GPU, refresh 2026-10-24 00:00Z).
Parent: EXAM 17 falsified (commit 5b12747) -- recombining the existing reranker signals is closed; the
full-chunk view already carries what the window view adds. EXAM 18 does not recombine signals: it adds a new
one (a 14B judge re-read) applied only where it can plausibly help.

## Hypothesis
On temporal-recall queries, the S11 top-1 is wrong more often because the reranker orders by semantic
similarity, not by the time constraint in the question. A 14B judge that re-reads the S11 top-k with the
question's time constraint made explicit will fix some of those top-1 errors, while leaving non-temporal
queries untouched. Net R@1 on dev should rise over S11 with no category going backwards.

## Two parts, committed separately so neither can be tuned to the other
1. **Router** (`benchmarks/temporal_router_dev.py`, sha256 `8eec6576adaa3895...`, this is part 1, already
   committed): a pure query-regex classifier, encoder-free and offline, that reads only the question text and
   the qtype label. It is FROZEN before any judge output exists. The regex is designed from temporal-recall
   language cues and the published LongMemEval category definitions; it is never fitted to `answer_sids`.
2. **Judge** (part 2, to be written and run after the GPU refresh): a 14B model, temperature 0, fixed prompt,
   re-reads the S11 top-k for routed-temporal queries only and returns a reordering of those k. Non-routed
   queries keep the S11 order byte-for-byte.

## Router result on dev (fire rate per qtype -- input-vs-label face validity, NOT accuracy vs gold)
Run 2026-10-23, CPU, encoder-free. Outputs `~/.local/share/mnemonics-exam18/router_{set}.jsonl`.

| set | qtype | n | routed temporal | frac |
|---|---|---|---|---|
| locomo | cat1 | 36 | 4 | 0.111 |
| locomo | cat2 (temporal cat) | 59 | 58 | **0.983** |
| locomo | cat3 | 11 | 2 | 0.182 |
| locomo | cat4 | 94 | 29 | 0.309 |
| locomo | ALL | 200 | 93 | 0.465 |
| pref2 | single-session-preference | 80 | 16 | 0.200 |

Diagnosis (face validity, by reading the fired questions, not gold):
- cat2 recall 58/59 -- the one miss is acceptable; the router captures the temporal category cleanly.
- cat4 29/94 are mostly GENUINELY temporal ("What dish did Nate make on 9 November 2022?", "How long has
  Jolene been doing yoga?"); cat4 is "single-session knowledge" by label but many carry a time anchor. ~3-5
  are borderline (relational "when/after", e.g. "How did Melanie feel ... after the accident?").
- pref2 16/80 are mostly FALSE POSITIVES for memory-temporal purposes: future planning ("in May", "next
  Saturday"), relational use ("after washing", "during my commute"), or a product year ("2010 pu-erh").

Decision on the false positives: the router is FROZEN as-is and NOT re-tuned after seeing these questions
(re-tuning against peeked questions is a forking-paths / reward-hacking risk, and separating future-May from
past-May by regex is speculative complexity). The known over-fire is recorded here as a limitation. It is
tolerable because a preference query has no temporal fact for the judge to reorder, so the judge should return
the S11 order unchanged there; the dev gate below measures real harm directly.

## Dev gate (decided now, before the judge exists)
Instrument: `benchmarks/order_fusion_dev.py` style, judge arm added. Dev sets: LoCoMo 200 + pref2 80, scored
over the S11 / EXAM 11 order (incumbent). **pref2 S11 is a prerequisite and does not exist yet** -- it needs
the pref2 full-chunk 4B + VL-8B scores (~1 T4 hour); compute those first, reproduce the documented pref2 S10
numbers, then measure pref2 S11.

PASS (all must hold):
- LoCoMo R@1(judge) - R@1(S11) >= +0.01 AND no R@k(judge) < R@k(S11) - 0.005.
- pref2 R@1(judge) >= R@1(S11) - 0.005 (the judge must not hurt preference queries it over-routed to).
- Paired McNemar net fixes-minus-breaks > 0 on LoCoMo, with fixes concentrated in routed-temporal queries.
- Encoder pinned `MNEMONICS_ENCODER_MODEL=sentence-transformers/all-MiniLM-L6-v2`; re-run byte-identical.

FAIL -> no M run, M untouched, record the falsification like EXAM 17.

## Commands (after GPU refresh)
1. pref2 full-chunk scores (GPU, ~1 T4 h): score_rerank_docs.py full3000 for `mnemonics-pref2-qrr4b-full`
   and `mnemonics-pref2-qvlrr8b-full` (mirror the locomo full paths in order_fusion_dev.py SETS).
2. Confirm S11 computable on pref2: `python3 -I benchmarks/order_fusion_dev.py --sets pref2` now prints S11/F
   (no longer "blocked").
3. Build + run the judge arm (part 2), temperature 0, router from part 1, over LoCoMo + pref2.

## What would make this reward-hacking (forbidden, listed so it is caught)
Re-tuning the frozen router after seeing dev gold; routing by anything the judge produced; dropping pref2 from
the gate because the judge hurts it; reporting LoCoMo-only when pref2 fails; calling an exploratory judge
prompt variant the pre-registered one after seeing its score.
