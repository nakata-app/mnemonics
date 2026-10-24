# EXAM 19 pre-registration: chunk-evidence pooling inside the S11 / EXAM 11 pipeline

Written 2026-10-24 after EXAM 18 was falsified and BEFORE any pooling variant was scored on any set.
Goal unchanged: LongMemEval-M R@1 = 1.0, deterministic, no memorization. Incumbent: EXAM 11 (M 0.894/0.966/0.984).

## Hypothesis
Every reranker stage in the M path collapses a session to the MAX over its chunk scores (`session_max`). A session that
answers a question usually has several supporting chunks (multi-session and preference sessions especially), while a
distractor session often has exactly one high-scoring chunk (a lexical collision). A pooling that credits a second /
third supporting chunk should separate gold from single-spike distractors at #1 without changing anything else.
Expected headroom, stated now: small (M 0.894 -> 0.894..0.906, i.e. 0..+6 q); it cannot reach 1.0 alone.

## What stays frozen
Models, revisions, docs files, scores files (all persisted; this exam is offline and needs no GPU), stage-1 = w3 4B
order over the retrieved pool, stage-2 = RRF(k=60) of 4B-full and VL-full over the stage-1 top-20, ties by stage-1
order, tail keeps stage-1 order. Encoder pin `MNEMONICS_ENCODER_MODEL=sentence-transformers/all-MiniLM-L6-v2` (not
loaded; instrument is encoder-free). Only the per-session pooling function changes.

## Fixed grid (decided now; nothing outside it may be reported as the exam)
Pooling P over a session's chunk scores s_1 >= s_2 >= s_3 (same model scale within a stage, so monotone transforms are safe):
- P0 max              = s_1                              (incumbent)
- P1 top2mean         = (s_1 + s_2) / 2, s_2 := s_1 if only one chunk
- P2 max+half-second  = s_1 + 0.5 * s_2, s_2 := 0-contribution if only one chunk (i.e. s_1 only)
- P3 top3sum          = s_1 + s_2 + s_3 (missing chunks contribute 0)
Scope W: (A) stage-2 signals only (4B-full, VL-full); (B) stage-1 w3 too.
Configs: P1..P3 x {A,B} = 6 candidates + P0 incumbent. VL scores are logits (can be negative): P2/P3 add raw
values, so for VL the 'missing chunk contributes 0' rule can rank a 1-chunk session above a 3-chunk negative one -- this
is part of the hypothesis test, not patched after seeing results.

## Dev gate (LoCoMo 200 + pref2 80 over the existing S11 instrument; selection among 6 configs is acknowledged)
Instrument: `benchmarks/pooling_dev.py` (reuses order_fusion_dev loaders/rrf/hits/mcnemar verbatim).
Selection rule: the candidate with the highest LoCoMo R@1; ties -> higher LoCoMo R@5, then fewer breaks.
PASS (all must hold for the selected candidate):
- LoCoMo R@1(cand) - R@1(S11) >= +0.015 (>= 3 q; stricter than EXAM 18's +0.01 because 6 candidates are compared)
  AND no LoCoMo R@k(cand) < R@k(S11) - 0.005.
- pref2 R@1(cand) >= R@1(S11) - 0.005 AND pref2 R@10(cand) >= R@10(S11) - 0.005.
- LoCoMo paired fixes - breaks > 0 with McNemar p < 0.10.
- Byte-identical re-run.
FAIL -> no M run, M untouched, falsification recorded.

## M exam (conditional on PASS; one deterministic offline pass)
Inputs: m_ex8_perq.json (sids-key `ex8`), m_rerank_docs_ex8top20_full3000.jsonl, EXAM 11 shards
mnemonics-m-ex11-4b-s{0,1} + mnemonics-m-ex11-vl-s{0,1} (scope A only; scope B on M would need the w3 M scores
mnemonics-m-qrr4b/scores.jsonl and m_rerank_docs_top75_w3.jsonl -- if the selected config is scope B and those files
do not line up, the exam runs scope A of the same pooling and says so).
Step 0 (before the exam): reproduce EXAM 11 M 0.894/0.966/0.984 with this instrument (P0). If it does not reproduce,
record the discrepancy (score_rerank_docs.py clamps session scores at -1.0, order_fusion_dev does not) and treat the
instrument's own P0 as the incumbent for the paired comparison, reporting both.
Report overall + by type + paired vs incumbent. No M per-question inspection before the run. Success = R@1 > 0.894
with no R@k regression; otherwise falsified.

## Reward-hacking list (forbidden)
Adding poolings after seeing dev; choosing the config on pref2 or on M; dropping pref2 from the gate; reporting scope
B on dev and scope A on M as the same system without saying so; changing RRF k or tie rule; patching the VL
negative-logit behaviour after seeing results; treating P0 reproduction mismatch as a gain.
