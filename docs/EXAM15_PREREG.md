# EXAM 15 (re-opened 2026-10-25): event-time date-consistency lane over the S11 order

Written 2026-10-25 00:40 local, before any temp1 item exists and before any temp1 score exists. Frozen at the
commit that adds this file; the ledger entry points here.

## Why now
EXAM 17 (order fusion), 18 (routed 14B judge), 19 (pooling) and 20 (HyDE) all failed their dev gates on
2026-10-23/24. The 2026-10-17 diagnosis stands: the residual M gap is top-1 discrimination among topically
similar near-tie sessions. EXAM 15 was registered on 2026-10-17 and marked "blocked, not falsified" on
2026-10-23 because no dev set carries question-relative time anchors. This re-opening builds that dev set and
turns EXAM 15 from a *candidate lane* (coverage is not binding: session coverage 1.000 at band 50) into a
*ranking* signal: when the question anchors the target event in time, a date-consistent session beats a
date-inconsistent one among the S11 near-ties.

## Dev set temp1 (synthetic, LongMemEval temporal-reasoning type, relative-anchor subclass)
Generator: `benchmarks/synth_temp1_gen.py` (the spec is its docstring). Items are written by open-weights
Qwen3-14B (HF 40c06982, NF4, greedy, Kaggle kernel `mnemonics-temp1-qwen14b-gen`) from a fixed prompt; the
dating scheme is seeded (15) and mechanical; validity is checked mechanically (structure, token placement,
detail uniqueness and leakage) and never by a score. Target 80 questions: 64 anchor items (anchor kinds cycle
days / weeks / months / last-weekday; evidence session dated after the event on even items; on even items one
distractor's session date falls inside the question window so session-timestamp routing alone points wrong)
and 16 control items (duration questions that name the event; the lever must be inert on them). Haystack 40.
No M/S data is read at any step. The generated items are cached as a file; the generation step is not claimed
deterministic (batched decoding), the retrieval pipeline measured on the set is.

Known limit, stated before measuring: temp1 covers the relative-anchor subclass only. M temporal-reasoning
also has duration, chronology and aggregate questions, on which this lever is inert. A temp1 PASS therefore
licenses the M exam; it does not predict its size.

## Pipeline for temp1 (identical to pref2's)
`synth_to_perq.py` (MiniLM-L6 dense base order over the 40-session haystack) -> `make_rerank_docs.py` w3
(top-40, 3 x 600-char CE windows) and full3000 -> Kaggle `mnemonics-temp1-qrr4b` (4B w3), `-qrr4b-full` (4B
full), `-qvlrr8b-full` (VL-8B full), kernels copied from the pref2/LoCoMo ones with only the dataset source
changed -> `order_fusion_dev` helpers give S11. S11 on temp1 is the baseline and is measured once, by the same
script run that scores the lever (no look at S11 misses first).

## Lever (frozen; `benchmarks/exam15_datelane_dev.py` at this commit)
- Route: `mnemonics.memory_route.route_memory_query(question, reference_time=question_date)` (code that
  predates this exam). If `target_time` is None the order is unchanged.
- Date set per session: `session` arm = {session date}; `event` arm = {session date} U event dates resolved
  from the user's turns by the rule-based extractor in the script (N units ago, today/yesterday/day before,
  last <weekday>, last week/month/year, "on <Month> <day>", ISO dates), resolved against the session date.
- Consistent = any date in [target - window - 1 d, target + window + 1 d], window = the router's tolerance.
- Re-order: stable partition of the S11 top-K, consistent first; tail unchanged.
- Grid (6, fixed): K in {5, 10, 20} x arm in {session, event}. Selection = best temp1-anchor R@1; ties ->
  smaller K, then `session`.

## Gate (all three)
- G1 temp1-anchor (64): R@1 >= S11 + 0.05, McNemar net > 0, p < 0.10.
- G2 temp1-control (16): no R@1/R@5/R@10 below S11 - 0.005.
- G3 LoCoMo 200 and pref2 80: no R@1/R@5/R@10 below S11 - 0.005. (LoCoMo has no question date, so the lever
  is inert there by construction; it is still scored and printed.)

## M exam (conditional on PASS, one run)
Apply the selected config offline to the persisted EXAM 11 M order (results/exam11 order file, m_qdates.json,
m_deep_top75_sessions.jsonl) -- no GPU, no cloud. Step 0 must reproduce EXAM 11 0.894/0.966/0.984 from the
persisted files before the lever is applied. Report overall + by type, routed count, fixes/breaks. The `session`
arm, if selected, is a re-test of timestamp-only routing that the 2026-10-07 arithmetic falsified; a PASS of it
on temp1 but not on M is the expected outcome and would be reported as such.

## Forbidden after this commit
Editing the router grammar, the extractor, TOL, the grid, the gate, the dating scheme or the validity rules;
re-generating items after seeing any score; selecting items by score; reading M per-question results before
the exam; running the M step on a FAIL.
