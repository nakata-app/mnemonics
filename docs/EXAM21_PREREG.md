# EXAM 21 (pre-registered 2026-10-25): recency tiebreaker over S11 near-ties, held-out M split

Written 2026-10-25, frozen at the commit that adds this file, BEFORE any EXAM 21 lever score exists on either
M half. The ledger entry points here.

## Why now, and why a held-out M split
EXAMs 17 (order fusion), 18 (14B judge), 19 (pooling), 20 (HyDE) failed their external dev gates; EXAM 15
(date-consistency) passed dev by construction but moved M only +2 (temporal, not significant). The 2026-10-25
diagnosis (goal-level, GPU-free, on the persisted EXAM 11 S11 order) found: of 53 M R@1 failures, 45 have gold
at rank 2-10 (36 at rank 2-5), BUT the dense base order also places gold at rank 2-5 (18 at base rank 2) and
the reranker demotes gold from base rank-1 in only 2 cases. The residual is **topical near-ties that neither
dense retrieval nor the reranker separates by content similarity**. That is exactly why external/synthetic dev
sets (LoCoMo, pref2, temp1) do not transfer to M: they do not reproduce M's near-tie structure. A verified
example: pref2 has S10 R@1 0.5125 > S11 0.3875, yet on M single-session-preference S10 R@1 = S11 R@1 = 0.567
(identical), the pref2 lever is inert on M.

Therefore EXAM 21 derives and tests the lever **on M itself, with a frozen 50/50 type-stratified split** (seed
21, saved to ~/.local/share/mnemonics-exam21/split.json BEFORE this commit): dev-M n=249 (R@1 0.880), test-M
n=251 (R@1 0.908). The lever family and grid are selected on **dev-M only**; the reported result is **test-M**,
held out and never inspected during design. This is the one reward-hack-free way to claim an M-transferable
gain when external dev does not transfer.

## Hypothesis
Among S11 top-K sessions that are topical near-ties, the session the user refers to is often the most RECENT
one dated on or before the question date ("our previous chat about X", "last time I ..."). Recency is orthogonal
to content similarity, which dense+reranker already exhaust. EXAM 15 already covers explicit relative time
anchors; this covers the residual where no explicit anchor exists but recency disambiguates.

## Lever family (frozen; grid fixed here)
Baseline = S11 (persisted EXAM 11 order in results/exam11_m_scored/calls.jsonl). For each question:
- Dates: question date from m_qdates.json (normalized "YYYY/MM/DD (Day) HH:MM" -> ISO); session date = the
  leading date of each m_deep_top75_sessions.jsonl entry.
- recency key per session s in the S11 top-K: age = (qdate - sdate).days if sdate <= qdate else +inf
  (future-dated sessions are never promoted).
- Reorder: stable-partition the S11 top-K: the single top-K session with the SMALLEST finite age (ties ->
  keep S11 order) is moved to the front IFF a config says so; the rest keep S11 order; tail (beyond K) unchanged.
  Two arms: `promote1` (move only that most-recent session to #1) and `sort` (stable-sort all top-K by age,
  finite before +inf). `oldest` arm = same with LARGEST finite age (negative control; recency reversed).
- Grid (fixed, 9): K in {2, 3, 5} x arm in {promote1, sort, oldest}. Plus the baseline S11 (no change).

## Selection and gate
- Selection: best **dev-M** overall R@1 among the 9 configs; ties -> smaller K, then `promote1`, then `sort`.
  If no config beats baseline S11 on dev-M R@1, the verdict is FAIL (falsified) and no test-M claim is made
  beyond reporting the selected config's test-M number for the record.
- Gate (test-M, all three):
  - G1 test-M overall R@1 > S11 test-M R@1 (strictly), McNemar net > 0, p < 0.10.
  - G2 no per-type R@1 on test-M below its S11 test-M value - 0.005 (no type sacrificed).
  - G3 determinism: byte-identical re-run of the M-apply script.
- Expected headroom stated in advance: 0..+6 questions on test-M (dev-M has 30 fails, ~half recency-separable
  at most). A null/negative result is a real and likely outcome and will be reported as falsification.

## Forbidden after this commit
Changing the split, seed, lever family, grid, arms, selection rule or gate; inspecting test-M per-question
results before the test-M run; re-selecting a config on test-M; adding a config after seeing dev-M scores;
running test-M if dev-M selects baseline (FAIL). The `oldest` arm is a negative control and may not be reported
as a result even if it happens to score high on test-M.

## Interpretation bounds
Reorder ceiling on full M is ~0.996 (2 of 500 questions have gold outside the retrieved pool; unreachable by
any reordering). EXAM 21 cannot reach R@1 = 1.0; at most it recovers recency-separable near-ties. A test-M PASS
is a genuine M gain on held-out data; it does not generalize beyond LongMemEval-M's distribution.
