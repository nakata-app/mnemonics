# EXAM 22 (pre-registered 2026-10-25): stronger stage-2 reranker (Qwen3-Reranker-8B), held-out M split

Written 2026-10-25, frozen at the commit that adds this file, BEFORE any Qwen3-Reranker-8B score exists (the
Kaggle kernels mnemonics-m-ex22-8b-s0/s1 are pushed but their scores are not downloaded or inspected at this
commit). The ledger entry points here.

## Why
The 2026-10-25 diagnosis: 45 of 53 M R@1 failures are topical near-ties where BOTH the dense base order and the
S11 reranker place gold at rank 2-5. Content similarity as currently scored cannot separate gold from its
distractor(s). Inspection of dev-M near-misses shows a subset are entity/number mismatches a stronger reader
should resolve (e.g. the wrong rank-1 session is about a backpack, not the tennis racket the question names; or
lacks the fish counts the "how many fish" question needs), while others are temporal or genuinely ambiguous.
EXAM 21 (recency) was falsified. Atakan authorized an architecture change ("mimariyi değiştir / daha güçlü
reranker"). The cheapest architecture change that targets near-tie discrimination is a **stronger stage-2
reranker model**: swap/add Qwen3-Reranker-8B (open weights, HF revision 77d193c791ed757ca307ee72715aa132723da912,
int8, greedy, fixed batch order, deterministic, NOT fine-tuned on LongMemEval, so no memorization) over the
SAME EXAM 8 top-20 candidate pool as S11, isolating the 4B->8B scorer upgrade.

## Held-out M split (reused from EXAM 21, seed 21)
~/.local/share/mnemonics-exam21/split.json, regenerated and asserted by the script: dev-M n=249, test-M n=251,
type-stratified. Configs are selected on dev-M only; the reported result is test-M, not inspected during design.
External dev (LoCoMo/pref2) is not used because it does not transfer to M (established 2026-10-25).

## Baseline and configs (frozen grid)
Baseline S11 = the persisted EXAM 11 order = RRF(k=60) of 4B-full and VL-8B-full session_max scores over the
ex8 base order (results/exam11_m_scored/calls.jsonl `base`/`order`). All configs reuse order_fusion_dev's frozen
helpers (session_max, ranks, rrf, ties by base order) over the same ex8top20 full docs; only the stage-2 score
set changes:
- C0 = RRF(4B, VL)      (= S11; step 0 must reproduce the persisted EXAM 11 `order` byte-for-byte)
- C1 = 8B alone
- C2 = RRF(8B, VL)      (replace 4B with 8B)
- C3 = RRF(8B, 4B)
- C4 = RRF(8B, 4B, VL)  (add 8B)

## Selection and gate
- Selection: best **dev-M** overall R@1 among {C1, C2, C3, C4}; ties -> fewer models, then the order C2, C4, C3,
  C1. If none beats C0 (S11) on dev-M R@1, verdict FAIL (the stronger reranker does not help) and no test-M claim
  is made beyond logging the selected config's test-M number.
- Gate (test-M, all): G1 test-M overall R@1 > C0 test-M R@1, McNemar net > 0, p < 0.10; G2 no per-type test-M
  R@1 below its C0 test-M value - 0.005; G3 determinism (8B kernel prints a twice-scored repeat-check with max
  |diff|; the M-apply fusion re-runs byte-identical).
- Expected headroom stated in advance: 0..+8 questions on test-M. A null result (8B ties 4B on these near-ties)
  is a real and likely-ish outcome and will be reported as such.

## Forbidden after this commit
Changing the split/seed/grid/selection/gate; inspecting test-M per-question results before the test-M run;
re-selecting on test-M; adding a config after seeing any 8B score; swapping the candidate pool (must stay
ex8top20, same as S11) so the only variable is the stage-2 scorer. If C0 does not reproduce EXAM 11, stop and
fix the instrument before any 8B comparison.

## Interpretation bounds
Reorder ceiling on full M is ~0.996 (2 questions have gold outside the pool). A stronger reranker over the same
pool cannot exceed that; it can only re-rank near-ties already in the pool. A test-M PASS is a genuine held-out
M gain attributable to the 8B scorer; it does not generalize beyond LongMemEval-M.
