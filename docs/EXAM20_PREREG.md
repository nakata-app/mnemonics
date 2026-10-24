# EXAM 20 pre-registration: deterministic hypothetical-memory query view (HyDE) as a third stage-2 signal

Written 2026-10-24 after EXAM 19 fell, BEFORE any rewrite or score exists. Goal unchanged (M R@1 = 1.0,
deterministic, no memorization). Incumbent: EXAM 11 / S11 (M 0.894/0.966/0.984).

## Hypothesis
The reranker scores a *question* against *statements*. For preference, multi-session and abstract questions the
gold chunk shares little surface form with the question ("what should I cook" vs "I'm vegetarian and hate cilantro").
A deterministic LLM rewrite of the question into the one sentence the speaker most likely said (hypothetical memory)
gives the same 4B reranker a statement-vs-statement view. Fused by rank with the two existing stage-2 signals it should
lift top-1 where the question-view signals are near-tied. Query rewrite / HyDE has been listed since 2026-10-07 (phase 2
table) and never run as an exam. Expected headroom stated now: 0..+8 q on M; cannot reach 1.0 alone.

## Frozen pieces
Rewriter: Qwen/Qwen3-14B rev 40c069824f4251a91eefaf281ebe4c544efd3e18, NF4, greedy, thinking off, max_new_tokens 96,
Kaggle kernel = copy of mnemonics-exam18-qwen14b-judge run.py (repeat-check 10). Prompt (one for all sets, fixed):
  "Question asked about past conversations: {question}\nWrite the ONE sentence from a past conversation that would
   answer it, phrased exactly as the speaker would have said it at the time (first person, concrete, specific).
   Output only that sentence."
Reranker: Qwen/Qwen3-Reranker-4B rev 22e68366 fp16, same INSTRUCT, same full-3000 docs files, kernel = copy of
mnemonics-locomo-qrr4b-full; only the `question` field of each docs row is replaced by the rewrite (H) or by
"{question}\nLikely memory: {rewrite}" (Q+H). Stage-1 w3 order, RRF k=60, top-20 scope, tie rule, tail: unchanged.

## Fixed configs (nothing else may be reported as the exam)
- C1 "H3":  stage-2 = RRF(4B-full(q), VL-full(q), 4B-full(H))         -- 3 signals
- C2 "QH2": stage-2 = RRF(4B-full(Q+H), VL-full(q))                   -- replaces the 4B question view
Selection: higher LoCoMo R@1, then R@5, then fewer breaks.

## Dev gate (LoCoMo 200 + pref2 80; all must hold for the selected config)
- LoCoMo R@1(C) - R@1(S11) >= +0.015 AND no LoCoMo R@k(C) < R@k(S11) - 0.005
- pref2 R@1(C) >= R@1(S11) - 0.005 AND pref2 R@10(C) >= R@10(S11) - 0.005
- LoCoMo paired net > 0 with McNemar p < 0.10
- Rewriter repeat-check 0 mismatches; reranker repeat |diff| 0; scorer byte-identical
FAIL -> no M run, M untouched, falsification recorded.

## M exam (conditional on PASS, one deterministic pass)
Rewrites for the 500 M questions are generated in the same kernel run as dev (cheap) but their file is NOT opened
before the gate. If PASS: 4B-full(H or Q+H per selected config) over m_rerank_docs_ex8top20_full3000.jsonl (2 shards
like EXAM 11), then the selected fusion with the persisted EXAM 11 shards. Step 0: reproduce EXAM 11 0.894 with the
instrument (pooling_dev P0 path). Report overall + by type + paired. Success = R@1 > 0.894, no R@k regression.

## Forbidden
Editing the prompt after seeing any rewrite; adding configs; choosing by pref2 or M; dropping pref2; reading M rewrites
before the gate; counting a non-reproduced step 0 as a gain; mixing H on dev with Q+H on M.
