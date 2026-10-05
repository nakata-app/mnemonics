# Mnemonics scale roadmap: M -> 100M -> 1B

## North star

Retrieve the right memory from lifetime-scale agent history without making the
expensive ranking cost grow linearly with corpus size.  The intended scale
ladder is LongMemEval-S -> LongMemEval-M (~1-1.5M-token stress) -> synthetic
100M-token corpora -> 1B-token corpora.

The target is not only benchmark accuracy.  Every scale gate must also record
candidate recall, p50/p95/p99 query latency, ingest/index cost, resident memory,
and index size.

## Verified baseline (2026-10-05)

- LongMemEval-S R1000 exact: R@1/R@5/R@10 = 1.000/1.000/1.000.
- LongMemEval-M same full-stack policy: R@1/R@5/R@10 =
  0.850/0.952/0.964 (500 questions).
- M R@1 misses: 75.  Correct answer is already rank 2 on 31 misses and rank 3
  on another 11.
- M fused cheap-retrieval recall: R@50=0.978, R@100=0.992, R@200=1.000.
- 67/75 full-stack R@1 misses have the gold session inside the separate fused
  top-50 probe.  Most of the observed loss is therefore ranking/policy, not
  total semantic blindness.
- Weakest M categories: single-session-preference R@1=0.500,
  temporal-reasoning=0.805, multi-session=0.850, single-session-user=0.857.

## Falsified shortcuts

1. **Just increase CE candidate_k to 200.** It recovers the cheap-retrieval
   ceiling but makes expensive CE work scale ~4x versus candidate_k=50.
2. **Top-200 chunks -> first 50 unique sessions.** On the 11 fused@50 ceiling
   misses, gold unique-session ranks reach 88; only 6/11 are <=50.
3. **Universal per-session RRF/sum aggregation.** It helps some ceiling misses
   but harms others; no tested aggregation puts all 11 inside 50.
4. **Timestamp-only routing.** For the 20 query-only temporal-target routes,
   nearest session timestamps place only 10/20 gold sessions inside top 50.
   Event time and session time are not interchangeable.
5. **Preference-only retrieval as a replacement.** A dedicated extracted-memory
   lane is complementary, not sufficient by itself. On the 30 M preference
   questions it has R@50=0.733 after adding explicit user-owned objects, while
   existing semantic top-10 UNION preference top-50 has 30/30 gold coverage.
6. **Generic CE over the preference union.** mn-ce-v1 drops preference R@1 from
   0.500 to 0.467; chat-ce-v3 is worse. Candidate generation and personalized
   memory usefulness are distinct ranking problems.
7. **Global user-turn-max CE.** Re-scoring every top-10 session by its best
   user-authored turn is not globally safe: M moves only 425/500 -> 426/500 and
   harms 9/56 single-session-assistant questions. It is useful only behind a
   query lane.

## Proven query-lane gains

- **Temporal-duration event-first:** on all 45 query-only duration routes, strip
  removable count wording (e.g. "How many weeks ago did I attend X?" ->
  "I attend X"), then choose the top-10 session with the strongest matching
  user turn. Result: 35/45 -> 42/45, **+7 fixes / 0 harms**.
- **Aggregate/comparison evidence:** a deliberately narrow query-only grammar
  (``in total``, total number/cost/weight/amount, ``compared to``, min/max
  amount, ``most money``, ``how many different``) selects 47/500 M questions;
  45 are multi-session. Re-ranking only this lane by strongest user turn moves
  34/47 -> 41/47, **+7 fixes / 0 harms**.
- The duration and aggregate fixes are disjoint in the current replay, so their
  offline combined ceiling is 425/500 -> 439/500 (R@1 0.850 -> 0.878) before
  an exact end-to-end Kaggle confirmation.
- **Preference candidate union:** semantic top-10 UNION extracted-preference
  top-50 contains the gold session on **30/30** M preference questions after
  indexing explicit user-owned objects. Final personalized selection remains
  unsolved; generic CE selectors regress R@1.

## Architecture to test

```text
query
  -> label-free memory intent router
       -> semantic lane (high-recall cheap retrieval)
       -> temporal/event lane (event-time index, not only session timestamp)
       -> preference lane (derived preference/user-owned-memory records)
       -> personal-state/fact lane
       -> chronology/multi-session lane
  -> bounded union of session candidates
  -> session evidence aggregation
  -> expensive CE only on a small fixed band (target 20-50)
  -> final policy / trust gate
```

The expensive stage must remain bounded as corpus size grows.  Scaling 100M ->
1B should primarily expand cheap indexes, not CE pair count.

## Immediate M gates

1. Query-only router has no access to LongMemEval question_type or answers.
2. Validate rich preference augmentation on M; keep it only if it improves M
   without regressing the exact S proof.
3. Add event-time metadata extraction/indexing and use it as a candidate lane.
4. Add a session-level candidate union with explicit provenance per lane.
5. Re-run M, inspect changed QIDs, and require no gold-label routing.
6. Re-run exact S and require R@1=1.000 before claiming an improvement.

## Scale gates after M

### 100M

Generate a reproducible corpus with planted episodic, preference, state-update,
temporal and multi-session memories.  Report Recall@1/5/10 plus candidate
coverage and latency/resource distributions.  The benchmark must include hard
semantic distractors, contradictory updates and dense same-day events.

### 1B

Reuse the same protocol at 10x corpus size.  The pass condition is not only
accuracy: expensive reranker work should remain approximately bounded by the
candidate budget while p95/p99 growth is dominated by cheap-index lookup.
