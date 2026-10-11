# Original-score plan (no memorization)

Decided: 2026-10-06, Atakan. Goal: an original R@1 = R@5 = R@10 = 1.0, with no memorization.

## Why

- plus58 (exact R@1 0.992) contained 17 memorized lane groups (41 routers). Example: lane Q copies the
  test-question typo `buisiness`; lane T scores gold-session words (`debate team`, `stratocaster`).
- LongMemEval S and M share all 500 question ids. Every S/M score so far was tuned on these questions.
- 2026-10-07, base pipeline audit (verified): the clean 0.934 is not original either.
  - Every M gold session id starts with `answer_`, and no distractor id does. The id is embedded in the
    chunk text as `SID=<sid>|`, which both the encoder and mn-ce-v1 see.
  - mn-ce-v1 (reranker) and chat-ce-v3 (trust gate) were fine-tuned with gold labels on 100 of the 500
    evaluated questions (LongMemEval "train split").
  - The gate margins 3.0 and 4.0 and the couple/mentioned-week tie-break were each chosen to flip one
    question.
  - Fix: run `colab_clean2.py`, which uses an off-the-shelf ms-marco-MiniLM-L-12-v2 CE, no trust gate,
    sha1-anonymized session ids, and no single-question tie-break.
- The offline replay predicted 1.000 but the exact run gave 0.992. Cause: lanes were validated on deeper
  forensic fused@200 pools, while the exact pipeline uses its own, shallower pools.

## Rule zero: M is the exam, never the textbook

1. No per-question inspection of M misses. Only aggregate numbers (overall, by type) are looked at.
2. Any new rule must be domain-independent. Topic words, brands, or phrases taken from a question or
   session are forbidden. Every lane needs a one-line justification of the general mechanism.
3. Exams are logged. Each full M run gets a row in the ledger below, good or bad. No silent reruns.
4. Development happens only on data that is not LongMemEval S/M:
   - LoCoMo (local: `benchmarks/data/locomo/locomo10.json`, 10 conversations, 1986 QA).
   - A synthetic memory-QA set we generate ourselves, covering the 6 LongMemEval question types.
5. Final independent proof: LongMemEval-V2 (451 new questions, frozen core, never scored).

## Phase 0: clean baseline (running)

- Runner: `~/.cache/mnemonics-scale-lab/local_clean_full.py`.
- Contents: plus58 minus the memorized lanes, plus the K7 full-timestamp fix.
- Output: the honest starting score, broken down by type.

## If the clean score is low

The low number is reported as is and becomes the baseline. Nothing is rerun until it looks better.

1. Compare the by-type breakdown with plus58: which types lost the most? This is an aggregate view
   only, with no per-question reading of M.
2. Run the same clean policy on the LoCoMo dev set (`local_clean_locomo.py`). If the same type is weak
   on dev too, the gap is a real capability gap, so work on it on dev.
3. Map each weak type to a Phase 2 capability:
   - temporal: relative-date window + timestamps
   - preference: profile extraction
   - multi-session: deeper pool + aggregate rerank
   - knowledge-update: recency
4. Build one capability at a time on dev. Accept it only if dev improves with zero dev regressions.
   Then take one M exam and log it.
5. The 22 TAILORED-GRAMMAR lanes stay for now, but each must be re-proven on dev. If a lane does not
   help on dev, delete it even if M drops.

## Phase 1: dev data

- Build an adapter so LoCoMo runs through the same retrieval and scoring.
- Generate the synthetic set from personas with dated multi-session chats. Questions include:
  - relative time ("N weeks ago")
  - "most recently"
  - knowledge updates
  - preference/advice
  - abstract-to-concrete paraphrase
  - typos

## Phase 2: general capabilities

Each one is accepted only if it improves dev and harms nothing on dev.
The 4-miss diagnosis is the motivation; the fixes must not be shaped to those 4 questions.

| Capability | Gap it closes | Existing hook |
|---|---|---|
| Query rewrite / HyDE: the LLM writes a concrete hypothetical memory for an abstract question | "business milestone" vs "signed first client contract" | `--hyde` |
| Deeper candidate pool + stronger rerank together | gold never entered the pool (2 cases) | `--candidate-k` (depth alone was falsified at k=200) |
| LLM rerank of top-N sessions | CE scored all date-window candidates ~0 | `--llm-rerank-top-n` |
| Relative-date window as a general router (any "N units ago", not 3-condition grammar) | lane I only fired on "what was" | lane I mechanism |
| Typo-robust matching (char n-gram / spell normalization) | "buisiness" | char TF-IDF already in lane I |
| Ingest-time preference/profile extraction | preference questions with no lexical overlap | `augment_prefs` |

Constraint: memory query latency must stay at or below 20 s (Atakan's bar). LLM steps must fit inside it.

### Notes (2026-10-07)

- LoCoMo questions are third person ("When did Caroline ..."). Most M lanes route on first-person grammar
  ("did I", "I mentioned"), so they never fire on LoCoMo. LoCoMo therefore cannot re-prove them.
  A synthetic set written in LongMemEval first-person style is required for that.
- `--hyde` and `--llm-rerank-top-n` call an OpenAI-compatible LLM. DeepSeek is used if `DEEPSEEK_API_KEY`
  is set, otherwise NVIDIA NIM. Both keys exist locally. Using them is an external, possibly paid, call,
  so it needs Atakan's approval before the first experiment.
- Never run M or LoCoMo evals on the 8GB Mac. They caused watchdog resets (5.24GB process, 8-34GB index
  writes). Use Colab: session `mn-gpu` (T4) and the scripts `colab_*.sh` in `~/.cache/mnemonics-scale-lab/`.

### Target arithmetic (clean baseline, 2026-10-07)

The goal is R@1 = R@5 = R@10 = 1.0 on M. Two different gaps stand between the baseline and that goal:

- Top-1 selection gap (R@10 hit but R@1 miss): 0.966 - 0.934 = 16 questions. A final-step reranker,
  such as an LLM over the top-5, can close this gap.
- Recall gap (gold not in the top 10): 1 - 0.966 = 17 questions. No reranker can fix these, because the
  gold session never reaches the candidate list. They need better candidates:
  - ingest-time preference/profile extraction (preference R@10 is 0.900)
  - query rewrite / HyDE
  - a deeper pool fused with a stronger first-stage ranker

Because of the recall gap, R@5 and R@10 cannot reach 1.0 through reranking alone. Recall work is a
first-class track, not a follow-up.

## Phase 3: exam

- Freeze the code, then run M full-500 once and log it.
- Then run V2 once with the official reader and judge, and log it.

## Exam ledger

| date | policy | R@1 | R@5 | R@10 | note |
|---|---|---|---|---|---|
| 2026-10-06 | plus58 (memorized) | 0.992 | 0.994 | 0.994 | invalid for claims, memorized lanes |
| 2026-10-09 | **ORIG-V2 M**: no lanes, no temporal flags, no gate, off-the-shelf CE, anonymized SIDs; generic choices only | 0.800 | 0.924 | 0.952 | By type R@1: ku 0.936, multi 0.782, ss-assistant 0.982, preference 0.167, ss-user 0.814, temporal 0.797. Same generic setup with the 22 tailored lanes (clean2) scored 0.848, so on M the lanes add +4.8 R@1 and nothing to R@5/R@10. /Volumes/WD-Backup/datasets/longmemeval/results/origv2_colab_20261009/m/ |
| 2026-10-09 | **ORIG-V2 S**: no lanes, no temporal flags, no gate, off-the-shelf CE, anonymized SIDs; generic choices only (turn chunks, candidate_k 50, top_k 50) | 0.952 | 0.986 | 0.994 | Colab T4. turn chunks and candidate_k 50 were likely picked earlier from S experience: generic, not question-specific. /Volumes/WD-Backup/datasets/longmemeval/results/origv2_colab_20261009/s/ |
| 2026-10-08 | **FULLY ORIGINAL** S: library defaults only (no lanes, no temporal flags, no gate, off-the-shelf CE, anonymized SIDs) | 0.768 | 0.790 | 0.790 | R@5=R@10 because the default top_k is 10 chunks (about 5 sessions), so R@10 is under-measured; R@1 is valid. |
| 2026-10-08 | **FULLY ORIGINAL** M: same | 0.618 | 0.668 | 0.668 | Same caveat. By type R@1: ss-user 0.243, preference 0.300, ku 0.641, multi 0.692, temporal 0.699, ss-assistant 0.857. /Volumes/WD-Backup/datasets/longmemeval/results/orig_colab_20261008/ |
| 2026-10-08 | clean2 + evren top-5 rerank, frozen dev settings (one-shot M exam) | 0.858 (base 0.848) | 0.928 | 0.948 | 14 fixed / 9 broken; 59% timeouts at 20 s. Cause, measured: M candidate sessions have a median of 14,266 chars, but the prompt showed the first 3,000 chars only. LoCoMo sessions are short, so dev did not reveal this. Next: query-relevant snippet selection, developed on dev. /Volumes/WD-Backup/datasets/longmemeval/results/clean2_m_evren_exam/ |
| 2026-10-08 | clean2 on **S** (Colab T4, colab_s_clean2.py) | 0.942 | 0.988 | 0.994 | previously claimed S 1.000 with LME-trained CEs + answer_ leak; same 500 questions as M. Results: /Volumes/WD-Backup/datasets/longmemeval/results/clean2_s_colab_20261008/ |
| 2026-10-08 | **clean2 = first original baseline** (Kaggle CPU, 10 shards, off-the-shelf ms-marco CE, no gate, anonymized SIDs) | **0.848** | **0.928** | **0.948** | by type R@1: ku 0.962, multi 0.835, ss-assistant 0.982, preference 0.233, ss-user 0.886, temporal 0.857. /Volumes/WD-Backup/datasets/longmemeval/results/clean2_m_kaggle_cpu/MERGED_summary.json |
| 2026-10-07 | clean baseline (Colab T4, colab_clean.py sha 3ce50d91), CONTAMINATED: LME-trained CEs + answer_ SID leak | 0.934 | 0.960 | 0.966 | by type R@1: ss-assistant 1.000, ss-user 0.914, multi 0.955, ku 0.987, temporal 0.940, preference 0.600. Results: /Volumes/WD-Backup/datasets/longmemeval/results/clean_m_colab_20261007/ |

## Dev ledger (LoCoMo, not an exam; may be looked at per question)

| date | policy | R@1 | R@5 | R@10 | note |
|---|---|---|---|---|---|
| 2026-10-08 | bge-v2-m3 CE + evren top-5 rerank (same 200 q, seed 11) | 0.820 (base 0.710) | n/a | n/a | Paired with ms-marco+evren on the same 200: 0.820 vs 0.845, top-5 ceiling 0.910 vs 0.925. bge's gain does not survive the LLM rerank. DECISION: keep ms-marco CE + evren rerank. 44 questions time out in both runs; this is the next lever. |
| 2026-10-10 | M exam: orig-v2 + qwen3-reranker whole-session rerank (LoCoMo-frozen settings), STOPPED at 396/500 | 0.843 vs base 0.864 on the same 396 | n/a | n/a | HURT on M: on 219 successful calls R@1 0.881 -> 0.845. 145 HTTP 400 (requests too long) and 32 HTTP 429 (rate limit) fell back to base order. Cause: M sessions are long (median 14k chars, assistant-heavy), so whole-session scoring dilutes the evidence; LoCoMo sessions are short and hid this. Stopped early to stop burning evren quota. Next: chunk-level rerank (max over turn chunks), validated on dev first. |
| 2026-10-10 | clean2 + **qwen3-reranker-8b** (evren /v1/rerank) final-step rerank, top-10, full session text, 429 backoff | 0.849 | 0.938 | 0.943 | All 1536 q, 4 failed calls, p50 0.51 s. +11.1 R@1 over base 0.738. FROZEN for the M exam: top-n 10, max-chars 100000. /Volumes/WD-Backup/datasets/longmemeval/results/locomo_clean2_qwenrr_full_v3/ |
| 2026-10-10 | CE fine-tune on LoCoMo TRAIN (7 convs, 10,319 pairs, 2 epochs) vs base ms-marco, held-out LoCoMo DEV (conv-30/43/48, 450 q), orig-v2 pipeline | ft 0.682 vs base 0.773 | ft 0.911 vs 0.931 | ft 0.971 vs 0.962 | Fine-tuning HURT R@1 by 9 points (overfit: train loss reached 0.015; single-turn training text vs two-turn chunks at inference). Discarded. Kaggle atakanakbaba/mnemonics-train-ce-cpu; /Volumes/WD-Backup/datasets/longmemeval/results/train_ce_kaggle/ |
| 2026-10-08 | clean2 + **bge-reranker-v2-m3** CE (off-the-shelf, free) | 0.764 | 0.923 | 0.950 | +2.6 R@1 over ms-marco on all 1536; latency on the Mac not measured yet (568M params). /Volumes/WD-Backup/datasets/longmemeval/results/locomo_bge_colab_20261008/ |
| 2026-10-08 | clean2 + **evren glm-5.3 top-5 rerank** (200 random q, seed 11, 20 s timeout, base fallback) | 0.845 (base 0.705 on same 200) | n/a | n/a | 29 fixed / 1 broken; 52/200 timed out (26%) and kept base order, so the gain is a lower bound; top-5 ceiling 0.925; 504k tokens. /Volumes/WD-Backup/datasets/longmemeval/results/locomo_clean2_evren_rr200/ |
| 2026-10-08 | **clean2 dev baseline** (colab_locomo_clean2.py, off-the-shelf ms-marco CE, no gate) | 0.738 | 0.917 | 0.943 | Beats the LME-trained-CE dev run (0.690): independent evidence that mn-ce-v1 overfit LongMemEval. By category R@1: cat1 0.614, cat2 0.760, cat3 0.391, cat4 0.810. Results: /Volumes/WD-Backup/datasets/longmemeval/results/locomo_clean2_colab_20261008/ |
| 2026-10-07 | clean baseline (colab_locomo.py) | 0.690 | 0.907 | 0.938 | 1536 q, own-conversation haystack (19-35 sessions). By LoCoMo category R@1: cat1 0.585, cat2 0.707, cat3 0.424, cat4 0.748. Results: /Volumes/WD-Backup/datasets/longmemeval/results/locomo_dev_colab_20261007/ |

## Dev: embedder recall, LoCoMo (2026-10-10, Kaggle CPU, benchmarks/dev_embed_recall_locomo.py)
Pure dense retrieval, 2-turn chunks, session = best chunk, 1536 q (cat 5 excluded), 272 sessions.
| model | pool | R@1 | R@5 | R@10 | R@30 |
|---|---|---|---|---|---|
| MiniLM-L6 | own conv | 0.4336 | 0.7461 | 0.8724 | 0.9974 |
| MiniLM-L6 | all 10 convs | 0.4076 | 0.7246 | 0.8496 | 0.9714 |
| Qwen3-Embedding-0.6B | own conv | 0.6237 | 0.8704 | 0.9427 | 0.9980 |
| Qwen3-Embedding-0.6B | all 10 convs | 0.6159 | 0.8620 | 0.9355 | 0.9941 |
Caveat: dense only (no BM25/RRF/CE), LoCoMo not M. Next: swap MiniLM for Qwen3-Embedding in the orig-v2 pipeline and widen rerank pool to 30, validate on LoCoMo, then one logged M exam.

## V2 scoring, NIM, informational only (2026-10-10)
Reader openai/gpt-oss-20b, judge z-ai/glm-5.3 (NOT the official Qwen3.5-9B / GPT-5.2 setup). Reader was picked only because it answered while evren quota was out; no reader comparison was done, so these numbers say nothing about memory quality.
- web: 451/451 scored, accuracy 0.2129
- enterprise: crashed at 164/451 (judge returned empty content, harness raised ValueError), partial accuracy 0.2073
- first attempt hung 7.5 h on one judge call (harness default timeout 43200 s); rerun used --timeout-seconds 120 --evaluator-timeout-seconds 120
Next: rerun both domains on evren after quota reset, reader deepseek-v4-flash, judge glm-5.3, --evaluator-max-completion-tokens 16384.

## Dev: Qwen3-Reranker-0.6B on LoCoMo, Kaggle CPU (2026-10-10) - INCOMPLETE
Killed by Kaggle 12 h limit after 41/200 questions (~1000 s per question on CPU). Last printed running R@1 at q=40:
base 0.732, whole-session 0.659, chunk-max 0.780. n=41 is too small for a decision; direction matches the M diagnosis (whole-session rerank hurts, chunk-max helps). Replaced by qwen3-reranker-8b chunk mode via evren /v1/rerank after quota reset.

## Dev: qwen3-reranker-8b chunk-max on LoCoMo, evren /v1/rerank (2026-10-11 03:05)
200 q (seed 11), top-10 of the LoCoMo orig-v2-style perq, chunk 3000 chars, session = best chunk.
base R@1/5/10 0.705 / 0.925 / 0.945 -> reranked 0.840 / 0.935 / 0.945. failed 0, latency p50 0.85 s, p90 1.22 s.
Evren usage shows routed_model qwen/qwen3-vl-reranker-8b. ~5.7k tokens per question.

## EXAM 4 (pre-registered 2026-10-11, before any M call): orig-v2 M top-10 + qwen3-reranker-8b chunk-max
Frozen: dev_api_rerank.py --perq m_perq.json (orig-v2 M, base 0.800/0.924/0.952) --sessions origv2_m_top10_sessions.json
--chunks --chunk-chars 3000 --batch 64 --top-n 10 --timeout 60 --strict, model qwen3-reranker-8b. No M per-question inspection.
R@10 cannot change (rerank inside top-10). Cost ~17.4M tokens > 15M evren daily limit, so it runs as --parts 2 (part 0 on 2026-10-11, part 1 after the next reset).
--strict aborts on any failed call so quota errors can never silently fall back to base order; an aborted part is resumed on the
remaining questions only and the abort is logged here. V2 evren rerun deferred behind this exam.
- 2026-10-11 03:18 part 0 STRICT_ABORT after 71 questions: HTTP 429 survived 5 retries (62 s). A probe right after succeeded, so it was a rate limit, not the daily quota. Fix (no effect on scores): 8 retries up to 256 s, and resume skips qids already in calls.jsonl. Exam score is computed from calls.jsonl over all 500 when both parts finish.
- 2026-10-11 03:30 part 0 complete (250 q, 8.5M tokens): 250 base [0.812, 0.916, 0.948] reranked [0.872, 0.944, 0.948]. Part 1 (250 q) after next evren reset.
- 2026-10-11 03:49 part 1 STRICT_ABORT on its first question (0 scored): evren code daily_token_limit_exceeded, 11k of 15M left (shared across sessions), resets 2026-10-12T00:00Z. Part 1 auto-resumes 2026-10-12 03:05 TR; V2 evren rerun after it.

## Dev: orig-v2 pipeline, MiniLM vs Qwen3-Embedding-0.6B, LoCoMo own-conv (2026-10-11, Kaggle CPU) - Qwen arm INCOMPLETE
MiniLM arm, 400 q (seed 11, 4 shards): R@1/5/10 0.720 / 0.9175 / 0.960.
Qwen arm killed by 12 h limit after 80 q (20/25/20/15 per shard, ~2000 s/q on CPU). Same first-80 questions, read from the progress lines:
MiniLM 56/74/76 hits vs Qwen 56/72/75 -> R@1 0.700 vs 0.700, R@5 0.925 vs 0.900, R@10 0.950 vs 0.9375.
Inside the full hybrid pipeline (BM25 + RRF + CE) the stronger embedder gave no gain on these 80 q, unlike the dense-only test
(R@10 0.85 -> 0.94). Reading: BM25 + CE already cover what the dense side misses; remaining R@10 misses are a ranking problem
(gold is in the 50-chunk candidate band but the final order puts it below 10 sessions), not a candidate-recall problem.
Next: 8B chunk rerank over a wider session pool (top-30 instead of top-10) so R@10 can move; needs perq with top-30 sids.

## Dev: pool depth, orig-v2 MiniLM, LoCoMo own-conv 400 q (2026-10-11 04:57)
Base reproduced exactly (0.720 / 0.9175 / 0.960). Distinct sessions in the 50-chunk result: median 20 (13-27).
Gold within top-k sessions: k=20 0.9925, k=30 0.995, whole pool 0.995.
Caveat: own-conv haystack is only 19-32 sessions, so top-20 covers most of it; this does not transfer to M (~500 sessions).
Next dev step: same check on LoCoMo with all 10 conversations as haystack (272 sessions) before any wider-pool exam.

## Dev: pool depth, orig-v2 MiniLM, LoCoMo all-10-conv haystack 200 q (2026-10-11 06:06)
Haystack = all 272 LoCoMo sessions per question (seed 11, 4 Kaggle CPU shards). R@1/5/10 0.705 / 0.935 / 0.965.
Distinct sessions in the 50-chunk result: median 31 (18-43). Gold within top-k sessions: k=20 0.995, k=30 0.995, whole pool 0.995.
With 9x more distractor sessions the pool still holds gold at k=20 for 199/200, so on LoCoMo the R@10 misses are ranking misses.
Not yet shown for M (~500 sessions, long assistant-heavy sessions). Merged perq: rerank_inputs/locomo_poolall_perq.json.
Next: qwen3-reranker-8b chunk-max over top-30 of this pool (dev, ~3.4M evren tokens), after exam 4 part 1 on the next reset.

## M pool dump for a wider-rerank exam (started 2026-10-11, Kaggle CPU, 10 shards of 50 q)
Same orig-v2 M runner (pinned SHA, same splits), one change: perq also records retrieved_all_sids (every session in the 50-chunk result).
Retrieval is deterministic, so R@1/5/10 must reproduce 0.800 / 0.924 / 0.952; any drift voids the dump.
No M per-question inspection and no M gold-in-pool statistics are computed before the exam; pool depth 30 is fixed from LoCoMo.
Session texts are pulled afterwards with extract_m_candidates.py --sids-key retrieved_all_sids --top-n 30.
Runner: benchmarks/kaggle_origv2pool_m_shard0.py (shards 1-9 differ only in the split id list). Kaggle allows 5 CPU sessions, so shards 5-9 start when 0-4 finish.

## EXAM 5 (pre-registered 2026-10-11, conditional, before any M pool call)
Policy: orig-v2 M + qwen3-reranker-8b chunk-max over the top-30 sessions of retrieved_all_sids (R@10 can move).
Frozen: dev_api_rerank.py --sids-key retrieved_all_sids --top-n 30 --chunks --chunk-chars 3000 --batch 64 --timeout 60 --strict.
Runs only if the LoCoMo all-conv dev run with these exact settings (200 q, seed 11) gives R@10 above its base 0.965 and R@1 not
below the top-10 rerank level seen on LoCoMo (0.840). If the dev run fails that bar, exam 5 is cancelled and the cancellation logged here.
