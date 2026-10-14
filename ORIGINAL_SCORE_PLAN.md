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

### Synthetic preference dev set (2026-10-12) -- built, preference lane proven on dev

Why preference first: it is M's worst recall/ranking hole (exam-4 reranked single-session-preference
R@1 0.533, R@10 0.733). LoCoMo is third-person so it cannot exercise first-person preference lanes;
a synthetic first-person set was the stated prerequisite.

- benchmarks/synth_pref_gen.py -- deterministic (seed 11), no LLM. Each question: a GOLD first-person
  preference whose wording is disjoint from the question, buried mid-session in filler, plus topic-echo
  decoys that reuse the question's surface words but state no user preference (news, other people,
  logistics). 55 q, 60 sessions each. This reproduces M's symptom: the real preference statement is
  out-ranked by sessions that merely mention the topic.
- benchmarks/synth_recall_probe.py -- dense MiniLM, session = best 2-turn chunk; local CPU, memory-safe.
- Measured: base R@1 0.000 / R@5 0.545 / R@10 1.000, median gold rank 5 (base R@1 0.0 ~ M's preference
  base ~0.17-0.23: the set is faithful). Preference lane (extract first-person preference statements by
  GENERAL grammatical markers -- no topic words; rank preference questions by a dedicated preference-record
  lane, gating preference-bearing sessions above the rest): R@1 0.909 / R@5 1.000 / R@10 1.000, median
  rank 1. Zero regression, so it clears the Phase-2 acceptance rule on dev.
- HONEST CAVEAT: dev/synthetic proof-of-mechanism ONLY, not an M result. The regex extractor is brittle
  (real M preferences have arbitrary phrasing); the production extractor should be an LLM (augment_prefs),
  which is an external/possibly-paid call needing Atakan's approval. Next real test: implement the lane in
  the pipeline and run one logged M exam on Kaggle; dev success does not transfer until measured on M.

### EXAM 6 (pre-registered 2026-10-12, before any M preference-lane result): preference post-hoc rerank

Policy (benchmarks/pref_lane_rerank.py, FROZEN, developed only on synthetic dev, never on M): over the M
top-30 pool dump (m_poolall_perq retrieved_all_sids + m_poolall_top30 session texts), apply a preference
lane. ROUTER is label-free -- a question is a preference question from its OWN wording, never from the
question_type label; non-matching questions keep base order untouched (so other types cannot regress).
EXTRACTOR + GATE: sessions whose user turns contain a first-person preference statement (general
grammatical markers, no topic words) are ordered first by that statement's similarity to the question.
On dev this moved R@1 0.0 -> 0.909 with zero regression. Run ONCE on M, log overall + by type as is, no
M per-question inspection, no pattern tuning against M. Measures only the RANKING portion (rerank within
the top-30 pool), not recall. If it does not help M, that is logged and justifies an LLM extractor (B).
- 2026-10-12 RESULT: **FAILED on M (regressed slightly), logged as is, not retried.** Overall R@1
  0.800 -> 0.788, R@5/R@10 unchanged. single-session-preference UNCHANGED 0.167/0.533/0.733. Router fired
  on only 6/500, all non-preference false positives (2 multi, 3 ss-user, 1 temporal) which it mildly hurt.
  Aggregate diagnosis (mechanism only, no answer/per-question peeking): the label-free ROUTER matched
  **0/30** preference questions -- real M preference questions are worded nothing like the synthetic ones,
  so the lane never fired where intended; and the EXTRACTOR covers only 19/30 (a catchable first-person
  statement in the top-30). Both failures are the dev-tuned regex not transferring to M phrasing. The
  patterns are NOT tuned to M (that would be memorization). CONCLUSION: the regex approach (A) is dead;
  robustness to arbitrary phrasing needs an LLM router+extractor (B). results/exam6_pref_lane.json.

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
| 2026-10-13 | **EXAM 8**: DETERMINISTIC open-weights Qwen3-Reranker-4B (pinned 22e68366, fp16, Kaggle T4) over the deep top-75 (3 CE windows/session), pre-registered | **0.854** | **0.958** | **0.976** | 500/500, repeat check max |diff| 0, NaN 0, 13350 s. Matches the cloud qwen8b exam-4 R@1 and breaks the 0.952 R@10 ceiling (deep candidates promoted). By type R@1 (exam-4 -> exam 8): ku 0.974->1.000, multi 0.820->0.865, ss-user 0.829->0.871 (R@10 1.000), ss-assistant 1.000->0.982, temporal 0.842->0.782 (WORSE), preference 0.533->0.467 (R@10 0.733->0.867). LoCoMo dev same docs: 4B 0.785, 8B-int8 0.800 vs cloud 8B full chunks 0.860, so the gap is the 3x600-char window preselection more than model size. results/m_qrr4b, exam8_m_scored. |
| 2026-10-14 | **EXAM 9**: deterministic Qwen3-14B judge (NF4 greedy, thinking off, front-k 3) over the EXAM 8 order, pre-registered | **0.872** | **0.966** | **0.976** | 500/500, repeat 0/10, replay --strict 0 failed. temporal 0.782->0.850, multi 0.865->0.887, ku 1.000->0.974, pref 0.467->0.433. results/exam9_m_judge. |
| 2026-10-14 | **EXAM 10**: EXAM 8 scores RRF-fused (k=60, equal) with the same 4B under a personalization instruct, pre-registered | **0.890** | **0.968** | **0.982** | 500/500, repeat |diff| 0, NaN 0, 17144 s. ku 0.987, multi 0.910, ss-assistant 1.000, pref 0.567, ss-user 0.914, temporal 0.827. Best deterministic M. results/exam10_m_scored. |
| 2026-10-14 | **EXAM 12**: EXAM 9 judge over the EXAM 10 order, pre-registered | 0.884 | 0.974 | 0.982 | 500/500, repeat 0/10. pref 0.567->0.367, temporal 0.827->0.857, ku 0.987->0.974. Below EXAM 10 at R@1. results/exam12_m_judge. |
| 2026-10-15 | **EXAM 11**: full-chunk (3000) Qwen3-Reranker-4B + Qwen3-VL-Reranker-8B, RRF k=60, over the EXAM 8 top-20, pre-registered | **0.894** | **0.966** | **0.984** | 500/500, all shards repeat |diff| 0, NaN 0; last 4B shard on Colab T4 (verified bit-identical to Kaggle on 20 q). ku 1.000, multi 0.925, ss-assistant 1.000, pref 0.567, ss-user 0.900, temporal 0.827. Best deterministic M. results/exam11_m_scored. |
| 2026-10-13 | **EXAM 7**: exam-4 top-10 + LLM listwise judge (deepseek-v4-pro via Ollama cloud, reasoning medium, front-3) + final duel (top-2, 8000 chars), pre-registered | **0.910** | 0.944 | 0.952 | 500/500, 0 failed (one strict abort on a timeout, resumed). judge alone 0.906; duel fired on 265. By type R@1 base->final: multi 0.820->0.947, ss-user 0.829->0.900, temporal 0.842->0.895, ku 0.974->0.987, ss-assistant 1.000->0.982 (-1 q), preference 0.533->0.500 (-1 q). R@10 ceiling 0.952 unchanged by design. NOT DETERMINISTIC: cloud LLM, temperature 0 but no bitwise-reproducibility guarantee; kept as an upper-bound reference, not as the final deterministic result. results/exam7_m_judge, exam7_m_duel. |
| 2026-10-12 | **EXAM 4 COMPLETE**: orig-v2 M + qwen3-reranker-8b chunk-max top-10 (chunk-chars 3000, max-chars 6000), evren /v1/rerank, --strict, pre-registered | **0.854** | **0.944** | **0.952** | 500/500, base 0.800/0.924/0.952. R@10 unchanged (rerank inside top-10). By type R@1 base->rerank: preference 0.167->0.533, ku 0.936->0.974, temporal 0.797->0.842, multi 0.782->0.820, ss-assistant 0.982->1.000, ss-user 0.814->0.829. Chunk-max over 3000-char turn chunks fixes the earlier "no gain on M" result (that used whole-session/3000-char-truncated prompts). EXAM4_FINAL.json in results/exam4_m_qwen8b_chunks/. |
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
- 2026-10-11 03:49 part 1 STRICT_ABORT after 141/250 scored: evren code daily_token_limit_exceeded (shared across sessions), resets 2026-10-12T00:00Z. The "auto-resume 03:05 TR" note was aspirational: nothing (cron/launchd) was actually scheduled, so the run sat idle ~2 days.
- 2026-10-12 resumed manually from the Mac (HTTP-only, --strict resume skips the 391 already in calls.jsonl, did the remaining 109). No 429/quota error. **EXAM 4 COMPLETE, 500/500: base 0.800/0.924/0.952 -> reranked 0.854/0.944/0.952** (logged in the exam ledger above). Score computed over the union of part0+part1 calls.jsonl, base order from m_perq retrieved_top10_sids. V2 evren rerun still deferred.

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
- 2026-10-12 DEV GATE PASSED -> **EXAM 5 GREENLIT**. LoCoMo all-conv poolall, top-30 chunk-max rerank, 200 q, evren, 0 failed, p90 1.31 s:
  base 0.705/0.935/**0.965** -> reranked **0.860/0.980/0.990**. Both bars clear (R@10 0.990 > 0.965; R@1 0.860 >= 0.840).
  First evidence that a top-30 pool breaks the R@10 ceiling that top-10 rerank cannot (R@10 moved +0.025). results/exam5_dev_locomo_poolall_top30/.
- BLOCKER: exam 5 needs the M top-30 pool dump (perq with retrieved_all_sids). That dump is 4/10 Kaggle shards done
  (s0,s1,s3,s4 in results/origv2pool_m/); s2 + s5-s9 are missing. Completing it needs Atakan's Kaggle session;
  M evals are never run on the 8 GB Mac (encoder RSS 5.24 GB + 8-34 GB index writes trip the watchdog). The original
  orig-v2 M perq saved only retrieved_top10_sids, so the deeper pool cannot be reconstructed offline; retrieval must re-run.
- 2026-10-12 PREP (no Kaggle, no Mac eval): the 6 missing shard runners are generated and verified -
  benchmarks/kaggle_origv2pool_m_shard{2,5,6,7,8,9}.py. Split convention (confirmed against the 4 done shards):
  shard k = sorted(all 500 M qids)[k::10], 50 qids each, 10-shard union = 500 with no overlap. Each is shard0's
  runner with only the split list, filename and shard number changed; same pinned SHA ca53594, same orig-v2 policy.
  To finish: run these 6 on Kaggle (5 CPU sessions at a time), download each kernel's results_origv2pool_m/perq.json
  to results/origv2pool_m/s<k>/results_origv2pool_m/perq.json, then `python3 benchmarks/merge_m_pool_shards.py`
  (merges 10 shards, aborts unless base reproduces 0.800/0.924/0.952, writes rerank_inputs/m_poolall_perq.json).
  Then the frozen exam 5 command: dev_api_rerank.py --perq m_poolall_perq.json --sessions <M top-30 session texts>
  --sids-key retrieved_all_sids --top-n 30 --chunks --chunk-chars 3000 --batch 64 --timeout 60 --strict.
- 2026-10-12 UNBLOCKED autonomously (Kaggle API token worked, no manual session needed): 9/10 pool shards had already
  completed on Kaggle days earlier and were only un-downloaded; pulled them. Shard 9 had never run -> pushed cpu9,
  it completed (~2 h CPU) and was pulled. merge_m_pool_shards.py merged all 10, base reproduced EXACTLY 0.800/0.924/0.952
  (dump valid) -> rerank_inputs/m_poolall_perq.json. Top-30 session texts: extract_m_candidates.py --sids-key
  retrieved_all_sids --top-n 30 -> rerank_inputs/m_poolall_top30_sessions.json (500/500, 14139 sessions, 0 unresolved).
- 2026-10-12 EXAM 5 STARTED, results/exam5_m_qwen8b_top30/. RUNNING, quota-chunked: no --parts, single --out, --strict;
  it runs until the shared evren daily limit 429s, aborts clean, and the next run (same --out) skips qids already in
  calls.jsonl and continues. Day 1: STRICT_ABORT at 18/500 (today's 15M shared quota was already spent by exam4 resume +
  the dev gate + other sessions). Progress is gated by free daily quota, not compute.
  TOKEN-COST CAVEAT: per-question token counts cannot be read from calls.jsonl -- in chunks mode dev_api_rerank logs only
  the LAST batch's usage per question (the `usage` var is overwritten each batch), so any multi-batch question is undercounted.
  Do not project total cost from these logs. Earlier "~51M tokens" was a bad 3x extrapolation from that undercount; retracted.
  DECISION (Atakan, 2026-10-12): resume MANUALLY each day after the 00:00Z reset, no launchd auto-resume job.
  RESUME COMMAND (from benchmarks/, after quota reset): python3 dev_api_rerank.py
  --perq /Volumes/WD-Backup/datasets/longmemeval/rerank_inputs/m_poolall_perq.json
  --sessions /Volumes/WD-Backup/datasets/longmemeval/rerank_inputs/m_poolall_top30_sessions.json
  --out /Volumes/WD-Backup/datasets/longmemeval/results/exam5_m_qwen8b_top30
  --model qwen3-reranker-8b --sids-key retrieved_all_sids --top-n 30 --chunks --chunk-chars 3000 --batch 64 --timeout 60 --strict
  When calls.jsonl reaches 500, score it the same way exam 4 was scored (union, base from retrieved_top10_sids vs answer_sids).
- 2026-10-12 18:40Z EXAM 5 SUSPENDED at 18/500 (not cancelled, not scored; resumable with the same command).
  Cause, measured from session lengths (not from the undercounting usage log): qwen8b over full top-30 text is
  ~43M tokens (median 87k/q) = ~3 days of the shared 15M/day evren budget, which was exhausted today
  (evren body: daily_token_limit_exceeded, 203 tokens left, resets 2026-10-13T00:00Z). Atakan's directive
  (2026-10-12): single goal M R@1 = 1.0, decide autonomously. Budget goes to the LLM listwise judge instead:
  same candidate ceiling (gold in top-30) at ~7.8M tokens, and it targets R@1 directly.

## LLM listwise judge (2026-10-12, dev first)
benchmarks/llm_judge_rerank.py. For each of the top-N candidates (base order = a previous rerank's `order`, or
the pool order), split the session into messages and long messages into ~600-char windows, score windows
against the question with a local ms-marco-MiniLM-L-12-v2 CE, show the best 3 windows per session with the
session date (plus the question date when known). The LLM returns the session numbers holding the needed
information, most relevant first; those go to the front, the rest keep base order. No question_type, no gold.
Provider: Ollama cloud (deepseek-v4-pro / deepseek-v4.1-flash, reasoning_effort none), independent of the evren
budget. Other providers checked 2026-10-12: DeepSeek direct 402 insufficient balance, Gemini 429 quota, NIM
12-60 s per trivial call (fails the 20 s bar). M question dates: rerank_inputs/m_qdates.json (500/500).

Dev results, LoCoMo all-conv poolall 200 q, base = qwen8b top-30 rerank order (exam-5 dev gate), judge over top-10:
| config | R@1 | R@5 | R@10 | p90 s |
|---|---|---|---|---|
| base (qwen8b) | 0.860 | 0.980 | 0.990 | |
| judge pro, 3 windows, no reasoning, front-5 | 0.870 | 0.980 | 0.990 | 1.3 |
| (replay) front-5 vs front-3 on first 77 q | front-5 lost R@5 (0.987->0.974), front-3 kept it -> front-k 3 | | | |
| judge pro, 6 windows, no reasoning, front-3 | 0.875 | 0.975 | 0.990 | 1.1 |
| judge kimi-k3, 3 windows, no reasoning, front-3 | 0.885 | 0.975 | 0.990 | 1.5 |
| judge pro, 3 windows, reasoning medium, front-3 | **0.885** | 0.980 | 0.990 | 9.0 |
| + final duel (top-2 picks, 8000 chars/session, pro reasoning medium; llm_judge_duel.py) | **0.900** | 0.980 | 0.990 | |
Miss analysis of the reasoning judge (dev, per question allowed): of 23 R@1 misses, 11 had gold at the judge's
second pick, 6 gold not picked, 2 gold outside top-10; earlier window check: ~3/10 misses have answer words
nowhere in the gold session (abstractive answer or label noise). The duel targets the second-pick cases.

### EXAM 7 (pre-registered 2026-10-12, before any M judge call): judge + duel over exam-4 top-10
Frozen: base = exam-4 order (qwen3-reranker-8b chunk-max top-10, 0.854/0.944/0.952; merged part0+part1 calls).
Stage 1: llm_judge_rerank.py --provider ollama --model deepseek-v4-pro:cloud --reasoning medium --max-tokens 8192
--timeout 120 --top-n 10 --windows-per-session 3 --window-chars 600 --front-k 3 --qdates m_qdates.json, window CE
scores from the Kaggle CE kernels (same windows() and ms-marco-MiniLM-L-12-v2; cache m_ce_cache_w600.json).
Stage 2: llm_judge_duel.py --duel-k 2 --duel-chars 8000, same model and reasoning. R@10 cannot move (ceiling 0.952).
Run once on all 500, log overall + by type (base / judge / duel) as is; no M per-question inspection.
- 2026-10-12 21:20Z, before any M judge call, two execution notes (no policy change): (1) window CE scores come
  from the Kaggle GPU kernel mnemonics-m-deep-ce-gpu (same windows(), same ms-marco-MiniLM-L-12-v2, top-75 union
  candidates, covers every exam-4 top-10 session) instead of the slower CPU kernels; GPU vs CPU float noise only.
  Session texts from m_deep_top75_sessions.jsonl (same "[role] content" text). (2) --strict: a failed LLM call
  aborts without writing a row and the run resumes later, so a provider limit can never silently keep base order.
  Ollama Pro 5-hour limit was hit at ~20:00Z by the dev runs (same account as Atakan's svelka); EXAM 7 waits for it.

## Path to R@1 = 1.0: deterministic, verifiable, no memorization (Atakan, 2026-10-13)
Bar: same inputs -> same ranking on re-run; every stage pinned (code SHA, HF model commit) and logged; no tuning
on LongMemEval S/M questions; final independent proof on LongMemEval-V2 (never scored, never trained on).
1. Candidates: deep pool (candidate_k/top_k 300). Aggregate recall: top-10 0.952, top-30 0.984, top-75 0.996,
   top-150 1.000. Recall is solved by depth; the rest is ranking.
2. Deterministic rerank: Qwen3-Reranker (open weights, pinned commit) on Kaggle GPU, fp16, deterministic torch
   flags, built-in repeat check (score the first 20 q twice, report max |diff|). Docs = CE-selected windows.
3. Deterministic judge: open-weights LLM (Qwen3-8B/14B, pinned commit), greedy decoding, same prompts as the
   cloud judge. The cloud judge (EXAM 7, 0.910) is the reference to match or beat.
4. Training if 2-3 fall short: distill the cloud judge into the open reranker/judge on NON-LongMemEval data only
   (LoCoMo train convs + a large LLM-generated first-person synthetic memory set with planted facts, preferences,
   updates and dated events). Earlier CE fine-tune on LoCoMo alone overfit (-9 R@1), so the synthetic set is the
   main training source; held-out LoCoMo dev decides acceptance.
5. Preference is the open problem: judge and regex lanes both failed on M (0.500-0.533 R@1). Needs a capability
   developed on synthetic first-person preference data, not on M.
Each step: dev first, then one pre-registered M exam, logged as is.

### EXAM 8 (pre-registered 2026-10-13, before any M call): deterministic open-weights rerank over the deep top-75
Candidates: m_deep_perq `cands` (exam-4 qwen top-10 first, then the deep pool), top-75 (aggregate recall 0.996).
Docs: make_rerank_docs.py, 3 CE-selected 600-char windows per session with the session date (GPU deep CE cache).
Scoring: kaggle_qwen_rerank_docs.py, Qwen/Qwen3-Reranker-4B @ 22e683669bc0f0bd69640a1354a6d0aebcfeede5, fp16,
generic instruction (no LongMemEval wording), BATCH 16, MAX_LEN 2048, deterministic flags, repeat check on the
first 20 q. Session score = max over its docs (score_rerank_docs.py). Run once, log overall + by type + the
repeat-check |diff| as is. Kernel atakanakbaba/mnemonics-m-qrr4b, dataset atakanakbaba/mnemonics-m-rerank-docs.
- 2026-10-13 run v1 died with CUDA OOM before writing any score (full-vocab logits for every position). Execution fix, no
  scores seen: logits_to_keep=1 (only the last position, which is all the yes/no score uses) and BATCH 8. Re-pushed as v2.

### Dev: deterministic judge (2026-10-13, LoCoMo all-conv 200 q)
Base = deterministic Qwen3-Reranker-4B order (0.785/0.960/0.975). Judge prompts dumped by llm_judge_rerank.py
--dump-prompts (top-10, 3 CE windows, front-3), answered by kaggle_qwen_generate.py: Qwen/Qwen3-14B @ 40c06982,
NF4, greedy, thinking off, 1108 s, repeat check 0/10 mismatches; replayed with --replay-raw --strict.
Result 0.815/0.960/0.975 (+3.0 R@1, no R@5/R@10 change, 0 failed). Cloud judge on the stronger cloud-8B base was
+2.5 (0.860->0.885), so the deterministic judge gives a comparable gain. Remaining gap is the rerank stage.

### EXAM 9 (pre-registered 2026-10-13, before any M call): deterministic judge over the EXAM 8 order
Base = EXAM 8 order (exam8_m_scored/calls.jsonl: deterministic Qwen3-Reranker-4B over the deep top-75,
0.854/0.958/0.976). Prompts: llm_judge_rerank.py --dump-prompts, --top-n 10 --windows-per-session 3
--window-chars 600, question dates (m_qdates.json), GPU deep CE cache, sessions m_deep_top75_sessions.jsonl.
Generation: kaggle_qwen_generate.py, Qwen/Qwen3-14B @ 40c069824f4251a91eefaf281ebe4c544efd3e18, NF4, greedy,
thinking off, MAX_NEW 48, repeat check. Replay: --replay-raw --strict --front-k 3. Fully deterministic. Run once,
log overall + by type as is; no M per-question inspection.
- 2026-10-14 run v1 died with CUDA OOM on prompt ~5 (generate() materialised full-vocabulary fp32 logits for every
  prompt position: 1.94 GiB at ~3.2k tokens; M prompts are longer than LoCoMo's). Execution fix, no scores seen (4 partial
  outputs not read): greedy is now a hand loop with logits_to_keep=1 and the KV cache, same argmax policy. Equivalence
  is checked by rerunning the LoCoMo dev prompts and comparing raw outputs with the generate() run. Re-run as v2.
- 2026-10-14 v2 also died with CUDA OOM, at prompt 4 (14.9k chars; only prompts up to ~11.6k chars fit). Not logits this
  time: full-prompt prefill attention (~1.3 GB fp16 scores per layer at ~3.7k tokens) on top of ~10 GB weights. Execution
  fix, no scores seen: prefill in 1024-token chunks through the KV cache, attn_implementation sdpa, expandable_segments.
  Same greedy argmax policy; equivalence re-checked on the LoCoMo dev prompts with this exact code. Re-run as v3.
- 2026-10-14 v3 equivalence (eq3, LoCoMo dev prompts, same chunked-prefill code): raw outputs equal the generate()
  run on 186/200 prompts; the 14 that differ are mostly reordered/shortened pick tails. Replayed with --front-k 3 the
  scores are identical: 0.815/0.960/0.975 both. Repeat check 0/10.
- **EXAM 9 result (v3, 500 q, 5348 s, repeat check 0/10, replay --strict 0 failed): 0.872 / 0.966 / 0.976**
  (base EXAM 8 0.854/0.958/0.976: +1.8 R@1, +0.8 R@5). By type R@1 base->judge: knowledge-update 1.000->0.974,
  multi-session 0.865->0.887, single-session-assistant 0.982->0.982, single-session-preference 0.467->0.433,
  single-session-user 0.871->0.871, temporal-reasoning 0.782->0.850. Best deterministic M score so far; cloud EXAM 7
  (non-deterministic) 0.910/0.944/0.952 is still higher at R@1. R@10 0.976 is the rerank pool ceiling for this order.

### EXAM 10 (pre-registered 2026-10-14, before any M call): personalization instruct fused with EXAM 8 by RRF
Motivation: preference is the weakest type (EXAM 9 R@1 0.433). The personalization instruct (commit 0e33ba3) alone
regressed LoCoMo R@5 (0.960->0.940) so it was not adopted. Rank-level fusion, checked offline on dev score files:
session score = max over its docs, rank each list (ties by base order), fuse 1/(60+r1) + 1/(60+r2).
Dev, default 4B -> fused: LoCoMo 0.785/0.960/0.975 -> 0.805/0.960/0.975 (no regression); synthetic pref2
0.475/0.762/0.800 -> 0.512/0.812/0.887. Disclosure: on dev I also looked at weights 0.3/0.7, k=10 and a z-score sum;
equal-weight k=60 (the textbook RRF default) is fixed here and was tied-best on LoCoMo.
M run: kaggle_qwen_rerank_docs.py, Qwen/Qwen3-Reranker-4B @22e68366 fp16, RR_INSTRUCT = the personalization instruct
(the same string the LoCoMo/pref2 dev kernels used, set in run.py line 1; 0e33ba3 only added the env override):
"Given a question or request a user makes about their own life, judge whether this excerpt from their past conversations contains information needed to answer it, including anything the user revealed about themselves (what they own, like, dislike, cannot do, or have experienced) that a personalized answer should take into account".
Same docs as EXAM 8 (m_rerank_docs_top75_w3.jsonl), repeat check. Score: score_rerank_docs.py --perq
m_deep_perq.json --sids-key cands --scores m_qrr4b/scores.jsonl --scores2 <new scores>. (The scorer change reproduces
the EXAM 8 order byte-identically without --scores2.) Run once, log overall + by type; no M per-question inspection.
A judge pass over the fused order, if run, is a separate pre-registration.
- **EXAM 10 result (2026-10-14, 500 q, 17144 s, repeat |diff| 0, NaN 0): 0.890 / 0.968 / 0.982** (EXAM 8
  0.854/0.958/0.976, EXAM 9 0.872/0.966/0.976). By type R@1/R@5/R@10: knowledge-update 0.987/1/1, multi-session
  0.910/0.992/1.000, single-session-assistant 1/1/1, single-session-preference 0.567/0.800/0.867, single-session-user
  0.914/0.986/1.000, temporal-reasoning 0.827/0.940/0.962. Best deterministic M score; R@1 vs EXAM 8: ku -1.3,
  every other type up (pref +10.0, multi +4.5, ss-user +4.3, temporal +4.5, ss-assistant +1.8). results/exam10_m_scored.

### EXAM 12 (pre-registered 2026-10-14, before any M call): the EXAM 9 judge over the EXAM 10 order
Identical to EXAM 9 except --base-calls results/exam10_m_scored/calls.jsonl: same prompt builder flags (--top-n 10
--windows-per-session 3 --window-chars 600, m_qdates, GPU deep CE cache, m_deep_top75_sessions), prompts
m_judge_prompts_exam10.jsonl (500), kaggle_qwen_generate.py v3 code (chunked prefill, verified equivalent on dev),
Qwen3-14B @40c06982 NF4 greedy thinking off MAX_NEW 48, replay --strict --front-k 3. Run once, log overall + by type.
- **EXAM 12 result (4915 s, repeat 0/10, replay --strict 0 failed): 0.884 / 0.974 / 0.982** (base EXAM 10
  0.890/0.968/0.982: -0.6 R@1, +0.6 R@5). By type R@1 base->judge: knowledge-update 0.987->0.974, multi-session
  0.910->0.917, single-session-assistant 1.000->1.000, single-session-preference 0.567->0.367, single-session-user
  0.914->0.900, temporal-reasoning 0.827->0.857. The judge undoes the preference gain of EXAM 10 (its prompt asks
  for sessions that answer the question, the same failure the default reranker instruct had) while helping temporal.
  EXAM 10 stays the best deterministic M R@1. results/exam12_m_judge.

### Dev 2026-10-14: open Qwen3-VL-Reranker-8B (the model evren's reranker endpoint routes to) on LoCoMo
First run died: the repo at b212dc8 (its latest commit) lists module 1 at 1_CausalScoreHead but ships 1_LogitScore,
so sentence-transformers built LogitScore without true_token_id. Fix 0400319 (symlink the missing path; weights and
revision unchanged). v2: 8-bit, default prompt, full chunks (3000 chars, same docs as the 4B full run), 4769 s,
repeat |diff| 0, NaN 0. Scored with score_rerank_docs.py on locomo_poolall_perq:
- VL-8B alone 0.815/0.970/0.985 (4B full 0.795/0.980/0.980; cloud 0.860/0.980/0.990). Not the cloud number alone.
- 4B full + VL-8B, equal-weight RRF k=60 (--scores2, fixed before this check): **0.860/0.980/0.985** = cloud R@1,
  deterministic. M uses 3x600-char windows, not full chunks, so the same fusion is being checked on the LoCoMo window
  docs (kernel mnemonics-locomo-qvlrr8b-w3) before any M pre-registration.
- Window docs (3x600, same as M): VL alone 0.790/0.960/0.980; 4B-w3 + VL-w3 RRF 0.800/0.955/0.990 (R@5 -0.5, fails
  the zero-regression rule). The 0.860 comes from full chunks, not from the model pair alone.
- Two-stage simulation on LoCoMo (stage 1 = 4B-w3 order, stage 2 = full-chunk 4B+VL RRF over its top-N, rest kept):
  N=5 0.850/0.960/0.975, 10 0.850/0.970/0.975, 15 0.855/0.980/0.990, 20 0.850/0.980/0.985, 30 0.850/0.980/0.985
  (stage-1 alone 0.785/0.960/0.975). No N regresses.

### EXAM 11 (pre-registered 2026-10-14, before any M call): full-chunk 4B+VL RRF over the EXAM 8 top-20
N=20 fixed because the EXAM 8 order has recall@20 0.986 (R@10 can rise above 0.976) and N=20 does not regress on dev;
N=15 was 0.005 better on dev and is not chosen for that. Docs: make_rerank_docs.py --full-chunks 3000 over the deep
top-75 (m_rerank_docs_top75_full3000.jsonl), kept for the EXAM 8 top-20 sessions only (48946 docs, 118.5M chars,
m_rerank_docs_ex8top20_full3000.jsonl; every top-20 session has docs). Scoring on Kaggle T4, two question shards each:
kaggle_qwen_rerank_docs.py Qwen3-Reranker-4B @22e68366 fp16 default instruct, and kaggle_qwen_vl_rerank_docs.py
Qwen3-VL-Reranker-8B @b212dc8 8-bit default prompt; repeat checks as coded. Score: score_rerank_docs.py --perq
m_ex8_perq.json (m_deep_perq + key ex8 = EXAM 8 order) --sids-key ex8 --scores <4B> --scores2 <VL>; sessions
outside the top-20 keep the EXAM 8 order. Run once, log overall + by type; no M per-question inspection.
- 2026-10-14 status: vl-s0 (248 q, 15134 s), vl-s1 (252 q, 15974 s), 4b-s0 (248 q, 9195 s) done, all repeat |diff| 0,
  NaN 0. 4b-s1 could not start: Kaggle "Maximum weekly GPU quota of 30.00 hours reached". Not scored on the partial
  shards (that would be a look at M before the pre-registered run is complete); scored once 4b-s1 has run.
- 4b-s1 ran on a free Colab T4 (Kaggle quota exhausted): same kaggle_qwen_rerank_docs.py with only the input glob
  root changed (/kaggle/input is read-only on Colab), torch 2.11 / transformers 5.18. Cross-platform check first:
  the first 20 questions of shard 0 (1778 docs) re-scored on Colab vs the Kaggle 4b-s0 scores: max |diff| 0.0,
  session rankings 20/20 identical. 4b-s1: 252 q, 9653 s, repeat |diff| 0, NaN 0. All four shards cover all 500 q with
  doc counts matching the docs file.
- **EXAM 11 result: 0.894 / 0.966 / 0.984** (base = EXAM 8 order, reproduced 0.854/0.958/0.976). By type R@1/R@5/R@10:
  knowledge-update 1/1/1, multi-session 0.925/0.992/1.000, single-session-assistant 1/1/1, single-session-preference
  0.567/0.800/0.900, single-session-user 0.900/0.971/1.000, temporal-reasoning 0.827/0.940/0.962. No type drops at
  R@1 vs EXAM 8. Best deterministic M R@1 and R@10 so far (EXAM 10 0.890/0.968/0.982). results/exam11_m_scored.

### Literature check 2026-10-15: published LongMemEval-M retrieval numbers (self-reported, protocols differ)
- LongMemEval paper (2410.10813) Table 3, Stella V5, session value, K=V+fact: R@5 0.732, R@10 0.862 (recall variant not
  stated in the table; the official eval script reports recall_any and recall_all and skips abstention questions).
- MemGAS (2505.19549) Table 2: R@5 63.62, R@10 77.02. RF-Mem (2603.09250): best R@10 0.6635 (bge-base).
- ReMind (2610.16137) Table 1, frozen top-50, trained on a 60% split of LongMemEval and tested on 40%: R@1 50.07,
  R@5 82.82; Qwen3-Reranker-4B there: R@1 45.08, R@5 79.89 (R@k definition not given; R@1 far below 1 suggests a
  fraction-of-gold recall).
- lcm-x PR #941 (GitHub, not a paper): R@10 0.950 on a 483-question privacy-filtered subset (456 scored).
- Ours under the same variants (EXAM 11): all 500 recall_any 0.894/0.966/0.984; 470 non-abstention recall_any
  0.900/0.968/0.983, recall_all 0.319/0.872/0.923. recall_all@1 cannot reach 1.0 for multi-gold questions, so the
  1.0 target is recall_any. No published M number found that is above ours on a comparable metric.
