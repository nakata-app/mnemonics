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
| 2026-10-08 | clean2 on **S** (Colab T4, colab_s_clean2.py) | 0.942 | 0.988 | 0.994 | previously claimed S 1.000 with LME-trained CEs + answer_ leak; same 500 questions as M. Results: /Volumes/WD-Backup/datasets/longmemeval/results/clean2_s_colab_20261008/ |
| 2026-10-08 | **clean2 = first original baseline** (Kaggle CPU, 10 shards, off-the-shelf ms-marco CE, no gate, anonymized SIDs) | **0.848** | **0.928** | **0.948** | by type R@1: ku 0.962, multi 0.835, ss-assistant 0.982, preference 0.233, ss-user 0.886, temporal 0.857. /Volumes/WD-Backup/datasets/longmemeval/results/clean2_m_kaggle_cpu/MERGED_summary.json |
| 2026-10-07 | clean baseline (Colab T4, colab_clean.py sha 3ce50d91), CONTAMINATED: LME-trained CEs + answer_ SID leak | 0.934 | 0.960 | 0.966 | by type R@1: ss-assistant 1.000, ss-user 0.914, multi 0.955, ku 0.987, temporal 0.940, preference 0.600. Results: /Volumes/WD-Backup/datasets/longmemeval/results/clean_m_colab_20261007/ |

## Dev ledger (LoCoMo, not an exam; may be looked at per question)

| date | policy | R@1 | R@5 | R@10 | note |
|---|---|---|---|---|---|
| 2026-10-08 | **clean2 dev baseline** (colab_locomo_clean2.py, off-the-shelf ms-marco CE, no gate) | 0.738 | 0.917 | 0.943 | Beats the LME-trained-CE dev run (0.690): independent evidence that mn-ce-v1 overfit LongMemEval. By category R@1: cat1 0.614, cat2 0.760, cat3 0.391, cat4 0.810. Results: /Volumes/WD-Backup/datasets/longmemeval/results/locomo_clean2_colab_20261008/ |
| 2026-10-07 | clean baseline (colab_locomo.py) | 0.690 | 0.907 | 0.938 | 1536 q, own-conversation haystack (19-35 sessions). By LoCoMo category R@1: cat1 0.585, cat2 0.707, cat3 0.424, cat4 0.748. Results: /Volumes/WD-Backup/datasets/longmemeval/results/locomo_dev_colab_20261007/ |
