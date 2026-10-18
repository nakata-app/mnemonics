# Mnemonics Final-Duel Switch — Active Checkpoint

Updated: 2026-10-18 (repo HEAD `4bd8c81`; cont. 7 gate results appended below)

Everything below is dev-only (LoCoMo dev, 200 questions, one fixed deterministic judge order). No LongMemEval-M
data was read or written by any of these runs.

## Ground truth (verified, replayable offline)

| run | R@1 | R@5 | R@10 | paired vs judge | McNemar |
|---|---|---|---|---|---|
| base = 4B CE order | 0.785 | 0.960 | 0.975 | — | — |
| judge = Qwen3-14B deterministic listwise | 0.815 | 0.960 | 0.975 | — | — |
| duel 14B @40c0698 NF4 (MAX_NEW 48, bare index) | 0.775 | 0.960 | 0.975 | 12F / 20B | p=0.215 |
| duel **32B @9216db5 NF4** (MAX_NEW 512, reasons) | **0.830** | 0.960 | 0.975 | 10F / 7B | p=0.629 |
| duel deepseek-v4.1-flash (Ollama cloud) | 0.845 | 0.960 | 0.975 | 13F / 7B | p=0.263 |

All three duel runs see byte-identical prompts (`rerank_inputs/locomo_duel_prompts_reason.jsonl`, sha256
`6eac7eebbc508a80…`, 183 lines) and share judge order + gold on all 200 rows. 183/200 questions enter the duel
(the judge named >=2 picks); the other 17 keep the judge order.

**Verdict: the deterministic duel does not pass the dev gate** (32B +3, p=0.63 -> null). No M run. M untouched.

### Cont. 7: the CE-agreement gate (Step 1 attempted) -- precision up, still underpowered

Rule (one, zero free parameters, gold/qtype/qid never read): take the duel's move only when the CE ranks the judge's
#2 above its #1 (session score = max window CE); otherwise keep the judge order. `benchmarks/duel_switch_gate.py`
emits a gated raw that the normal `--replay-raw` path scores.

| run | ungated | **gated** | paired | p | switches | precision |
|---|---|---|---|---|---|---|
| 32B | 0.830 | **0.845** | 8F/2B | 0.109 | 26 -> 12 | 0.588 -> 0.800 |
| cloud flash | 0.845 | **0.850** | 10F/3B | 0.092 | 29 -> 17 | 0.650 -> 0.769 |
| 14B | 0.775 | **0.825** | 9F/7B | 0.803 | 57 -> 27 | 0.375 -> 0.562 |

- Pre-registered dev gate (**net >= +8 and p < 0.05**): **FAIL** (best +7, p=0.092). No promotion, no M run.
- Rescues pass the gate more than breaks in all three models (8/2 p=0.054; 9/7 p=0.066; 10/3 p=0.174) -- same 200
  questions, so this is one signal seen three times, not three replications.
- Negative control: CE alone as the swapper hurts (judge 0.815 -> 0.785, 12F/18B; pro judge 0.890 -> 0.865). The
  information is in the conjunction "duel switches AND CE agrees", not in the CE.
- Text-only rules from the 32B reasoning were also tried and are dead (best cue: 2 rescues / 0 breaks; all 26 switches
  are when-questions, so qtype cannot gate). Recorded in the ledger so nobody re-runs them.
- Power: 8F/2B per 200 rows -> p=0.012 at n=400, p=0.0014 at n=600 if the rates hold.

**Next experiment (pre-registered, not yet run):** ~400-600 *unseen* LoCoMo questions (1986 QA total, dev uses 200 ->
1786 free) with their own CE cache and deterministic 14B judge order, then the same margin-0 gate judged at
net >= +4% of n with p < 0.05. Do not tune the margin or any other threshold on the 200 dev rows first.

## Artifacts

| what | path |
|---|---|
| instrument | `benchmarks/llm_judge_duel.py` |
| offline switch report | `benchmarks/duel_switch_report.py` (new) |
| stage-1 judge (frozen) | `/Volumes/WD-Backup/datasets/longmemeval/results/dev_detjudge_replay/calls.jsonl` (sha `e6fd2accb8ffd742`) |
| 14B raw | `.../results/dev_duel_qwen14b/raw.jsonl` (sha `80ea7040cc09bfb3`) |
| 32B raw | `.../results/dev_duel_qwen32b/raw.jsonl` (sha `d6a4f2a06d574c30`) |
| cloud flash raw | `.../results/dev_duel_evren_flash/flash_raw.jsonl` (sha `c35152f48f00a3a9`) |
| inputs | `rerank_inputs/locomo10.json`, `locomo_poolall_perq.json`, `locomo_ce_cache_w600.json` |
| 32B kernel source | `~/krun-kernels/mnemonics-locomo-duel-qwen32b/run.py` (Kaggle `atakanakbaba/mnemonics-locomo-duel-qwen32b`) |
| ledger entry | `ORIGINAL_SCORE_PLAN.md` "Dev 2026-10-17 (cont. 6)" |

## Reproduce (offline, CPU only — verified from a clean output dir)

```sh
cd ~/Projects/mnemonics
R=/Volumes/WD-Backup/datasets/longmemeval

# rescore any raw run through the same instrument (no LLM calls)
.venv/bin/python benchmarks/llm_judge_duel.py \
  --judge-calls $R/results/dev_detjudge_replay/calls.jsonl \
  --perq $R/rerank_inputs/locomo_poolall_perq.json \
  --locomo $R/rerank_inputs/locomo10.json \
  --ce-cache $R/rerank_inputs/locomo_ce_cache_w600.json \
  --duel-k 2 --duel-chars 8000 --reason-prompt \
  --replay-raw $R/results/dev_duel_qwen32b/raw.jsonl \
  --out /tmp/resume_check          # -> 0.785 / 0.815 / 0.830, 183 duels

# switch statistics for one or more finished runs
.venv/bin/python benchmarks/duel_switch_report.py \
  $R/results/dev_duel_qwen14b $R/results/dev_duel_qwen32b $R/results/dev_duel_evren_flash
```

## Diagnosis: the binding constraint is WHEN to switch, not discrimination

32B, per row (from `duel_switch_report.py`):

| | count |
|---|---|
| judge #1 wrong, judge #2 gold ("rescuable") | 17 |
| …of those, duel switched to the right one | 10 (0.588) |
| judge #1 already correct | 163 |
| …of those, duel still switched | 12, of which 7 became wrong (5 landed on another gold session) |
| switches on rows where neither pick is gold (harmless) | 4 |
| switches in total | 26 (= 10 rescues + 12 on already-right rows + 4 harmless) |
| switch yield (fixes / switches) | 0.385 |
| switch precision (fixes / (fixes+breaks)) | 0.588 |

So it can tell the right tab from the wrong one about 6 times in 10, but it spends 12 of its 26 switches on rows
that were already right. Same near-tie story as cont. 5, now with a mechanism. Model size did not move this:
14B switches 57 times (yield 0.21, net -8), 32B 26 times (yield 0.385), cloud flash 29 times (yield 0.448,
p=0.26 and not deterministic).

## Falsified / null (do not retry as-is)

1. Deterministic 32B final duel as a promotion stage on dev: +3, p=0.63.
2. Post-hoc text cue "switch only if session-0 reasoning says it lacks the information": 32B 0.815 = judge;
   14B/flash 0.815 by suppressing every switch. The cue regex fires on 0 of the 14B/flash raws -> not portable.
3. Bare 2-way choice without reasoning (14B, MAX_NEW 48): -8, negative.
4. The old answer-parser fallback (last integer anywhere in prose) was an instrument bug, not a result: it cost
   the 32B 1 question (0.820 -> 0.830 with strict parse). Do not resurrect it.

## How to continue (ordered; each step is offline and cheap until a gate passes)

**Step 1 — DONE (cont. 7), verdict: underpowered null, do not promote.** The CE-agreement gate is the version worth
scaling (see above): it raises switch precision in all three duel models without a single GPU call, but it reaches
+6/+7 of the required +8 and p=0.09-0.11. Text-only rules from the model's own reasoning are falsified.

**Step 2 — needs more questions, not more GPU.** The resolving experiment is ~400-600 unseen LoCoMo questions.
PRE-REGISTERED 2026-10-18 02:45, before any holdout judge output existed:

- Set: 500 qids sampled with `random.seed(13)` from the 1336 LoCoMo questions that are in the qwen-rerank full
  run (`results/locomo_clean2_qwenrr_full/calls.jsonl`) and NOT in the 200 dev questions. Frozen in
  `rerank_inputs/locomo_holdout_qids.json`; perq in `rerank_inputs/locomo_holdout_perq.json`.
- Base order for holdout = that run's qwen-rerank `order` (8 candidates, base R@1 0.826). DEVIATION from dev, where
  the judge stage saw qrr4b-base prompts (verified byte-identical: sha `1d1badf1…`) and was scored on the CE base
  (0.785). Holdout is self-consistent (prompts and scoring use the same base) but its absolute numbers are NOT
  comparable to dev -- only the paired gate effect inside holdout is.
- Judge: unchanged deterministic kernel (Qwen3-14B @40c0698, NF4, greedy, MAX_NEW 48, thinking off, repeat-check
  10), same prompt template and flags (top-n 10, windows-per-session 3, window-chars 600, no --personal).
- Gate: unchanged margin-0 sign rule from `duel_switch_gate.py`, tested in BOTH forms, because the mechanism is the
  conjunction: (i) `--mode veto` on a 32B duel run on the same 500 questions -- the actual claim; (ii) `--mode swap`,
  the negative control already measured as harmful on dev. Bar: **net >= +4% of n (= +20 of 500) and McNemar
  p < 0.05**, for the veto form. If (ii) holds up as harmful and (i) still gains, the conjunction claim survives a
  new sample; if either flips, the dev gain was noise and the line closes with judge-only 0.815.
- Two GPU kernels, run in order: deterministic 14B judge on the 500 holdout prompts (~dm dev: 200 prompts took
  1664 s on Colab T4), then the Qwen3-32B duel kernel on the holdout duel prompts (dev: 183 prompts, 5672 s on
  2xT4). Both are existing kernel sources pointed at the new prompt dataset; check the weekly GPU quota first.
- Amendment log: this design was written before any holdout output existed. The first draft omitted the 32B duel and
  would only have re-measured the negative control; it was replaced in the same minute, still before any judge
  output, because a swap-only holdout cannot test the conjunction at all.

### Holdout progress (2026-10-18 04:05)

- 500 unseen questions built, CE cache filled (500 qids, `locomo_holdout_ce_w600.json`), judge prompts dumped and
  uploaded as Kaggle dataset `atakanakbaba/mnemonics-locomo-holdout-judge-prompts`.
- Deterministic 14B judge **COMPLETE**: 500 prompts in 1771 s, repeat-check 0/10 mismatches (deterministic).
  Judge order replayed onto the holdout base: **base 0.826 / 0.948 / 0.950 -> judge 0.852 / 0.948 / 0.950** on 500
  questions (`results/holdout_judge`). So the judge stage itself replicates on unseen data (+13 rows).
- Negative control `--mode swap` on holdout (`results/holdout_judge_swap`): 0.822, **net -15, p=0.036** (15F/30B,
  70 switches). The dev finding that CE disagreement alone is harmful replicates on new questions -- this is the
  first result in this line that is significant and on held-out data, and it is a NEGATIVE result for the CE.
- 32B duel on the holdout: 383 duel prompts (the judge named >=2 picks on 383 of 500), kernel
  `atakanakbaba/mnemonics-locomo-holdout-duel-qwen32b` pushed and RUNNING (~3.3 h expected at the dev rate of
  31 s/prompt on 2xT4). The veto-form gate result is the pending measurement.
- Note for the veto test: with judge 0.852 the rescuable ceiling is 28/500 (judge #1 wrong and judge #2 gold), so a
  perfect switch rule tops out at 0.908; the pre-registered bar is +20 net.

The operational steps are (unchanged from the original plan):

1. `python benchmarks/dev_llm_rerank.py --ce-only` (or the existing rerank path) over the unseen qids to fill a new
   CE cache; LoCoMo QA total is 1986, the dev 200 is a subset, so pick the complement explicitly and freeze the
   qid list in a file next to the cache (`locomo_holdout_qids.json`).
2. Run the deterministic 14B judge over those prompts on Kaggle (check the weekly GPU quota first; the 32B duel used
   2xT4 for 5672 s).
3. `duel_switch_gate.py` + `llm_judge_duel.py --replay-raw` as usual, and score the gate at net >= +4% of n, p < 0.05.
   Only if that passes does EXAM/M come back into play -- and the EXAM gate is pre-registered there.

**Step 3 — if the holdout also comes back flat:** accept judge-only (0.815 deterministic) for this line and move the
lever to the judge's near-tie top-1 (cont. 5: 11 demoted-correct CE#1s) instead of another duel round.

**Step 3 — if Step 1 fails again,** the honest fallback is judge-only (0.815 deterministic) and the next lever is
the judge's near-tie top-1 (cont. 5: 11 demoted-correct CE#1s), not another duel round. A listwise judge that sees
dates + 2 windows per candidate is the cheapest thing not yet tried.

## Instrument contract (do not change silently)

- Answer parse: `ANSWER: <n>` wins; otherwise only a bare index (`"0"`, `"[1]"`) is accepted; prose keeps the
  judge order. Any change here changes historical numbers — record both numbers if it ever changes again.
- `model` in summary.json is `replay:<file>` for replay runs (never name a cloud model that was not called).
- duel_k=2, duel_chars=8000, window_chars=600, CE cache `locomo_ce_cache_w600.json`, prompt `--reason-prompt`.

## Limits / open

- n=200, single dev set; +3..+6 differences are ~1 SD and none is significant.
- The cloud flash run is not reproducible bit-for-bit (non-deterministic provider) — it is a reference point only.
- `mnemonics-locomo-duel-qwen14b-reason` kernel never ran (slug unreachable); its source exists locally but there is
  no result to recover.
- Kaggle GPU quota: the 32B run used 2xT4 for 5672 s. Check weekly quota before launching another GPU kernel.
- M data: still untouched by this line of work.

## PRE-REGISTERED 2026-10-18 04:20 — complement-836 replication (written before any 836 output exists)

Why now: the running 500-question veto test is not decisive against a shrunken effect. Power at n=500 is 0.84 if the
true gated rate is 4% fixes / 1% breaks, but only 0.54 at 3%/1% and 0.23 at 2.5%/1.25% -- and the dev estimate
(8F/2B of 200) is itself selection-biased upward because the gate was chosen on those rows. A flat n=500 result would
therefore be uninformative, which is the exact failure mode this line already had at n=200. Power at n=836: 0.98 /
0.79 / 0.38 for the same three rates. (Monte-Carlo McNemar power, 4000 sims, seed 1.)

- Set: every remaining unseen question, not a sample. `complement = (qwenrr_full 1536 qids - dev 200) - holdout 500`
  = **836 qids**, frozen in `rerank_inputs/locomo_comp836_qids.json` (sorted), perq in
  `rerank_inputs/locomo_comp836_perq.json`. There is no seed and no judgement here: the set is the arithmetic
  complement, so no researcher freedom exists in what goes in it.
- Instrument: byte-identical reconstruction of the holdout pipeline. Verified before this pre-registration by
  re-dumping the 500 holdout judge prompts from perq + base order + `locomo10.json` + the existing CE cache:
  sha256 `af6ac22c9e54de13…` matched `locomo_holdout_judge_prompts.jsonl` exactly. Same flags, same 14B judge
  kernel, same 32B duel kernel, same gate at margin 0. New CE cache file
  `rerank_inputs/locomo_comp836_ce_w600.json` -- no scores are copied from dev or holdout, so nothing can leak across
  sets through the cache.
- Bars, both pre-registered:
  1. Per-set (836): veto form net >= +4% of n (= **+34**) and McNemar p < 0.05. Swap form must stay harmful.
  2. Confirmatory pooled (500 + 836 = 1336): veto net >= **+54** and p < 0.05. The pooled number is the one that
     decides whether the duo (duel shift ∧ CE agreement) beats judge-only; the per-set numbers are replication
     diagnostics, not pick-your-favourite evidence.
- No peeking: all 836 are generated, scored and reported in one pass. No stopping early on a partial subset, no
  dropping questions, no re-running with a different duel_k.
- What would falsify: veto failing the 836 bar while the swap control also stops being harmful, or a pooled net below
  +54 with p >= 0.05. In that case the line closes at judge-only and the lever moves to the judge's near-tie top-1.

### Complement-836 progress

- CE cache `rerank_inputs/locomo_comp836_ce_w600.json`: 836/836 qids, no empty score lists.
- Judge prompts `rerank_inputs/locomo_comp836_judge_prompts.jsonl`: 836 rows, 0 overlap with dev and 0 with holdout,
  template suffix identical to the holdout prompts.
- Dataset `atakanakbaba/mnemonics-comp836-judge-prompts` (ready), kernel `atakanakbaba/mnemonics-comp836-judge`
  (deterministic 14B, same source, only the dataset source changed) pushed 05:02 and RUNNING. Kaggle accepted a
  second concurrent GPU session, so the 500-question duel was not disturbed (still RUNNING).
- Remaining: judge raw -> duel prompts -> 32B duel kernel on 836, then per-set and pooled scoring.
