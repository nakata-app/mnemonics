# Mnemonics Final-Duel Switch — Active Checkpoint

Updated: 2026-10-18 (repo HEAD `4bd8c81`)

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

**Step 1 — build a switch gate that is actually calibrated (no GPU).**
The 32B outputs already contain the reasoning. Fit/derive a decision rule from the *question+session text only*
(no gold, no qtype, no qid) that predicts "judge #1 is insufficient" better than the model's own choice: e.g.
require the duel's chosen session to be supported by a quoted span, require the unchosen session to be explicitly
negated, or abstain (keep judge order) unless the margin between the two sessions is large. Score it with
`llm_judge_duel.py --replay-raw` (store the rule's verdicts as a raw.jsonl; the rule is then just another raw run,
so the instrument stays untouched).
Pre-registered gate for any promotion: **net >= +8 of 200 and McNemar p < 0.05 on dev**, plus the rule must be
derivable from text alone (audit it for qid/gold/qtype literals before scoring).

**Step 2 — only if Step 1 passes dev:** repeat with the deterministic judge order and a second seed/temperature-0
rerun to prove determinism (the 32B kernel already showed repeat-check 0/10). Then, and only then, run the
pre-registered EXAM gate on M (see the "EXAM" sections). Do not touch M before the dev gate is met — the ledger has
already burned three null promotions.

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
