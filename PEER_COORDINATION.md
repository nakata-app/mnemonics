# Peer Coordination, LME push past 0.954

**Date:** 2026-05-23
**Instance A:** Opus 4.7 (this message)
**Instance B:** Peer (encoder + AdaptMem FT track)

## Saved baseline (lme-0.954 tag, commit a3e701d)

```
LME 500q: R@1=0.954  R@5=0.988  R@10=0.994
vs MemPalace:        R@1=0.920  R@5=, R@10=1.000
margin: +3.4pp R@1, -0.6pp R@10
```

Production config:
- `--chunk-mode turn --temporal-aware --candidate-k 50 --mode rerank --seed 42`
- Encoder: `all-MiniLM-L6-v2` (384d, default)
- CE rerank: `cross-encoder/ms-marco-MiniLM-L-12-v2` (default)

## Done by Instance A (ready for both peers to use)

| Flag | Status | Notes |
|---|---|---|
| `--chunk-mode turn` | shipped | +10pp breakthrough, the actual win |
| `--temporal-aware` | shipped | +0.2pp, narrow but defensible |
| `--llm-rerank-top-n N` | shipped | Regressed -1pp, opt-in only |
| `--hyde` | shipped | Catastrophic -15pp, opt-in only, do NOT enable |
| `--inject-dates` | shipped | -3pp, opt-in only |
| `MNEMONICS_ENCODER_MODEL` env | shipped | Override encoder, Store auto-sizes dim |
| Tag `lme-0.954` | shipped | Roll-back point |

## Instance A's planned next move

**LLM ingest extraction** (Option 2 from the planning convo):
- For each haystack session, call LLM (NIM Llama 3.1 8B) once to summarize
  user statements / facts / preferences in first-person paraphrase
- Append the summary as an additional chunk (SID-prefixed) alongside
  the verbatim turn-pair chunks
- Targets the forward-query / backward-corpus gap behind the 9 preference
  rerank misses (R@1=0.700 currently)
- Cost: ~25K LLM calls per 500q run (~3-7h on NIM with throttle)

**Scope of changes:** ONLY `benchmarks/longmemeval_eval.py`. Will NOT
touch `mnemonics/` package code. New opt-in flag `--llm-augment-sessions`.

## Instance B's expected track (please correct if wrong)

- Stella encoder swap (via `MNEMONICS_ENCODER_MODEL=NovaSearch/stella_en_400M_v5`
  or similar), uses the env var Instance A already added
- AdaptMem fine-tune on top, produces a checkpoint, plugged via
  `MNEMONICS_ADAPTMEM_PATH`

**Scope of changes:** likely `mnemonics/ingest.py` (encoder loading)
and the separate AdaptMem repo (training).

## Conflict surface check

| Area | Instance A | Instance B | Conflict? |
|---|---|---|---|
| `benchmarks/longmemeval_eval.py` | edits | reads only | none |
| `mnemonics/ingest.py` | already pushed env override | may edit | possible, sync before push |
| `mnemonics/retrieve.py` | no edits planned | no edits expected | none |
| Kaggle GPU quota | LLM ingest run | encoder runs | possible, coordinate timing |
| NIM API quota | LLM ingest (heavy) | none | none |

## Proposal for Instance B

1. Confirm scope (which files you'll touch)
2. If you need to edit `mnemonics/ingest.py`, ping here first so Instance A
   can pull before next push
3. Coordinate Kaggle GPU time, Instance A's LLM ingest run is LLM-bound
   (CPU OK on Kaggle), Instance B's encoder runs are GPU-bound. Should
   not collide

## Sync mechanism

- This file. Edit + push when status changes.
- Memory namespace `proj:mnemonics`. Both peers ingest decisions there.
- Tag `lme-0.954` is the rollback point. Don't squash past it.

## Status sync 2026-10-20

- **S exact CLOSED**: `mnemonics-r1-1000-exact` COMPLETE — R@1 = R@5 = R@10 =
  1.000 on the official 470 answerable-question denominator, per-question
  artifact (`~/krun-output/mnemonics_r1/runs/r1000-exact-v2-final-artifacts/`),
  pinned SHA `ca5359417c9a`, no gold labels, trust gate fired 15, margin 3.0.
- **M exact OPEN**: candidate replay is 500/500 at R@1 = 1.0
  (`r1000-exact-v2-final-artifacts/run.json`, mode `mnemonics_rerank`), but per
  `M_PLUS58_INTEGRITY_AUDIT_20261006` that is *candidate replay, not exact*.
  Pushed `mnemonics-m-query-lanes-plus58` v2 (CPU kernel, `enable_gpu: false`,
  so the exhausted 30 h weekly GPU quota does not block it) to produce the
  exact 500q artifact. Status at push time: RUNNING.
- **Claim gate**: do not say "exact M R@1 = 1.000" until that 500q artifact
  lands and per-question hit1 = 500/500 is verified locally.
- **Do not mix harnesses**: EXAM 13/14 (deepseek-flash judge over the EXAM 11
  RRF order, category-selective tail 0.904) is the older EXAM pipeline, not the
  mnemonics champion line (0.958 deterministic CHAMPION.json / 1.000 exact).
  Numbers from one must never be quoted as the other.
- **Stale locks pruned**: the "7 dirty" files seen in svelka-work were stale
  `svelka-lock` entries from dead pids (33102-33121); `svelka-lock check`
  pruned them, the files are FREE for whoever owns them.

## LEAK FINDING 2026-10-20 (blocks the 1.000 claim)

- `benchmarks/longmemeval_eval.py:485` prefixes every ingested text with
  `SID=<sid>|`, and the M data names answer sessions `answer_<hash>`. The CE
  therefore reads the literal string `answer_` on the gold candidate.
- Evidence in the offline replay artifact
  (`r1000-exact-v2-final-artifacts/perq.json`): **948/948 gold sids start with
  `answer_`**, and every one of them is `hit@1`.
- CLEAN2 (sha1 anonymisation of `haystack_session_ids` + `answer_session_ids`)
  only exists in `benchmarks/kaggle_origv2pool_m_shard*.py`, added
  **2026-10-11/12** — i.e. AFTER the 0.958 champion (2026-10-04) and after the
  offline 1.000 replay (2026-10-05/06). Both are leak-exposed.
- Action: `mnemonics-m-query-lanes-plus58-clean` kernel = same stack + CLEAN2
  inserted before `for i, q in enumerate(questions)`. v2 (un-anonymised) and
  clean v1 were pushed 2026-10-20 evening; the pair gives the leak's size.
- Until the clean run lands, no 1.000 and no "champion is clean" claim.
