# LongMemEval benchmark landscape and claim ledger

Updated: 2026-10-06

This file separates benchmark generations, evaluation protocols, denominators,
and internal evidence levels. Do not compare rows as if they were identical
unless `metric`, `n_scored`, and protocol match.

## Mnemonics verified state

| Benchmark | Metric | n | Result | Evidence | Claim status |
|---|---:|---:|---:|---|---|
| LongMemEval-S | session retrieval R@1 / R@5 / R@10 | 500 | 1.000 / 1.000 / 1.000 | `r1000-exact-v2-final-artifacts` | verified exact |
| LongMemEval-M | session retrieval R@1 / R@5 / R@10 | 500 | .884 / .952 / .964 | `m-query-lanes-small` exact artifacts | verified exact |
| LongMemEval-M | candidate/offline R@1 | 500 | 1.000 (500/500) | plus58 replay chain | offline candidate only; not exact |
| LongMemEval-V2 Small | official QA accuracy | 451 | pending | frozen core `8aa584d`; adapter `21db999` | no score yet |

### M final validation rule

`atakanakbaba/mnemonics-m-query-lanes-plus58-cpu` is the final end-to-end exact
validation. Until its artifact completes and reports 500/500, the public exact
M claim remains R@1=.884. Offline 500/500 must never be described as exact.

## V1 / LongMemEval-S public retrieval reports

These are useful context, not a single official leaderboard. Protocols differ.

| System | R@1 | R@5 | R@10 | n scored | Retrieval protocol / caveat |
|---|---:|---:|---:|---:|---|
| Mnemonics | **1.000** | **1.000** | **1.000** | 500 | exact internal artifact; includes full 500 evaluation rows |
| cogito-ergo runP-v35 | .964 | — | — | 470 | selective qwen-turbo LLM filter; 30 abstention rows excluded; 453/470 |
| QMG v1.2 | .906 | .986 | .994 | 500 reported | cleaned LongMemEval-S retrieval report; gte-large/session pooling |
| TopoDB hybrid | .894 | .987 | .996 | 470 graded | 500-question run, 30 abstention rows excluded; MiniLM + BM25 RRF |

Important: the original LongMemEval retrieval convention commonly excludes the
30 information-unavailable (`*_abs`) rows because they have no gold evidence
session. A 470-row retrieval result and a 500-row internal result are not
strictly apples-to-apples until the exact scoring convention is reconciled.

Public source references checked 2026-10-06:
- Hermes Labs `fidelis/WRITEUP-LONGMEMEVAL-20260423.md` (cogito-ergo runP-v35)
- TopoDB `benchmarks/longmemeval/RESULTS.md`
- xiaowu0162/LongMemEval GitHub issues / QMG v1.2 report
- xiaowu0162/LongMemEval README for official retrieval convention

## LongMemEval-V2 fixed protocol

Official repository facts used for the first frozen Mnemonics run:

- 451 manually curated questions, web + enterprise.
- Small and Medium public tiers; Small uses 100 trajectories per domain-shared haystack.
- Memory backend contract: `insert(trajectory)` and `query(query, query_image=None)`.
- The evaluator keeps question id, question type, gold answer, and evaluator config
  private from the backend query call.
- Paper / leaderboard reader: `Qwen/Qwen3.5-9B`.
- LLM judge: `gpt-5.2`, medium reasoning.
- Leaderboard package validation requires reader model metadata containing
  `qwen3.5-9b` and judge metadata containing `gpt-5.2`.

Frozen Mnemonics provenance:

- Core before first V2 score: tag `v2-untouched-core-20261006`
  -> `8aa584d523f587d9a918f9ec9321ffe90532c50a`.
- Schema-only adapter: `21db999`.
- Adapter leakage guard test deliberately injects fake `question_id`, `answer`,
  `answer_session_ids`, and `eval_function` fields and verifies they are absent
  from indexed text.
- First fixed policy: MiniLM embeddings, vector+BM25 hybrid, candidate_k=64,
  CE rerank, top_k=16, max 4 returned chunks per trajectory.
- First pass is text-only. Screenshot/image support is a separate later operating
  point and must not overwrite the frozen text-only first result.

Current context-export kernel:
`atakanakbaba/mnemonics-v2-small-contexts`

It may produce latency/context artifacts, but **it does not produce official V2
accuracy**. Do not inspect or tune against exported contexts before the first
fixed-reader score if preserving the untouched-evaluation claim.

## Claim discipline

Safe now:
- "Mnemonics has a verified exact 1.000 session-retrieval R@1 on its full
  500-row LongMemEval-S artifact."
- "Mnemonics' best verified exact LongMemEval-M R@1 is .884; a frozen plus58
  policy covers 500/500 in offline candidate replay and is under exact validation."
- "The V2 retrieval core was frozen before the first V2 score and the adapter is
  schema-only; no V2 accuracy has been measured yet."

Not safe yet:
- "LongMemEval-M exact 1.000" before plus58 completion.
- "World #1 / SOTA on LongMemEval-S" before the 470-vs-500 protocol audit.
- "V2 beats AgentRunbook-C" before an official Qwen3.5-9B reader + GPT-5.2
  judge run is complete.
- Any V2 leaderboard-equivalent claim from the context-export kernel alone.
