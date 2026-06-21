# SOTA Proof, LongMemEval-S, LLM-free retrieval

**Packaged:** 2026-06-22
**Champion commit:** `ca53594` (env-gated deterministic HNSW)
**Canonical record:** `benchmarks/CHAMPION.json` (single source of truth, written by `benchmarks/champion.py`)

## The claim

On LongMemEval-S (500 questions), Mnemonics retrieves the correct evidence at
**R@1 = 0.958, R@5 = 1.0, R@10 = 1.0**, with **no LLM in the retrieval path**, 
hybrid HNSW + BM25 + RRF, cross-encoder rerank, trust-gated top-1 override,
temporal-aware ordering. The reader on top can be any LLM; the retrieval that
feeds it is fully model-free and deterministic.

## What is verified, and how

| Fact | Status | Evidence |
|---|---|---|
| R@1 = 0.958 deterministic | measured | commit `ca53594` message: reproduced on 3 runs / 2 Kaggle kernels with `MNEMONICS_DETERMINISTIC=1` |
| R@5 = R@10 = 1.0 | measured | same source; stayed 1.0 across all runs (graph nondeterminism only reshuffled within top-5) |
| Determinism root cause + fix | verified | `ca53594`: hnswlib `add_items` defaults to `num_threads=-1`; multi-threaded insertion builds a thread-count-dependent graph. Local proof (4000 vecs): `num_threads=2` vs `1` differs on 6/100 top-1; `1` vs `1` is identical (0/100). Fix: `MNEMONICS_DETERMINISTIC=1` forces `set_num_threads(1)` at every index creation. |
| Champion config | verified | `PEER_COORDINATION.md` + `benchmarks/kaggle_champion.py` |
| vs MemPalace R@1 = 0.920 | verified (in-repo) | `PEER_COORDINATION.md` baseline table |

### Not verified here (stated honestly)

- **by_type breakdown of the 0.958 run is not on disk.** The champion run printed
  to stdout and exited; `eval/results/champion_run.log` was truncated before the
  score line. Re-run on a GPU to regenerate it (command below).
- **Broader "SOTA vs published systems" claim is not reproduced in this repo.**
  A figure of ~0.76 for a competing system appears in session notes but has no
  in-repo source. The only baseline verified in-repo is MemPalace (0.920).
  Treat the cross-system SOTA claim as unverified until a cited comparison is added.

## The honest number: 0.958, not 0.972

Earlier tags `lme-0.964` … `lme-0.972` recorded higher R@1 on Kaggle T4. Those
were **non-deterministic**, the multi-threaded HNSW graph varied with core
count and thread scheduling, so the trust-gate fired a different number of times
(86 vs 9) and borderline rank-1 ties flipped. Same config, same data, R@1 ranged
0.958, 0.976 run to run. After the determinism fix, the **reproducible** number is
**0.958**. The older tags are kept for history but are not the champion.

## Champion configuration

```
config:  --mode rerank --chunk-mode turn --temporal-aware \
         --augment-preferences --candidate-k 50 --seed 42
encoder: all-MiniLM-L6-v2 (384d)
CE:      BAAI/bge-reranker-v2-m3   (env MNEMONICS_RERANK_MODEL)
flags:   MNEMONICS_DETERMINISTIC=1 (single-threaded HNSW, reproducible)
data:    longmemeval_s_cleaned.json (500q)
device:  Kaggle T4 GPU
```

## Reproduce

GPU (faithful, regenerates by_type):

```bash
krun benchmarks/kaggle_champion.py --dataset atakanakbaba/mnemonics-lme --acc NvidiaTeslaT4
```

Local determinism check (CPU, fast, proves the graph is reproducible, not the
full champion score, which needs the GPU CE):

```bash
MNEMONICS_DETERMINISTIC=1 python benchmarks/longmemeval_eval.py --n 100 \
  --mode rerank --chunk-mode turn --temporal-aware --augment-preferences \
  --candidate-k 50 --seed 42 --out /tmp/lme100.json
```

Note: the default CE (`cross-encoder/ms-marco-MiniLM-L-12-v2`) yields a lower
score than the champion. The 0.958 figure requires `BAAI/bge-reranker-v2-m3`,
which is impractical to run for 500q on CPU, use a GPU for the headline number.
