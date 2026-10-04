# SOTA Proof, LongMemEval-S, LLM-free retrieval

**Packaged:** 2026-06-22
**Forensically corrected:** 2026-10-04
**Champion source commit:** `ca53594` (`ca5359417c9a4692d86e58bc36253e00b54172f8`)
**Canonical record:** `benchmarks/CHAMPION.json`

> **Correction, 2026-10-04: use this section instead of the older narrative.**
>
> The original proof package described the 0.958 configuration incorrectly. The
> archived Kaggle kernels are still available and settle the question directly.
> `krun-det1` and `krun-det2` each checked out `ca53594` and independently
> produced **R@1=0.958, R@5=1.0, R@10=1.0**. `krun-confirm` then produced
> **R@1=0.958 for seeds 42, 123 and 7**.
>
> The real champion **does include the trust gate**. In both deterministic
> kernels `chat-ce-v3-20260516` fired on **86/500** questions at margin `0.2`,
> with pin margin `0.5`. The base reranker is the checkpoint
> `mn-ce-v1-20260604`, not `BAAI/bge-reranker-v2-m3`. The run also uses
> `--temporal-v2` and does **not** use `--augment-preferences`.
>
> The earlier correction claiming the gate was absent was based on a later
> `kaggle_champion.py`/`CHAMPION.json` packaging drift rather than the actual
> June run. That claim is superseded by the recovered kernel source, logs and
> output JSONs.

## The measured claim

On LongMemEval-S (500 questions), the historical benchmark harness retrieves the
correct evidence at **R@1=0.958, R@5=1.0, R@10=1.0**. The retrieval path is
LLM-free: MiniLM embeddings, hybrid HNSW+BM25 retrieval, `mn-ce-v1` cross-encoder
reranking, `chat-ce-v3` trust-gated top-1 correction, temporal-v2
post-processing, and a high-confidence gate pin. HNSW construction is made
reproducible with `MNEMONICS_DETERMINISTIC=1`.

This is deliberately a **benchmark-harness claim**, not yet a production-library
claim. The historical ordinal temporal branch is routed using the dataset's gold
`question_type == "temporal-reasoning"` and uses haystack session dates. Those
signals are not both available through the normal production retrieval API.

## Authoritative evidence

| Fact | Status | Evidence |
|---|---|---|
| R@1=0.958, R@5=R@10=1.0 | verified | archived `krun-det1` and `krun-det2`, two independent Kaggle T4 kernels |
| Seed stability | verified | archived `krun-confirm`: seeds 42, 123, 7 all R@1=0.958 |
| Trust gate is part of champion | verified | det1/det2 logs + result JSON: `chat-ce-v3-20260516`, margin 0.2, fired 86/500 |
| Gate pin | verified | archived kernel source: `--trust-gate-pin-margin 0.5` |
| Base CE | verified | archived kernel source: `MNEMONICS_RERANK_MODEL=mn-ce-v1-20260604` |
| Temporal policy | verified | archived kernel source: `--temporal-aware --temporal-v2` |
| No preference augmentation | verified | archived kernel source: no `--augment-preferences` flag |
| Source code | verified | det1/det2 log: `ca53594 fix(store): env-gated deterministic HNSW for reproducible benchmarking` |
| by-type breakdown | verified | archived `krun-confirm/results/champ_seed42.json` |

Recovered seed-42 by-type R@1:

| Type | n | R@1 |
|---|---:|---:|
| single-session-user | 70 | 0.9714 |
| multi-session | 133 | 0.9850 |
| single-session-preference | 30 | 0.8000 |
| temporal-reasoning | 133 | 0.9398 |
| knowledge-update | 78 | 0.9615 |
| single-session-assistant | 56 | 1.0000 |

## Exact historical configuration

```text
source:   ca5359417c9a4692d86e58bc36253e00b54172f8
config:   --mode rerank --chunk-mode turn
          --temporal-aware --temporal-v2
          --candidate-k 50 --seed 42
          --trust-gate-ce chat-ce-v3-20260516
          --trust-gate-margin 0.2
          --trust-gate-pin-margin 0.5
env:      MNEMONICS_DETERMINISTIC=1
reranker: mn-ce-v1-20260604
encoder:  all-MiniLM-L6-v2 (384d)
data:     longmemeval_s_cleaned.json (500q)
device:   Kaggle NvidiaTeslaT4
```

`--augment-preferences` is **not** part of this configuration.

## Determinism evidence

`ca53594` fixed the HNSW source of cross-machine drift: hnswlib insertion with
multiple threads can build a different graph, while `MNEMONICS_DETERMINISTIC=1`
forces single-threaded index construction. After that fix, det1 and det2 both
produced the same 0.958 score, and `krun-confirm` reproduced it across three
seeds.

The old `0.976` observation should not be treated as the canonical champion. It
came from a different/non-canonical experimental state. The archived
fully-specified deterministic evidence supports **0.958**.

## Temporal ablation requirement

Because the champion has a trust gate before temporal processing and a
confidence pin after it, a raw post-CE/pre-gate candidate dump cannot reproduce
0.958. The correct one-dump ablation contract is:

1. run the exact historical champion;
2. snapshot rows **after trust gate, before temporal**;
3. record `gate_top_id`, `ftce_margin`, pin margin and temporal-v2/v3 settings;
4. replay three variants from that same snapshot:
   - `off`: no temporal stage;
   - `on_labeled`: historical temporal-v2 + gold-label ordinal routing + pin;
   - `on_no_label`: temporal-v2 with ordinal gold-label routing disabled + pin;
5. accept the ablation only if `on_labeled` reproduces **R@1=0.958**.

`benchmarks/temporal_ablation_offline.py` enforces this contract and rejects the
older pre-gate `--dump-candidates` format.

### Measured temporal ablation, 2026-10-04

The exact `ca53594` historical replay completed at **R@1=0.958, R@5=1.0,
R@10=1.0**, with the trust gate firing **86/500** and a complete 500-row
post-gate/pre-temporal dump. Replaying all temporal variants from that one shared
dump produced:

| variant | R@1 | R@5 | R@10 | paired delta vs off | helped / hurt |
|---|---:|---:|---:|---:|---:|
| temporal off | 0.956 | 0.998 | 1.000 | — | — |
| historical labeled routing | 0.958 | 1.000 | 1.000 | +0.002, 95% CI [-0.004, 0.008] | 2 / 1 |
| **label-free temporal-v2** | **0.960** | **1.000** | **1.000** | **+0.004, 95% CI [0.000, 0.010]** | **2 / 0** |

The historical gold-label ordinal branch is therefore **not responsible for the
measured temporal gain**. On the same candidates it hurts one question relative
to the label-free variant and helps none: labeled minus no-label is -0.002 R@1
(95% CI [-0.006, 0.000], helped 0, hurt 1). The productionizable conclusion is
to keep the label-free relative-time temporal-v2 logic and disable gold-label
ordinal routing. On this exact replay that policy scores **R@1=0.960**, two
points above the historical 0.958 benchmark-harness configuration.

## Reproduce

`benchmarks/kaggle_temporal_champion_exact.py` checks out `ca53594`, uses the
recovered historical checkpoints/configuration, and adds dump-only
instrumentation for the temporal ablation. It fails the run if the aggregate
score or 86/500 gate count does not reproduce.

The old `benchmarks/kaggle_champion.py` drifted after the June measurement and
must not be used as provenance for the historical 0.958 claim unless it is first
brought back into alignment with the exact configuration above.
