# Encoder swap and BEAM position, measured 2026-07-25

Two questions, both answered with measurements on this machine, no GPU, no paid API.

1. The store still runs `all-MiniLM-L6-v2` (2021, 384d) while 2026 shipped
   Qwen3-Embedding, Jina v5, Gemini Embedding 2, KaLM-Gemma3. Is the encoder
   holding the champion pipeline back?
2. LongMemEval and LoCoMo are saturated. BEAM (ICLR 2026) is not. Where does
   mnemonics actually stand on it?

Scripts: `benchmarks/encoder_probe.py`, `benchmarks/beam_probe.py`.
Raw output: `benchmarks/results/encoder_probe/`, `benchmarks/results/beam/`.

---

## 1. Encoder swap: no headroom, hypothesis closed

The champion pipeline is `encoder top-50 + BM25 top-50 -> RRF -> cross-encoder
-> top_k`. The cross-encoder is fixed across any encoder choice, so the only
thing a different encoder can change is the 50-row band handed to the CE.

Measured on all 500 LongMemEval questions, champion chunking
(`--chunk-mode turn --augment-preferences --candidate-k 50`):

```
arm           dim   vec@1   vec@5   vec@50   bm25@50  fused@1   fused@50     sec
minilm        384   0.882   0.974    0.998     0.996    0.894      1.000    1434
```

`fused@50 = 1.000`. The gold session is in the CE's input band on 500 of 500
questions.

Why this closes the question rather than merely discouraging it: `_ce_rerank`
(`mnemonics/retrieve.py:91`) scores every one of the 50 candidates and sorts by
CE score. It never reads the incoming order. So a different encoder can only
reach the final answer by changing the *membership* of the band, and membership
recall is already 1.000, any change either preserves the gold row or drops it.
There is no path by which better recall improves R@1, because recall is maxed.

The remaining theoretical path is distractor composition: a different encoder
could supply a band whose non-gold rows the CE finds easier to reject. That is
not a property a stronger retriever is designed to control, and betting a
training or migration effort on it is betting on luck.

Where the champion's R@1 actually comes from: pre-CE `fused@1 = 0.894`, final
`R@1 = 0.958` (CHAMPION.json). The 6.4 points are the CE's work. The encoder
cannot touch them.

**Decision: do not run the bge-large / Qwen3 arms, do not fine-tune an encoder,
do not migrate off MiniLM for retrieval quality reasons.** The arms are wired in
`encoder_probe.py` (`--arms bge-base,bge-large,e5-large,qwen3`, each with its
correct query instruction prefix) should a future pipeline change reopen this.

Caveat, stated plainly: this holds for the LongMemEval haystack shape, ~700
turn-chunks per question. It is not a claim about corpora orders of magnitude
larger, see BEAM below, where recall is *not* saturated.

---

## 2. BEAM: retrieval floor, and it degrades with scale

BEAM ships `source_chat_ids` for 9 of its 10 abilities, so retrieval is scorable
without an answering LLM or a judge. `abstention` is excluded: it has no source
turns by construction and a retriever cannot be scored on a question whose
correct answer is "the chat does not say".

Two metrics, because they mean different things:
- `hit@k`, at least one gold turn in the top-k. What single-hop needs.
- `cover@k`, *every* gold turn in the top-k. What multi-hop actually needs.

MiniLM, hybrid retrieval, **no cross-encoder**, no answerer:

| tier | questions | hit@10 | cover@10 | cover@50 |
|------|-----------|--------|----------|----------|
| 100K | 355 | 0.786 | 0.451 | 0.699 |
| 500K | 629 | 0.700 | 0.416 | 0.615 |
| 1M   | 625 | 0.706 | 0.237 | 0.448 |

> **Correction, same day.** These rows chunk at one whole turn-pair, inherited
> from the LongMemEval champion protocol (`--chunk-mode turn`, which passes
> `chunk_size=99999` to suppress re-splitting). **Production does not do this.**
> `ingest()` defaults to `chunk_size=200, chunk_overlap=40` (`mnemonics/ingest.py:249`)
> and neither `cli.py` nor `server.py` ever overrides it, so every live memory is
> already stored in 200-word pieces. Worse, BEAM turn-pairs average ~770 words
> while `all-MiniLM-L6-v2` has `max_seq_length = 256` tokens (measured), so the
> dense side of these rows never saw about three quarters of their text.
> The table above therefore measures a benchmark configuration, not the shipped
> system, and part of the collapse is encoder truncation rather than chunk
> granularity. Treat it as the ceiling of a misconfiguration, not as the live
> system's position. `beam_cover_diversity.py` re-measures at production
> granularity.

`hit@10` barely moves from 100K to 1M (0.786 -> 0.706) while `cover@10` halves
(0.451 -> 0.237). The system keeps finding *a* relevant turn as the haystack
grows; it stops finding *all* of them. That is the multi-hop failure mode, and
aggregate hit-rate hides it completely.

Worst abilities at 1M:

```
summarization             hit@10 0.697   cover@10 0.000
event_ordering            hit@10 0.543   cover@10 0.000
multi_session_reasoning   hit@10 0.771   cover@10 0.086
instruction_following     hit@5  0.186   cover@10 0.200
```

`summarization` and `event_ordering` have cover ~0 at every tier including 100K.
These questions depend on dozens of turns at once. A top-k retriever cannot
satisfy them by ranking better, 10 slots cannot hold 30 gold turns. This is a
structural gap, not a tuning gap, and it is where the 2026 literature has moved
(hierarchical summaries, event segmentation, temporal graphs) rather than to
better encoders.

Strongest abilities at 1M: `knowledge_update` (hit@10 0.957),
`contradiction_resolution` (0.957), `temporal_reasoning` (0.900). The
temporal work already in the champion config is holding up at scale.

**These numbers are a floor, not a BEAM score.** No CE, no answerer. On
LongMemEval the same CE moves 0.894 -> 0.958. The CE-reranked measurement is
`benchmarks/kaggle_beam_ce.py` (GPU job).

---

## 3. The ceiling: the evidence fits, retrieval delivers half of it

Every cover number above compares retrieval against retrieval. None of them
answers the prior question: can the evidence a question depends on even fit in
the budget? `oracle_cover.py` removes retrieval entirely and takes the cheapest
chunk set covering every gold message id (greedy set cover by word cost).

Budget 4000 words, all three tiers, 1609 questions:

| tier | chunk | oracle | measured | headroom | gold words needed | gold chunks |
|------|-------|--------|----------|----------|-------------------|-------------|
| 100K | s100 | 1.000 | 0.499 | **+0.501** | 263 | 2.75 |
| 100K | s200 | 1.000 | 0.462 | **+0.538** | 524 | 2.84 |
| 100K | w1   | 0.938 | 0.361 | +0.577 | 1663 | 2.71 |
| 500K | s100 | 1.000 | 0.475 | **+0.525** | 373 | 3.90 |
| 500K | s200 | 0.981 | 0.418 | **+0.563** | 744 | 4.02 |
| 500K | w1   | 0.847 | 0.334 | +0.513 | 2294 | 3.85 |
| 1M   | s100 | 0.982 | pending | - | 706 | 7.68 |
| 1M   | s200 | 0.893 | pending | - | 1386 | 7.92 |
| 1M   | w1   | 0.754 | pending | - | 4517 | 7.54 |

Two things fall out.

**The budget is not the constraint.** A question's evidence averages 263 to 706
words at production granularity against a 4000-word budget, roughly a sixth of
what is available, and the oracle sits at 0.98 to 1.00. Nothing physical stops a
retriever from covering these questions.

**Retrieval delivers about half of what it could.** Headroom is +0.50 across
every tier and both production-shaped granularities. Not the +0.04 that chunk
tuning offers, and not the zero that the encoder offers.

The `w1` rows also explain the original scare: at 1M its gold sets average 4517
words, past the 4000-word budget, so a quarter of those questions are
*physically* uncoverable at that granularity. The collapse attributed to scale
was substantially a benchmark configuration that production does not use.

Where the remaining half sits is now the only open question, and it is not the
encoder (section 1), not chunk size (section 4), and not cheap re-ranking:
`beam_cover_diversity.py` ran MMR, pseudo-relevance feedback and adjacency
expansion across five granularities, and the plain hybrid baseline won all
twenty cells.

## 4. Chunk size is already right in the live store

`ingest()` defaults to 200 words with 40 overlap and nothing overrides it, so
the question is what the live data actually looks like:

```
~/.mnemonics/memories.db   10,843 rows
mean 117 words, median 107, p90 200, max 343
rows over 200 words: 0.0%
rows over 100 words: 55.0%
```

Records are already at the granularity BEAM's sweep prefers (the s100/s200 peak
region). The measured s200 -> s100 gain of +0.037 was found on 770-word BEAM
turn-pairs; on 107-word records there is close to nothing left to split. Against
that, changing chunk size forces a re-chunk and re-embed of every store and
breaks the row ids that supersede/dedup history is keyed on.

**Decision: leave chunking alone.** Not "probably fine", measured.

## Reproduce

```bash
# encoder A/B, ~24 min for 500q on CPU
python benchmarks/encoder_probe.py \
    --data longmemeval_s_cleaned.json -n 500 --arms minilm

# BEAM retrieval floor, ~16 min for all three tiers on CPU
python benchmarks/beam_probe.py --sizes 100K,500K,1M

# BEAM + cross-encoder, GPU
krun benchmarks/kaggle_beam_ce.py --acc NvidiaTeslaT4
```

`kaggle_beam_ce.py` embeds the `mnemonics/` tree and `beam_probe.py` as base64
rather than cloning them. Local HEAD carries an unpushed commit touching
`ingest`, and a Kaggle clone of the remote would measure the CE delta against
different code than the floor above.
