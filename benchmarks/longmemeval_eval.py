"""Mnemonics-vs-MemPalace eval on LongMemEval-500.

Compares three retrievers on the same dataset (Sebastian Smolny's
longmemeval_s_cleaned, 500 questions × ~50 sessions/question):

  - MemPalace baseline    : IDF-weighted entity-graph overlap (R@1=0.354
                            already measured in
                            adaptmem/benchmarks/structural_memory_eval/
                            entity_graph_result.json — reused here)
  - Mnemonics no-rerank   : retrieve(rerank=False) — hybrid HNSW + BM25
                            RRF + tier decay
  - Mnemonics rerank=True : adds AdaptMem cross-encoder rerank over the
                            candidate band

For each question, we:
  1. ingest every haystack session into a fresh namespace
  2. retrieve top_k for the query
  3. for each k in {1, 5, 10}, count hit if any retrieved row's
     source session is in answer_session_ids

Usage:
  python benchmarks/longmemeval_eval.py            # 50-q subset, both modes
  python benchmarks/longmemeval_eval.py --n 500    # full set
  python benchmarks/longmemeval_eval.py --mode no_rerank
  python benchmarks/longmemeval_eval.py --mode rerank
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# "N (weeks|days|months|years) ago" with optional number word.
_REL_TIME_RE = re.compile(
    r"\b(?:(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|few)\s+)?"
    r"(day|week|month|year)s?\s+ago\b",
    re.IGNORECASE,
)
_WORD_TO_NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "few": 3,
}
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}


def _parse_lme_date(s: str | None) -> datetime | None:
    """Parse LongMemEval date string '2023/05/20 (Sat) 02:21' -> datetime."""
    if not s:
        return None
    head = s.strip().split(" ")[0]
    try:
        return datetime.strptime(head, "%Y/%m/%d")
    except ValueError:
        return None


def _detect_relative_target(query: str, question_date: str | None) -> tuple[datetime, int] | None:
    """If query mentions 'N weeks/days ago' and question_date is parseable,
    return (target_date, tolerance_days). Else None.
    """
    if not question_date:
        return None
    qdate = _parse_lme_date(question_date)
    if qdate is None:
        return None
    m = _REL_TIME_RE.search(query)
    if not m:
        return None
    num_str = (m.group(1) or "1").lower()
    unit = m.group(2).lower()
    num = _WORD_TO_NUM.get(num_str, int(num_str) if num_str.isdigit() else 1)
    delta_days = num * _UNIT_DAYS[unit]
    target = qdate - timedelta(days=delta_days)
    # Tolerance scales with delta. Tight for "a day ago", loose for "a year ago".
    tol = max(2, min(delta_days // 4, 14))
    return target, tol


# Ordinal/comparative temporal cues. "first/earliest/order" wants oldest-first
# (asc); "last/latest/most recent" wants newest-first (desc). Relative "N ago"
# is handled by _detect_relative_target and takes precedence over these.
_ORD_ASC_RE = re.compile(
    r"\b(first|earliest|oldest|chronolog|from earliest|in order)\b", re.IGNORECASE)
_ORD_DESC_RE = re.compile(
    r"\b(last|latest|most recent|newest)\b", re.IGNORECASE)


def _detect_ordinal(query: str) -> str | None:
    """'asc' (oldest first) | 'desc' (newest first) | None for a chronological
    extreme/order question. Used only when no relative-time target is found."""
    if _ORD_ASC_RE.search(query):
        return "asc"
    if _ORD_DESC_RE.search(query):
        return "desc"
    return None


# Cached LLM client for --llm-rerank and --hyde. Built lazily on first use.
_LLM_CLIENT = None
_LLM_INT_RE = re.compile(r"\d+")

# HyDE prompt: ask the LLM to write a user-style passage that *would* answer the
# question. The hypothetical sits in the same vocabulary/style as haystack
# chunks (first-person, specific details, no disclaimers), so its embedding is
# closer to the real answer chunk than the bare question's embedding.
_HYDE_PROMPT = (
    "You are simulating a short passage from a user's chat history that would "
    "answer this question. Write 1-2 sentences in the user's first-person voice "
    "with specific concrete details. Do not add disclaimers, hedging, or 'as "
    "an AI'. Just the passage.\n\nQuestion: {question}\n\nPassage:"
)


def _get_llm_client():
    """OpenAI-compatible client. Defaults to NVIDIA NIM; if DEEPSEEK_API_KEY is
    set it routes to DeepSeek instead (base_url auto-selected, still overridable
    via MNEMONICS_LLM_BASE_URL)."""
    global _LLM_CLIENT
    if _LLM_CLIENT is not None:
        return _LLM_CLIENT
    deepseek = os.environ.get("DEEPSEEK_API_KEY")
    api_key = deepseek or os.environ.get("NVIDIA_API_KEY") or os.environ.get("NIM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No LLM key set; need DEEPSEEK_API_KEY or NVIDIA_API_KEY for --llm-rerank/--hyde")
    default_base = "https://api.deepseek.com" if deepseek else "https://integrate.api.nvidia.com/v1"
    from openai import OpenAI  # local import; only required when an LLM feature is on
    _LLM_CLIENT = OpenAI(
        base_url=os.environ.get("MNEMONICS_LLM_BASE_URL", default_base),
        api_key=api_key,
    )
    return _LLM_CLIENT


def _llm_chat_with_backoff(prompt: str, model: str, max_tokens: int = 120,
                           temperature: float = 0.3) -> str | None:
    """One-shot chat completion with exponential backoff on 429. Returns the
    raw response string or None on terminal failure. Shared by --hyde and
    --llm-rerank so both paths follow the same retry discipline.
    """
    client = _get_llm_client()
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            err_str = str(e)
            is_rate = "429" in err_str or "rate" in err_str.lower() or "too many" in err_str.lower()
            if is_rate and attempt < 4:
                import time as _time
                _time.sleep(2 ** (attempt + 1))  # 2, 4, 8, 16
                continue
            print(f"  LLM error ({attempt=}): {type(e).__name__}: {e}", file=sys.stderr)
            return None
    return None


def _hyde_passage(question: str, model: str | None = None) -> str | None:
    """Generate a HyDE passage for the question. Returns None on LLM failure.

    HyDE (Hypothetical Document Embeddings, Gao et al. 2022): instead of
    embedding the literal question, embed a synthetic answer that mimics the
    style/vocabulary of the corpus. For LongMemEval the corpus is a user's
    chat history — first-person, concrete, no hedging — so the prompt steers
    the LLM toward that voice. The hypothetical may be factually wrong; that
    is fine, because we only use its embedding to *find* the real chunk.
    """
    model = model or os.environ.get("MNEMONICS_HYDE_MODEL", "meta/llama-3.3-70b-instruct")
    raw = _llm_chat_with_backoff(
        _HYDE_PROMPT.format(question=question),
        model=model,
        max_tokens=120,
        temperature=0.3,  # mild diversity so we don't always emit identical phrasing
    )
    if not raw:
        return None
    # Strip common LLM preambles ("Sure! Here's...", quotes, etc.)
    raw = raw.strip().strip('"').strip("'")
    # Take only the first paragraph — a long answer hurts the embedding.
    raw = raw.split("\n\n")[0].strip()
    return raw if len(raw) >= 10 else None


def _llm_rerank_topk(query: str, results: list, top_n: int = 5, model: str | None = None) -> list:
    """LLM-as-judge rerank over the top_n session-unique candidates.

    Sends the question and up to top_n deduped session passages to an LLM,
    asks for the 0-based index of the passage that best contains the answer,
    and moves the chosen one to position 0. The remaining candidates (both
    the unchosen head and any tail beyond top_n) keep their original relative
    order. Returns the original list unchanged on any error (no key, parse
    failure, API exception) so retrieval still degrades gracefully.
    """
    if len(results) < 2:
        return results
    model = model or os.environ.get("MNEMONICS_LLM_RERANK_MODEL", "meta/llama-3.3-70b-instruct")

    # Dedup by session id within the head — LLM only needs one chunk per session.
    head_idx: list[int] = []
    seen_sids: set[str] = set()
    for i, r in enumerate(results):
        if len(head_idx) >= top_n:
            break
        sid = _session_id_of(r.get("text"))
        if not sid or sid in seen_sids:
            continue
        seen_sids.add(sid)
        head_idx.append(i)
    if len(head_idx) < 2:
        return results

    head = [results[i] for i in head_idx]
    cands_block: list[str] = []
    for j, r in enumerate(head):
        text = r.get("text", "") or ""
        # Strip the "SID=xxx|" routing prefix so the LLM sees clean dialogue.
        if "|" in text:
            text = text.split("|", 1)[1]
        cands_block.append(f"[{j}] {text[:600]}")

    prompt = (
        "You are a retrieval judge. Pick the single passage that best contains "
        "the answer to the user's question.\n\n"
        f"Question: {query}\n\n"
        "Passages:\n" + "\n\n".join(cands_block) +
        f"\n\nReply with only the index number (0 to {len(head) - 1}). Index:"
    )

    raw = _llm_chat_with_backoff(prompt, model=model, max_tokens=8, temperature=0.0)
    if raw is None:
        return results

    m = _LLM_INT_RE.search(raw)
    if not m:
        return results
    chosen = int(m.group())
    if chosen < 0 or chosen >= len(head):
        return results
    if chosen == 0:
        return results  # already at top

    # Build the reordered list: chosen first, then everything else preserving order.
    chosen_global_idx = head_idx[chosen]
    chosen_item = results[chosen_global_idx]
    rest = [r for k, r in enumerate(results) if k != chosen_global_idx]
    return [chosen_item] + rest


def _rerank_fusion(results: list, rrf_k: float = 60.0) -> list:
    """Label-free rank ensemble: fuse the cross-encoder ranking with the
    retriever's (vec+BM25) ranking via Reciprocal Rank Fusion.

    After retrieve(rerank=True) the list is sorted by ce_score; each row also
    carries rrf_score — the pre-CE vec+BM25 fused rank score. The CE sometimes
    ranks the gold answer below a distractor it over-scores, while the
    retriever ranks it higher. Fusing the two *independent* rankings lets a
    strong retriever vote pull the gold back toward #1 without a single
    model's idiosyncratic error dominating. Reorders in place; adds no
    candidates (recall unchanged), needs no model, no training, no GPU.

    RRF score per row = 1/(k + ce_rank+1) + 1/(k + retr_rank+1), ranks 0-based.
    """
    if len(results) < 2:
        return results
    ce_rank = {r["id"]: i for i, r in enumerate(results)}  # current order = ce_score desc
    retr_sorted = sorted(
        results,
        key=lambda r: r.get("rrf_score", r.get("raw_score", 0.0)),
        reverse=True,
    )
    retr_rank = {r["id"]: i for i, r in enumerate(retr_sorted)}

    def _fused(r):
        rid = r["id"]
        return 1.0 / (rrf_k + ce_rank[rid] + 1) + 1.0 / (rrf_k + retr_rank[rid] + 1)

    return sorted(results, key=_fused, reverse=True)


def _trust_gate_rerank(query: str, results: list, ce, margin: float = 1.0) -> tuple[list, dict]:
    """Trust-gated FT-CE top-1 override (adaptmem Sprint 4 Stage 1, ported).

    A fine-tuned cross-encoder rescores the returned candidate band but is
    only allowed to replace the current #1 when it is *confident*: its best
    candidate must outscore the current top-1 by at least ``margin`` logits.
    Pure (always-on) FT-CE rerank regressed -3pp because it also overrode
    confident-correct champion rankings; the gate keeps the champion order
    unless the FT-CE strongly disagrees (Sprint 4: helped>0, hurt=0).

    Returns (possibly reordered results, gate_info) — gate_info carries the
    raw margin so per-question dumps allow an offline margin sweep without
    re-running the eval.
    """
    if len(results) < 2:
        return results, {"fired": False, "ftce_margin": None}

    def _clean(t: str | None) -> str:
        # Strip the "SID=<sid>|" bookkeeping prefix: the chat-ce-* checkpoints
        # were trained on raw chat text, the prefix is retrieval-only metadata.
        t = t or ""
        return t.split("|", 1)[1] if t.startswith("SID=") and "|" in t else t

    pairs = [(query, _clean(r.get("text"))) for r in results]
    scores = ce.predict(pairs, show_progress_bar=False)
    best = int(max(range(len(scores)), key=lambda j: float(scores[j])))
    gap = float(scores[best]) - float(scores[0])
    info = {
        "fired": False,
        "ftce_margin": round(gap, 4),
        "ftce_best_sid": _session_id_of(results[best].get("text")),
        "base_top1_sid": _session_id_of(results[0].get("text")),
    }
    if best != 0 and gap >= margin:
        info["fired"] = True
        results = [results[best]] + [r for j, r in enumerate(results) if j != best]
    return results, info


def _resolve_path(env_key: str, *candidates: Path) -> Path:
    """Env var varsa onu kullan, yoksa var olan ilk candidate'i döndür."""
    from_env = os.environ.get(env_key, "")
    if from_env:
        return Path(from_env)
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # yoksa ilkini döndür, çağıran hata verir

_THIS_DIR = Path(__file__).parent

DATA = _resolve_path(
    "LME_DATA",
    Path("/Users/macmini/Projects/metis-pair/benchmarks/data/longmemeval/longmemeval_s_cleaned.json"),
    _THIS_DIR / "data" / "longmemeval" / "longmemeval_s_cleaned.json",
    Path("/kaggle/input/longmemeval/longmemeval_s_cleaned.json"),
)
MEMPALACE_BASELINE = _resolve_path(
    "LME_BASELINE",
    Path("/Users/macmini/Projects/adaptmem/benchmarks/structural_memory_eval/entity_graph_result.json"),
    _THIS_DIR / "data" / "entity_graph_result.json",
    Path("/kaggle/input/longmemeval/entity_graph_result.json"),
)


def _session_text(session: list[dict]) -> str:
    """Flatten an LME session (list of {role, content}) into a single string."""
    return "\n".join(f"[{m.get('role','?')}] {m.get('content','')}" for m in session)


def _session_turn_chunks(sid: str, session: list[dict], session_date: str | None = None) -> list[str]:
    """Split a session into (user + assistant) turn-pair chunks, MemPalace-style.

    Each chunk = one user turn + the immediately following assistant turn.
    Preserves the SID= prefix so downstream session-id extraction still works.
    Consecutive user turns or trailing user-only turns are kept as their own chunk.

    When ``session_date`` is provided (LongMemEval haystack_dates), it is
    prepended to every chunk so temporal-reasoning queries embed near the
    sessions whose timestamps match the question's time references.
    """
    prefix = f"SID={sid}|"
    date_tag = f"[{session_date}] " if session_date else ""
    chunks: list[str] = []
    i = 0
    msgs = session
    while i < len(msgs):
        role = msgs[i].get("role", "")
        content = msgs[i].get("content", "")
        if role == "user":
            pair = f"{date_tag}[user] {content}"
            if i + 1 < len(msgs) and msgs[i + 1].get("role") == "assistant":
                pair += f"\n[assistant] {msgs[i + 1].get('content', '')}"
                i += 2
            else:
                i += 1
            chunks.append(prefix + pair)
        else:
            # assistant or system turn not preceded by user — store standalone
            chunks.append(prefix + f"{date_tag}[{role}] {content}")
            i += 1
    return chunks if chunks else [prefix + date_tag + _session_text(session)]


def _session_id_of(meta: str | None) -> str | None:
    """Extract LME session id from the meta we stored at ingest."""
    if not meta:
        return None
    # We prefix each text with "SID=<sid>|" at ingest time.
    if "SID=" in meta:
        return meta.split("SID=", 1)[1].split("|", 1)[0]
    return None


def evaluate_mnemonics(questions: list[dict], rerank: bool, top_k: int = 10,
                       candidate_k: int = 20,
                       augment_preferences: bool = False,
                       augment_assistant_facts: bool = False,
                       chunk_size: int = 200, chunk_overlap: int = 40,
                       chunk_mode: str = "word",
                       inject_dates: bool = False,
                       temporal_aware: bool = False,
                       llm_rerank_top_n: int = 0,
                       llm_rerank_margin: float = 0.0,
                       rerank_fusion: bool = False,
                       fusion_rrf_k: float = 60.0,
                       hyde: bool = False,
                       trust_gate_ce: str | None = None,
                       trust_gate_margin: float = 1.0,
                       per_q_out: Path | None = None) -> dict:
    """Run mnemonics retrieve() across every question, return aggregated metrics."""
    from mnemonics.store import Store
    from mnemonics.ingest import ingest, _get_encoder

    # Resolve encoder once up-front so we can size the per-question Store to
    # whatever dim the active model emits (MNEMONICS_ENCODER_MODEL may swap
    # all-MiniLM-L6-v2 -> 768d bge-base-en, etc.). Without this Store would
    # stay at 384d and hnswlib would refuse the larger vectors.
    enc = _get_encoder()
    store_dim = enc.get_sentence_embedding_dimension()
    print(f"  encoder dim={store_dim} (model={getattr(enc, '_first_module', lambda: None)() and ''}{os.environ.get('MNEMONICS_ENCODER_MODEL') or 'all-MiniLM-L6-v2'})", flush=True)
    from mnemonics.retrieve import retrieve

    gate_ce = None
    if trust_gate_ce:
        from sentence_transformers import CrossEncoder
        gate_ce = CrossEncoder(trust_gate_ce)
        print(f"  trust-gate CE loaded: {trust_gate_ce} (margin={trust_gate_margin})", flush=True)

    ks = [1, 5, 10]
    hits = {k: 0 for k in ks}
    by_type = defaultdict(lambda: {"n": 0, **{f"hit@{k}": 0 for k in ks}})
    per_q: list[dict] = []
    llm_fired = 0
    gate_fired = 0
    t0 = time.time()
    for i, q in enumerate(questions):
        # Fresh store per question — LongMemEval is per-question independent.
        with tempfile.TemporaryDirectory() as td:
            store = Store(td, dim=store_dim)
            # Ingest each session tagged with its sid.
            if chunk_mode == "turn":
                # MemPalace-style: each (user + assistant) turn pair = 1 chunk.
                # Pass pre-chunked texts; chunk_size=1 in ingest so no re-splitting.
                dates = q.get("haystack_dates") or []
                texts = []
                for idx, (sid, sess) in enumerate(zip(q["haystack_session_ids"], q["haystack_sessions"])):
                    sdate = dates[idx] if inject_dates and idx < len(dates) else None
                    texts.extend(_session_turn_chunks(sid, sess, sdate))
                ingest(texts=texts, store=store, ns="lme",
                       augment_preferences=augment_preferences,
                       augment_assistant_facts=augment_assistant_facts,
                       chunk_size=99999, chunk_overlap=0)
            else:
                texts = []
                for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"]):
                    texts.append(f"SID={sid}|{_session_text(sess)}")
                ingest(texts=texts, store=store, ns="lme",
                       augment_preferences=augment_preferences,
                       augment_assistant_facts=augment_assistant_facts,
                       chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not texts:
                continue
            query = q["question"]
            if inject_dates and q.get("question_date"):
                query = f"[{q['question_date']}] {query}"
            if hyde:
                # Concatenate (don't replace): the original query keeps the
                # retrieval anchored to literal wording, while the hypothetical
                # adds corpus-style vocabulary the chunks actually use. This is
                # safer than pure HyDE since hallucinated details can't fully
                # hijack the embedding.
                hyp = _hyde_passage(q["question"])
                if hyp:
                    query = f"{query} {hyp}"
            try:
                result = retrieve(
                    query=query,
                    store=store,
                    ns="lme",
                    top_k=top_k,
                    candidate_k=candidate_k,
                    rerank=rerank,
                )
            except RuntimeError as e:
                print(f"  q{i} ERROR: {e}", file=sys.stderr)
                continue

            # Rank-fusion post-rerank (label-free): ensemble the CE ranking with
            # the retriever's vec+BM25 ranking so a CE mis-rank gets corrected
            # when the retriever strongly disagrees. Runs before temporal-aware
            # so temporal promotion still wins on temporal queries.
            if rerank_fusion:
                result["results"] = _rerank_fusion(result["results"], fusion_rrf_k)

            # Trust-gated FT-CE override (Sprint 4 pattern): the fine-tuned CE
            # replaces the champion top-1 only when its score margin clears the
            # gate. Runs before temporal-aware so temporal promotion still wins
            # on temporal queries (same stage order as adaptmem Sprint 4).
            gate_info = None
            if gate_ce is not None:
                result["results"], gate_info = _trust_gate_rerank(
                    q.get("question", ""), result["results"], gate_ce,
                    trust_gate_margin)
                if gate_info["fired"]:
                    gate_fired += 1

            # Temporal-aware post-rerank. Only fires when the query contains a
            # relative-time expression ("N weeks/days/months ago"); other queries
            # pass through untouched. Sessions whose date falls inside the
            # computed target window get pushed to the front of the candidate
            # list, preserving their relative order among themselves and the
            # original order of the remaining sessions. Targets the temporal-
            # reasoning recall failures we measured (2/4 top-10 misses).
            if temporal_aware:
                sid_to_date: dict[str, datetime] = {}
                for sid_x, d_str in zip(
                    q.get("haystack_session_ids", []),
                    q.get("haystack_dates", []) or [],
                ):
                    sdate = _parse_lme_date(d_str)
                    if sdate is not None:
                        sid_to_date[sid_x] = sdate

                def _rdate(r):
                    return sid_to_date.get(_session_id_of(r.get("text")) or "")

                target_info = _detect_relative_target(
                    q.get("question", ""), q.get("question_date")
                )
                if sid_to_date and target_info is not None:
                    # "N ago": promote in-window candidates AND, within the window,
                    # rank by closeness to the target date so the date-correct chunk
                    # wins #1 (the CE frequently leaves the gold stuck at rank 2).
                    target_date, tol = target_info

                    def _in_win(r):
                        d = _rdate(r)
                        return d is not None and abs((d - target_date).days) <= tol

                    in_window = [r for r in result["results"] if _in_win(r)]
                    out_window = [r for r in result["results"] if not _in_win(r)]
                    in_window.sort(key=lambda r: abs((_rdate(r) - target_date).days))
                    result["results"] = in_window + out_window
                elif sid_to_date and q.get("question_type") == "temporal-reasoning":
                    # Ordinal/comparative ("first/earliest/order" vs "last/latest"):
                    # sort dated candidates chronologically so the chronological
                    # extreme lands at #1; undated keep their CE order behind.
                    # Gated to temporal-reasoning questions: "first/last" fire on
                    # non-temporal queries ("last name", "first purchase") ~6.5% of
                    # the time and would corrupt currently-correct answers. The gate
                    # uses the dataset label, so this measures the lever's ceiling;
                    # production would route via temporal-intent detection instead.
                    direction = _detect_ordinal(q.get("question", ""))
                    if direction is not None:
                        dated = [r for r in result["results"] if _rdate(r) is not None]
                        undated = [r for r in result["results"] if _rdate(r) is None]
                        dated.sort(key=_rdate, reverse=(direction == "desc"))
                        result["results"] = dated + undated

            # LLM-as-judge rerank over top-N session-unique candidates. Runs AFTER
            # temporal_aware so the LLM also weighs temporally-promoted results.
            # Scoped by --llm-rerank-margin: naive (every-question) rerank regressed
            # -1pp because the LLM added noise on easy questions the CE already
            # nailed. The gate fires the LLM only when the CE is *uncertain* — the
            # softmax gap between the two best candidate scores is below the margin
            # — so confident-correct CE rankings are left untouched. Cost: 1 LLM
            # call per fired question (printed at the end).
            if llm_rerank_top_n > 0:
                fire = True
                if llm_rerank_margin > 0.0:
                    import math
                    top = sorted((r.get("score", 0.0) for r in result["results"]),
                                 reverse=True)[:8]
                    if len(top) >= 2:
                        mx = top[0]
                        exps = [math.exp(s - mx) for s in top]
                        Z = sum(exps) or 1.0
                        fire = (exps[0] - exps[1]) / Z < llm_rerank_margin
                if fire:
                    llm_fired += 1
                    result["results"] = _llm_rerank_topk(
                        query=q.get("question", ""),
                        results=result["results"],
                        top_n=llm_rerank_top_n,
                    )

            answer_sids = set(q.get("answer_session_ids") or [])
            retrieved_sids: list[str] = []
            for r in result["results"]:
                sid = _session_id_of(r.get("text"))
                if sid and sid not in retrieved_sids:
                    retrieved_sids.append(sid)

            qtype = q.get("question_type", "unknown")
            by_type[qtype]["n"] += 1
            q_hits = {}
            for k in ks:
                hit = any(sid in answer_sids for sid in retrieved_sids[:k])
                q_hits[k] = hit
                if hit:
                    hits[k] += 1
                    by_type[qtype][f"hit@{k}"] += 1
            if per_q_out is not None:
                per_q.append({
                    "qid": q.get("question_id"),
                    "qtype": qtype,
                    "question": q.get("question"),
                    "answer": q.get("answer"),
                    "answer_sids": list(answer_sids),
                    "retrieved_top10_sids": retrieved_sids[:10],
                    "hit@1": q_hits[1], "hit@5": q_hits[5], "hit@10": q_hits[10],
                    **({"gate": gate_info} if gate_info is not None else {}),
                })

        step = 1 if len(questions) <= 10 else 5
        if (i + 1) % step == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(questions) - i - 1) / max(rate, 1e-6)
            print(
                f"  [{i+1}/{len(questions)}] "
                f"R@1={hits[1]/(i+1):.3f} R@5={hits[5]/(i+1):.3f} "
                f"R@10={hits[10]/(i+1):.3f}  "
                f"rate={rate:.2f} q/s  eta={eta:.0f}s",
                flush=True,
            )

    if llm_rerank_top_n > 0:
        print(f"  LLM rerank fired on {llm_fired}/{len(questions)} questions "
              f"(margin={llm_rerank_margin}, 0=always)", flush=True)
    if gate_ce is not None:
        print(f"  trust gate fired on {gate_fired}/{len(questions)} questions "
              f"(margin={trust_gate_margin})", flush=True)

    n = sum(v["n"] for v in by_type.values())
    out = {
        "mode": "rerank" if rerank else "no_rerank",
        "n": n,
        "runtime_s": round(time.time() - t0, 1),
        **{f"R@{k}": round(hits[k] / max(n, 1), 4) for k in ks},
        "by_type": {
            t: {
                "n": v["n"],
                **{f"R@{k}": round(v[f"hit@{k}"] / max(v["n"], 1), 4) for k in ks},
            }
            for t, v in by_type.items()
        },
    }
    if gate_ce is not None:
        out["trust_gate"] = {"model": trust_gate_ce, "margin": trust_gate_margin,
                             "fired": gate_fired}
    if per_q_out is not None:
        per_q_out.write_text(json.dumps(per_q, indent=2))
    return out


def mempalace_baseline_summary() -> dict:
    """Read AdaptMem's entity-graph baseline (already 500-q full)."""
    if not MEMPALACE_BASELINE.exists():
        return {"error": f"missing {MEMPALACE_BASELINE}"}
    d = json.load(open(MEMPALACE_BASELINE))
    s = d.get("summary", {})
    return {
        "mode": "mempalace_entity_graph",
        "n": s.get("n"),
        "R@1": s.get("R@1"),
        "R@5": s.get("R@5"),
        "R@10": s.get("R@10"),
        "by_type": {
            t: {"n": v["n"], "R@1": v["R@1"], "R@5": v["R@5"], "R@10": v["R@10"]}
            for t, v in (s.get("by_type") or {}).items()
        },
        "runtime_s": s.get("runtime_s"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="Question subset size (ignored if --split-file given)")
    ap.add_argument("--split-file", type=Path, default=None,
                    help="JSON with {'dev': [qid,...]} — subset by exact ids (apples-to-apples with MemPalace)")
    ap.add_argument("--mode", choices=["both", "no_rerank", "rerank"], default="both")
    ap.add_argument("--augment-preferences", action="store_true",
                    help="Pass augment_preferences=True to ingest (synth pref docs)")
    ap.add_argument("--augment-assistant-facts", action="store_true",
                    help="Pass augment_assistant_facts=True to ingest (synth numeric-fact docs over assistant turns)")
    ap.add_argument("--candidate-k", type=int, default=20,
                    help="Per-channel candidate band before fusion (default 20, MemPalace uses 50)")
    ap.add_argument("--chunk-size", type=int, default=200, help="Words per ingest chunk (default 200)")
    ap.add_argument("--chunk-overlap", type=int, default=40, help="Overlap words between adjacent chunks (default 40)")
    ap.add_argument("--chunk-mode", choices=["word", "turn"], default="word",
                    help="'word': sliding window (default); 'turn': one user+assistant pair per chunk (MemPalace-style)")
    ap.add_argument("--inject-dates", action="store_true",
                    help="Prepend haystack_dates to each chunk and question_date to the query (turn-mode only). Targets temporal-reasoning lift.")
    ap.add_argument("--temporal-aware", action="store_true",
                    help="Post-retrieval: when the query says 'N weeks/days/months ago', push sessions inside that date window to the front of the candidate list. Cheap, opt-in, no embedding change.")
    ap.add_argument("--llm-rerank-top-n", type=int, default=0,
                    help="If >0, send the top-N session-unique candidates to an LLM judge (NVIDIA NIM Llama 3.3 70B by default) which picks the index containing the answer. Requires NVIDIA_API_KEY. 0 disables.")
    ap.add_argument("--llm-rerank-margin", type=float, default=0.0,
                    help="Scope --llm-rerank to uncertain cases: fire the LLM only when the softmax gap between the two best candidate scores is below this value. 0 = fire on every question (legacy, regressed -1pp). Try 0.3.")
    ap.add_argument("--rerank-fusion", action="store_true",
                    help="Label-free rank ensemble: after CE rerank, fuse the CE ranking with the retriever's vec+BM25 ranking via RRF. Corrects CE mis-ranks when the retriever disagrees. No model/training/GPU. Try with the 0.954 baseline config.")
    ap.add_argument("--fusion-rrf-k", type=float, default=60.0,
                    help="RRF damping constant for --rerank-fusion (default 60). Lower = rank-1 dominates more.")
    ap.add_argument("--hyde", action="store_true",
                    help="Hypothetical Document Embeddings: ask an LLM (NIM Llama 3.3 70B) to draft a user-style passage that would answer the question, then append it to the query so the embedding lands closer to the corpus. Requires NVIDIA_API_KEY.")
    ap.add_argument("--trust-gate-ce", type=str, default=None,
                    help="Path/name of a fine-tuned CrossEncoder used as a trust-gated #1 override: it rescores the returned candidates and replaces the top-1 only when its pick beats the current top-1 by --trust-gate-margin logits. Pure (always-on) FT-CE rerank regressed -3pp; the gate is the proven variant (adaptmem Sprint 4 Stage 1).")
    ap.add_argument("--trust-gate-margin", type=float, default=1.0,
                    help="Logit margin the trust-gate CE must clear to override #1 (default 1.0, the adaptmem Sprint 4 value).")
    ap.add_argument("--out", type=Path, default=Path("/tmp/mnemonics_vs_mempalace.json"))
    ap.add_argument("--per-q-out", type=Path, default=None,
                    help="Optional path to dump per-question hit/miss records")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading {DATA}...", flush=True)
    all_q = json.load(open(DATA))
    if args.split_file:
        split = json.load(open(args.split_file))
        dev_ids = set(split.get("dev", []))
        questions = [q for q in all_q if q.get("question_id") in dev_ids]
        print(f"Split file: {len(questions)}/{len(dev_ids)} dev ids found in dataset", flush=True)
    elif args.n < len(all_q):
        import random
        rng = random.Random(args.seed)
        questions = rng.sample(all_q, args.n)
    else:
        questions = all_q
    print(f"Eval set: {len(questions)} questions (of {len(all_q)})", flush=True)
    am_path = os.environ.get("MNEMONICS_ADAPTMEM_PATH")
    print(f"MNEMONICS_ADAPTMEM_PATH = {am_path or '<unset, default MiniLM>'}", flush=True)

    results = {"n_questions": len(questions), "mempalace_full": mempalace_baseline_summary()}

    if args.mode in ("both", "no_rerank"):
        print(f"\n=== Mnemonics (no CE rerank) augment_prefs={args.augment_preferences} augment_facts={args.augment_assistant_facts} cand_k={args.candidate_k} chunk={args.chunk_size}/{args.chunk_overlap} mode={args.chunk_mode} dates={args.inject_dates} temporal_aware={args.temporal_aware} llm_rerank_top_n={args.llm_rerank_top_n} hyde={args.hyde} ===", flush=True)
        results["mnemonics_no_rerank"] = evaluate_mnemonics(
            questions, rerank=False, candidate_k=args.candidate_k,
            augment_preferences=args.augment_preferences,
            augment_assistant_facts=args.augment_assistant_facts,
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
            chunk_mode=args.chunk_mode,
            inject_dates=args.inject_dates,
            temporal_aware=args.temporal_aware,
            llm_rerank_top_n=args.llm_rerank_top_n,
            llm_rerank_margin=args.llm_rerank_margin,
            rerank_fusion=args.rerank_fusion,
            fusion_rrf_k=args.fusion_rrf_k,
            hyde=args.hyde,
            trust_gate_ce=args.trust_gate_ce,
            trust_gate_margin=args.trust_gate_margin,
        )

    if args.mode in ("both", "rerank"):
        print(f"\n=== Mnemonics (CE rerank) augment_prefs={args.augment_preferences} augment_facts={args.augment_assistant_facts} cand_k={args.candidate_k} chunk={args.chunk_size}/{args.chunk_overlap} mode={args.chunk_mode} dates={args.inject_dates} temporal_aware={args.temporal_aware} llm_rerank_top_n={args.llm_rerank_top_n} hyde={args.hyde} ===", flush=True)
        results["mnemonics_rerank"] = evaluate_mnemonics(
            questions, rerank=True, candidate_k=args.candidate_k,
            augment_preferences=args.augment_preferences,
            augment_assistant_facts=args.augment_assistant_facts,
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
            chunk_mode=args.chunk_mode,
            inject_dates=args.inject_dates,
            temporal_aware=args.temporal_aware,
            llm_rerank_top_n=args.llm_rerank_top_n,
            llm_rerank_margin=args.llm_rerank_margin,
            rerank_fusion=args.rerank_fusion,
            fusion_rrf_k=args.fusion_rrf_k,
            hyde=args.hyde,
            trust_gate_ce=args.trust_gate_ce,
            trust_gate_margin=args.trust_gate_margin,
            per_q_out=args.per_q_out,
        )

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}", flush=True)
    print("\n--- HEAD-TO-HEAD ---", flush=True)
    for key in ("mempalace_full", "mnemonics_no_rerank", "mnemonics_rerank"):
        v = results.get(key)
        if not v:
            continue
        print(
            f"{key:25} n={v.get('n') or 0:<4} "
            f"R@1={v.get('R@1')}  R@5={v.get('R@5')}  R@10={v.get('R@10')}"
        )


if __name__ == "__main__":
    main()
