"""Retrieval eval harness: MRR, Recall@k, NDCG@10 across encoders.

Test data format (JSONL):
  corpus.jsonl  : {"id": "<chunk_id>", "text": "..."}
  queries.jsonl : {"query": "...", "relevant_id": "<chunk_id>"}

Encoders:
  minilm           default base (all-MiniLM-L6-v2)
  adaptmem         fine-tuned checkpoint at --model-path or MNEMONICS_ADAPTMEM_PATH
  <any HF id>      treated as a sentence-transformers model id

Retrieval methods (orthogonal to encoder choice):
  vector           cosine similarity only (default)
  hybrid           vector top-N + BM25 (SQLite FTS5) top-N, fused with RRF
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

_RRF_K = 60.0


def reciprocal_rank(hits: list[str], relevant_id: str) -> float:
    for i, h in enumerate(hits, start=1):
        if h == relevant_id:
            return 1.0 / i
    return 0.0


def recall_at_k(hits: list[str], relevant_id: str, k: int) -> float:
    return 1.0 if relevant_id in hits[:k] else 0.0


def ndcg_at_k(hits: list[str], relevant_id: str, k: int) -> float:
    # Binary relevance + single relevant doc:
    #   DCG  = 1/log2(rank+1) when hit lands at `rank`, else 0
    #   IDCG = 1/log2(2) = 1 (best case rank 1)
    #   → NDCG = DCG / 1
    for i, h in enumerate(hits[:k], start=1):
        if h == relevant_id:
            return 1.0 / math.log2(i + 1)
    return 0.0


def aggregate(per_query: list[dict]) -> dict:
    n = len(per_query)
    if n == 0:
        return {"n": 0, "mrr": 0.0, "r@5": 0.0, "r@10": 0.0, "ndcg@10": 0.0}
    return {
        "n": n,
        "mrr": sum(q["rr"] for q in per_query) / n,
        "r@5": sum(q["r5"] for q in per_query) / n,
        "r@10": sum(q["r10"] for q in per_query) / n,
        "ndcg@10": sum(q["ndcg10"] for q in per_query) / n,
    }


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_encoder(encoder: str, model_path: str | None) -> Any:
    from sentence_transformers import SentenceTransformer

    if encoder == "minilm":
        return SentenceTransformer("all-MiniLM-L6-v2")
    if encoder == "adaptmem":
        path = model_path or os.environ.get("MNEMONICS_ADAPTMEM_PATH")
        if not path:
            raise ValueError(
                "adaptmem encoder requires --model-path or MNEMONICS_ADAPTMEM_PATH"
            )
        p = Path(path).expanduser()
        # Trained checkpoints may live under <root>/model/.
        candidate = p / "model" if (p / "model").exists() else p
        return SentenceTransformer(str(candidate))
    return SentenceTransformer(encoder)


def _build_bm25_index(chunk_texts: list[str]) -> sqlite3.Connection:
    """Build an in-memory FTS5 index over the corpus. rowid = list index."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE VIRTUAL TABLE fts USING fts5("
        "text, tokenize='unicode61 remove_diacritics 2')"
    )
    conn.executemany(
        "INSERT INTO fts(rowid, text) VALUES (?, ?)",
        [(i, t) for i, t in enumerate(chunk_texts)],
    )
    return conn


def _bm25_rank(
    conn: sqlite3.Connection,
    query: str,
    chunk_ids: list[str],
    top_k: int,
) -> list[str]:
    """Return up to top_k chunk_ids best-matching the query, or [] on no match."""
    from mnemonics.store import Store  # share the sanitizer

    match = Store._fts_sanitize(query)
    if not match:
        return []
    try:
        rows = conn.execute(
            "SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?",
            (match, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [chunk_ids[r[0]] for r in rows]


def rrf_fuse_ids(rankings: list[list[str]], top_k: int, rrf_k: float = _RRF_K) -> list[str]:
    """Reciprocal Rank Fusion over id-only ranked lists."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return [doc_id for doc_id, _ in sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]


def run_eval(
    corpus_path: str | Path,
    queries_path: str | Path,
    encoder: str = "minilm",
    model_path: str | None = None,
    top_k: int = 10,
    method: str = "vector",
    candidate_k: int = 20,
) -> dict:
    if method not in ("vector", "hybrid"):
        raise ValueError(f"unknown method: {method}")

    corpus = _load_jsonl(Path(corpus_path))
    queries = _load_jsonl(Path(queries_path))
    enc = _build_encoder(encoder, model_path)

    chunk_ids = [c["id"] for c in corpus]
    chunk_texts = [c["text"] for c in corpus]

    corpus_vecs = enc.encode(
        chunk_texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    query_vecs = enc.encode(
        [q["query"] for q in queries],
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    bm25_conn = _build_bm25_index(chunk_texts) if method == "hybrid" else None

    per_query = []
    for q, qv in zip(queries, query_vecs):
        scores = corpus_vecs @ qv
        if method == "hybrid":
            vec_top = np.argsort(-scores)[:candidate_k]
            vec_hits = [chunk_ids[int(i)] for i in vec_top]
            bm25_hits = _bm25_rank(bm25_conn, q["query"], chunk_ids, candidate_k)
            hits = rrf_fuse_ids([vec_hits, bm25_hits], top_k=top_k)
        else:
            top = np.argsort(-scores)[:top_k]
            hits = [chunk_ids[int(i)] for i in top]
        rel = q["relevant_id"]
        per_query.append(
            {
                "query": q["query"],
                "relevant_id": rel,
                "hits": hits,
                "rr": reciprocal_rank(hits, rel),
                "r5": recall_at_k(hits, rel, 5),
                "r10": recall_at_k(hits, rel, 10),
                "ndcg10": ndcg_at_k(hits, rel, 10),
            }
        )

    enc_label = encoder if encoder in ("minilm", "adaptmem") else f"hf:{encoder}"
    label = enc_label if method == "vector" else f"{enc_label}+hybrid"
    return {
        "encoder": label,
        "method": method,
        "agg": aggregate(per_query),
        "per_query": per_query,
    }


def compare_table(results: dict[str, dict]) -> str:
    """Render a side-by-side metric table. `results` keys = column labels."""
    labels = list(results.keys())
    header = ["metric"] + labels
    metrics = ["mrr", "r@5", "r@10", "ndcg@10"]
    rows = [header]
    for m in metrics:
        rows.append([m] + [f"{results[lab]['agg'][m]:.4f}" for lab in labels])
    if len(labels) >= 2:
        base, *rest = labels
        for i, m in enumerate(metrics, start=1):
            base_val = results[base]["agg"][m]
            for j, lab in enumerate(rest, start=2):
                delta = results[lab]["agg"][m] - base_val
                rows[i][j] += f" ({'+' if delta >= 0 else ''}{delta:.4f})"
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows)
