"""Label-free parity between the LongMemEval harness and the library path."""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np

import mnemonics.ingest as ingest_mod
import mnemonics.retrieve as retrieve_mod
from mnemonics.ingest import ingest, turn_pair_chunks
from mnemonics.retrieve import retrieve
from mnemonics.store import Store

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "lme_library_parity", _REPO / "benchmarks" / "longmemeval_eval.py"
)
H = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(H)


class StubEncoder:
    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts, **_kwargs):
        return np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (len(texts), 1))


class StubReranker:
    backend = "stub"

    def rerank(self, query, documents, max_length=None, batch_size=None):
        del query, max_length, batch_size

        def score(text: str) -> float:
            low = text.lower()
            if "user has mentioned: jasmine tea with lemon" in low:
                return 100.0
            if "jasmine tea with lemon" in low:
                return 50.0
            if "black coffee" in low:
                return 20.0
            return 1.0

        ranked = sorted(
            ((i, score(text)) for i, text in enumerate(documents)),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked


def _install_stubs(monkeypatch):
    enc = StubEncoder()
    reranker = StubReranker()
    monkeypatch.setattr(ingest_mod, "_get_encoder", lambda *_a, **_k: enc)
    monkeypatch.setattr(retrieve_mod, "_get_encoder", lambda *_a, **_k: enc)
    monkeypatch.setattr(retrieve_mod, "_get_rerank_ce", lambda *_a, **_k: reranker)
    return enc, reranker


def test_turn_pair_chunks_preserve_pairing_and_standalone_messages():
    messages = [
        {"role": "system", "content": "context"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
        {"role": "assistant", "content": "orphan"},
    ]
    assert turn_pair_chunks(messages) == [
        "[system] context",
        "[user] u1\n[assistant] a1",
        "[user] u2",
        "[user] u3\n[assistant] a3",
        "[assistant] orphan",
    ]
    assert turn_pair_chunks([]) == [""]


def test_harness_wrapper_adds_only_lme_metadata():
    session = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert H._session_turn_chunks("s1", session) == [
        "SID=s1|[user] hello\n[assistant] hi"
    ]
    assert H._session_turn_chunks("s1", session, "2026-01-02") == [
        "SID=s1|[2026-01-02] [user] hello\n[assistant] hi"
    ]


def test_label_free_harness_and_library_paths_are_identical(monkeypatch, tmp_path):
    _install_stubs(monkeypatch)
    question = {
        "question_id": "q1",
        "question_type": "preference-following",
        "question": "Which drink did I say I prefer?",
        "answer": "Jasmine tea with lemon.",
        "answer_session_ids": ["s-good"],
        "haystack_session_ids": ["s-other", "s-good"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I prefer black coffee in the morning."},
                {"role": "assistant", "content": "Got it."},
            ],
            [
                {"role": "user", "content": "I prefer jasmine tea with lemon."},
                {"role": "assistant", "content": "I will remember that."},
            ],
        ],
    }

    dump = tmp_path / "candidates.json"
    summary = H.evaluate_mnemonics(
        [question],
        rerank=True,
        top_k=10,
        candidate_k=50,
        augment_preferences=True,
        chunk_mode="turn",
        dump_candidates=dump,
    )
    harness_rows = json.loads(dump.read_text())[0]["rows"]

    texts: list[str] = []
    for sid, session in zip(
        question["haystack_session_ids"], question["haystack_sessions"]
    ):
        texts.extend(f"SID={sid}|{chunk}" for chunk in turn_pair_chunks(session))

    with tempfile.TemporaryDirectory() as td:
        store = Store(td, dim=4)
        ingest(
            texts=texts,
            store=store,
            ns="lme",
            augment_preferences=True,
            chunk_size=99999,
            chunk_overlap=0,
        )
        direct = retrieve(
            query=question["question"],
            store=store,
            ns="lme",
            top_k=10,
            candidate_k=50,
            rerank=True,
        )

    direct_rows = [
        {"sid": H._session_id_of(row["text"]), "text": row["text"]}
        for row in direct["results"]
    ]
    assert harness_rows == direct_rows
    assert harness_rows[0]["sid"] == "s-good"
    assert "User has mentioned: jasmine tea with lemon" in harness_rows[0]["text"]
    assert summary["R@1"] == 1.0
