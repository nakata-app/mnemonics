"""Tests for mnemonics.ingest."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from mnemonics.ingest import _chunk, ingest
from mnemonics.store import Store, DIM


# ── _chunk ───────────────────────────────────────────────────────────────────

def test_chunk_short_text():
    text = "hello world"
    assert _chunk(text, size=200) == [text]


def test_chunk_exact_size():
    words = " ".join(f"w{i}" for i in range(200))
    assert _chunk(words, size=200) == [words]


def test_chunk_splits_long_text():
    words = " ".join(f"w{i}" for i in range(400))
    chunks = _chunk(words, size=200, overlap=0)
    assert len(chunks) == 2
    assert chunks[0].startswith("w0")
    assert chunks[1].startswith("w200")


def test_chunk_overlap():
    words = " ".join(f"w{i}" for i in range(300))
    chunks = _chunk(words, size=200, overlap=50)
    # Second chunk should start at word 150
    second_start = chunks[1].split()[0]
    assert second_start == "w150"


def test_chunk_overlap_produces_more_chunks():
    words = " ".join(f"w{i}" for i in range(400))
    no_overlap = _chunk(words, size=200, overlap=0)
    with_overlap = _chunk(words, size=200, overlap=100)
    assert len(with_overlap) > len(no_overlap)


def test_chunk_empty_string():
    assert _chunk("") == [""]


def test_chunk_single_word():
    assert _chunk("hello") == ["hello"]


# ── ingest (mocked encoder) ──────────────────────────────────────────────────

def _mock_encoder(n_chunks: int) -> MagicMock:
    enc = MagicMock()
    rng = np.random.default_rng(0)
    vecs = rng.random((n_chunks, DIM)).astype("float32")
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    enc.encode.return_value = vecs
    return enc


@pytest.fixture
def mock_enc():
    with patch("mnemonics.ingest._get_encoder") as mock:
        yield mock


def test_ingest_returns_chunk_count(tmp_store, mock_enc):
    mock_enc.return_value = _mock_encoder(3)
    n = ingest(["doc one", "doc two", "doc three"], tmp_store)
    assert n == 3
    assert tmp_store.count() == 3


def test_ingest_empty_list(tmp_store, mock_enc):
    n = ingest([], tmp_store)
    assert n == 0
    assert tmp_store.count() == 0


def test_ingest_with_namespace(tmp_store, mock_enc):
    mock_enc.return_value = _mock_encoder(2)
    ingest(["a", "b"], tmp_store, ns="custom")
    assert tmp_store.count("custom") == 2
    assert tmp_store.count("default") == 0


def test_ingest_meta_passthrough(tmp_store, mock_enc):
    mock_enc.return_value = _mock_encoder(2)
    meta = [{"author": "alice"}, {"author": "bob"}]
    ingest(["text one", "text two"], tmp_store, meta=meta)
    results = tmp_store.search(mock_enc.return_value.encode.return_value[0], top_k=2)
    authors = {r["meta"].get("author") for r in results}
    assert "alice" in authors or "bob" in authors


def test_ingest_source_idx_in_meta(tmp_store, mock_enc):
    mock_enc.return_value = _mock_encoder(2)
    ingest(["first", "second"], tmp_store)
    results = tmp_store.search(mock_enc.return_value.encode.return_value[0], top_k=2)
    source_idxs = {r["meta"]["source_idx"] for r in results}
    assert source_idxs == {0, 1}


def test_ingest_long_text_produces_multiple_chunks(tmp_store, mock_enc):
    long_text = " ".join(f"word{i}" for i in range(500))
    expected_chunks = len(_chunk(long_text, size=200, overlap=40))
    mock_enc.return_value = _mock_encoder(expected_chunks)
    n = ingest([long_text], tmp_store, chunk_size=200, chunk_overlap=40)
    assert n == expected_chunks
    assert tmp_store.count() == expected_chunks


def test_ingest_multiple_docs_all_stored(tmp_store, mock_enc):
    texts = [f"document {i}" for i in range(10)]
    mock_enc.return_value = _mock_encoder(10)
    n = ingest(texts, tmp_store)
    assert n == 10
    assert tmp_store.count() == 10


def test_ingest_calls_encode_once(tmp_store, mock_enc):
    mock_enc.return_value = _mock_encoder(3)
    ingest(["a", "b", "c"], tmp_store)
    assert mock_enc.return_value.encode.call_count == 1


def test_ingest_batch_size_param(tmp_store, mock_enc):
    enc = _mock_encoder(2)
    mock_enc.return_value = enc
    ingest(["x", "y"], tmp_store)
    call_kwargs = enc.encode.call_args[1]
    assert call_kwargs.get("batch_size") == 64
    assert call_kwargs.get("normalize_embeddings") is True
