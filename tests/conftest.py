"""Shared fixtures."""

import numpy as np
import pytest

from mnemonics.store import DIM, Store


@pytest.fixture(autouse=True)
def _isolate_rerank_cache(tmp_path, monkeypatch):
    """Keep the persistent rerank-score cache out of tests.

    The cache is on by default in production and a shared ``$HOME`` database
    would let one test's scores satisfy another test's rerank call, silently
    skipping the model.  Each test therefore gets its own file.
    """
    monkeypatch.setenv("MNEMONICS_RERANK_CACHE", str(tmp_path / "rerank_scores.db"))



@pytest.fixture
def tmp_store(tmp_path):
    """Fresh Store in a temp directory."""
    return Store(tmp_path)


@pytest.fixture
def populated_store(tmp_store):
    """Store with 5 pre-ingested documents."""
    docs = [
        "The Eiffel Tower is located in Paris, France.",
        "Python is a high-level programming language created by Guido van Rossum.",
        "The speed of light is approximately 299,792 km/s in vacuum.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "Machine learning is a subset of artificial intelligence.",
    ]
    rng = np.random.default_rng(42)
    vecs = rng.random((len(docs), DIM)).astype("float32")
    # normalize
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    tmp_store.add(docs, vecs)
    return tmp_store, docs, vecs
