"""Cross-encoder rerankers behind one explicit interface.

Retrieval needs exactly one thing from a cross-encoder: score (query, document)
pairs, bounded by ``max_length`` and ``batch_size`` so a long-row band cannot
OOM the host. Every backend honours both limits identically, so a deployment
never changes behaviour just because an optional package happens to be
installed.

Backend choice is explicit and observable:

- ``MNEMONICS_RERANK_BACKEND`` = ``auto`` (default), ``adaptmem`` or
  ``sentence-transformers``.
- ``auto`` prefers adaptmem when it is installed *and* supports the limits,
  otherwise uses sentence-transformers. The reason for any fallback is logged.
- Backend failures raise ``RerankerError``; nothing is swallowed.
"""
from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Sequence
from typing import Any, Protocol

log = logging.getLogger("mnemonics")

BACKEND_ENV = "MNEMONICS_RERANK_BACKEND"
BACKENDS = ("auto", "adaptmem", "sentence-transformers")


class RerankerError(RuntimeError):
    """A reranker backend failed to load or to score."""


class RerankerUnavailable(RerankerError):
    """The backend's package is not installed."""


class RerankerIncompatible(RerankerUnavailable):
    """The backend is installed but cannot honour the reranker contract."""


class Reranker(Protocol):
    """Score documents against a query with a cross-encoder.

    Returns ``[(document_index, score), ...]`` sorted by score, best first.
    ``max_length`` and ``batch_size`` of ``None`` mean the model's defaults.
    """

    name: str
    backend: str

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        max_length: int | None = None,
        batch_size: int | None = None,
    ) -> list[tuple[int, float]]: ...


class SentenceTransformersReranker:
    """Bare ``sentence_transformers.CrossEncoder``."""

    backend = "sentence-transformers"

    def __init__(self, name: str) -> None:
        self.name = name
        self._ce: Any = None
        self._loaded_max_length: int | None = None

    def _load(self, max_length: int | None) -> Any:
        # max_length is fixed when the CrossEncoder is built, so a different
        # requested length reloads it.
        if self._ce is not None and self._loaded_max_length == max_length:
            return self._ce
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RerankerUnavailable(
                "rerank=True requires either 'adaptmem' (with rerank support) or "
                "'sentence-transformers'. Install with `pip install sentence-transformers`."
            ) from e
        kwargs: dict[str, Any] = {} if max_length is None else {"max_length": max_length}
        try:
            self._ce = CrossEncoder(self.name, **kwargs)
        except Exception as e:
            raise RerankerError(f"could not load cross-encoder {self.name!r}: {e}") from e
        self._loaded_max_length = max_length
        return self._ce

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        max_length: int | None = None,
        batch_size: int | None = None,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        ce = self._load(max_length)
        kwargs: dict[str, Any] = {"show_progress_bar": False}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        try:
            scores = ce.predict([(query, d) for d in documents], **kwargs)
        except Exception as e:
            raise RerankerError(f"cross-encoder {self.name!r} failed to score: {e}") from e
        ranked = sorted(enumerate(scores), key=lambda x: -float(x[1]))
        return [(int(i), float(s)) for i, s in ranked]


class AdaptMemReranker:
    """``adaptmem.AdaptMem.rerank``; needs a release that accepts the limits."""

    backend = "adaptmem"
    _REQUIRED = ("max_length", "batch_size")

    def __init__(self, name: str) -> None:
        self.name = name
        try:
            import adaptmem
            from adaptmem import AdaptMem
        except ImportError as e:
            raise RerankerUnavailable("adaptmem is not installed") from e
        try:
            rerank_method = AdaptMem.rerank
        except AttributeError as e:
            raise RerankerIncompatible(
                f"installed adaptmem {getattr(adaptmem, '__version__', '?')} has no rerank(); "
                "upgrade adaptmem"
            ) from e
        accepted = inspect.signature(rerank_method).parameters
        missing = [p for p in self._REQUIRED if p not in accepted]
        if missing:
            raise RerankerIncompatible(
                f"installed adaptmem {getattr(adaptmem, '__version__', '?')} does not accept "
                f"{', '.join(missing)} in rerank(); upgrade adaptmem"
            )
        self._am = AdaptMem(rerank_model=name)

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        max_length: int | None = None,
        batch_size: int | None = None,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        try:
            ranked = self._am.rerank(
                query, list(documents), max_length=max_length, batch_size=batch_size
            )
        except Exception as e:
            raise RerankerError(f"adaptmem reranker {self.name!r} failed: {e}") from e
        return [(int(i), float(s)) for i, s in ranked]


def make_reranker(name: str, backend: str | None = None) -> Reranker:
    """Build the reranker for ``name``; ``backend`` overrides the env var."""
    choice = (backend or os.environ.get(BACKEND_ENV) or "auto").strip().lower()
    if choice not in BACKENDS:
        raise ValueError(f"{BACKEND_ENV} must be one of {', '.join(BACKENDS)}; got {choice!r}")
    if choice == "sentence-transformers":
        return SentenceTransformersReranker(name)
    if choice == "adaptmem":
        return AdaptMemReranker(name)
    try:
        return AdaptMemReranker(name)
    except RerankerIncompatible as e:
        log.warning("rerank backend: sentence-transformers (adaptmem unusable: %s)", e)
    except RerankerUnavailable as e:
        log.info("rerank backend: sentence-transformers (%s)", e)
    return SentenceTransformersReranker(name)


def env_int(name: str) -> int | None:
    """An optional integer knob from the environment; unset or empty is None."""
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer; got {raw!r}") from e
