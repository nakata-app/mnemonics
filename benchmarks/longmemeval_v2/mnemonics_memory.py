from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from mnemonics.ingest import _get_encoder, ingest
from mnemonics.retrieve import retrieve
from mnemonics.store import Store


def _words(text: str) -> list[str]:
    return str(text or "").split()


def _window(text: str, size: int = 180, overlap: int = 30) -> list[str]:
    words = _words(text)
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    out: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        part = words[start:start + size]
        if not part:
            break
        out.append(" ".join(part))
        if start + size >= len(words):
            break
    return out


def trajectory_chunks(trajectory: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Convert a V2 trajectory into label-free text chunks.

    This uses only trajectory content available to every memory backend. It does
    not inspect question ids, answers, eval functions, or gold evidence.
    """
    tid = str(trajectory.get("id") or "")
    domain = str(trajectory.get("domain") or "")
    environment = str(trajectory.get("environment") or "")
    goal = str(trajectory.get("goal") or "")
    outcome = str(trajectory.get("outcome") or "")
    start_url = str(trajectory.get("start_url") or "")

    texts: list[str] = []
    meta: list[dict[str, Any]] = []

    trajectory_header = (
        f"TRAJECTORY={tid}\nDOMAIN={domain}\nENVIRONMENT={environment}\n"
        f"GOAL={goal}\nOUTCOME={outcome}\nSTART_URL={start_url}"
    )
    texts.append(trajectory_header)
    meta.append({"trajectory_id": tid, "kind": "trajectory", "domain": domain, "environment": environment})

    states = trajectory.get("states") or []
    if not isinstance(states, list):
        states = []
    for state in states:
        if not isinstance(state, dict):
            continue
        state_index = state.get("state_index")
        step = state.get("step")
        url = str(state.get("url") or "")
        action = str(state.get("action") or "")
        thought = str(state.get("thought") or "")
        tree = str(state.get("accessibility_tree") or "")
        screenshot = str(state.get("screenshot") or "")
        prefix = (
            f"TRAJECTORY={tid} STATE={state_index} STEP={step}\n"
            f"GOAL={goal}\nOUTCOME={outcome}\nURL={url}\n"
            f"ACTION={action}\nTHOUGHT={thought}\nSCREENSHOT={screenshot}"
        )
        # Keep an action/thought state summary independently searchable.
        texts.append(prefix)
        meta.append({
            "trajectory_id": tid,
            "kind": "state",
            "state_index": state_index,
            "step": step,
            "screenshot": screenshot,
            "domain": domain,
        })
        # Accessibility trees are large; pre-window them and repeat provenance
        # so every embedded chunk remains self-contained.
        for wi, window in enumerate(_window(tree)):
            texts.append(f"{prefix}\nACCESSIBILITY_TREE_WINDOW={wi}\n{window}")
            meta.append({
                "trajectory_id": tid,
                "kind": "accessibility_tree",
                "state_index": state_index,
                "step": step,
                "window": wi,
                "screenshot": screenshot,
                "domain": domain,
            })
    return texts, meta


class MnemonicsV2Backend:
    """Backend logic shared by the official LongMemEval-V2 Memory wrapper."""

    def __init__(self, memory_params: dict[str, object] | None = None) -> None:
        p = dict(memory_params or {})
        self.top_k = int(p.get("top_k", 16))
        self.candidate_k = int(p.get("candidate_k", 64))
        self.rerank = bool(p.get("rerank", True))
        self.max_per_trajectory = int(p.get("max_per_trajectory", 4))
        self.ns = str(p.get("namespace", "lme_v2"))
        self.model = str(p.get("model", "all-MiniLM-L6-v2"))
        store_dir = p.get("store_dir")
        if store_dir:
            self.store_dir = Path(str(store_dir)).expanduser().resolve()
            self.store_dir.mkdir(parents=True, exist_ok=True)
            self._tmp = None
        else:
            self._tmp = tempfile.TemporaryDirectory(prefix="mnemonics-lme-v2-")
            self.store_dir = Path(self._tmp.name)
        enc = _get_encoder(self.model)
        self.store = Store(self.store_dir, dim=int(enc.get_sentence_embedding_dimension()))
        self.inserted_trajectory_ids: set[str] = set()

    def insert(self, trajectory: dict[str, object]) -> None:
        tid = str(trajectory.get("id") or "")
        if tid and tid in self.inserted_trajectory_ids:
            return
        texts, meta = trajectory_chunks(trajectory)
        if texts:
            # Already pre-windowed: avoid a second generic chunking pass.
            ingest(
                texts=texts,
                meta=meta,
                store=self.store,
                ns=self.ns,
                model=self.model,
                chunk_size=999999,
                chunk_overlap=0,
                augment_preferences=False,
                augment_assistant_facts=False,
            )
        if tid:
            self.inserted_trajectory_ids.add(tid)

    def query_texts(self, query: str) -> list[str]:
        result = retrieve(
            query=query,
            store=self.store,
            ns=self.ns,
            top_k=max(self.top_k * 2, self.top_k),
            candidate_k=max(self.candidate_k, self.top_k * 2),
            model=self.model,
            decay=False,
            hybrid=True,
            rerank=self.rerank,
            boost_signals=True,
            touch=False,
        )
        selected: list[str] = []
        counts: dict[str, int] = {}
        for row in result.get("results", []):
            meta = row.get("meta") if isinstance(row, dict) else None
            tid = str(meta.get("trajectory_id") or "") if isinstance(meta, dict) else ""
            if tid and counts.get(tid, 0) >= self.max_per_trajectory:
                continue
            text = str(row.get("text") or "")
            if not text:
                continue
            selected.append(text)
            if tid:
                counts[tid] = counts.get(tid, 0) + 1
            if len(selected) >= self.top_k:
                break
        return selected
