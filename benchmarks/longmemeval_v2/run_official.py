#!/usr/bin/env python3
"""Run official LongMemEval-V2 harness with a frozen Mnemonics backend.

No benchmark answers or question ids are consumed by the backend. This launcher
only registers an adapter with the upstream Memory registry and then delegates
all prompt building, reader calls, latency accounting and scoring to upstream.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_launcher_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--v2-repo", required=True)
    p.add_argument("--mnemonics-top-k", type=int, default=16)
    p.add_argument("--mnemonics-candidate-k", type=int, default=64)
    p.add_argument("--mnemonics-max-per-trajectory", type=int, default=4)
    p.add_argument("--mnemonics-no-rerank", action="store_true")
    return p.parse_known_args()


def main() -> None:
    args, rest = parse_launcher_args()
    v2_repo = Path(args.v2_repo).expanduser().resolve()
    if not (v2_repo / "evaluation" / "harness.py").exists():
        raise SystemExit(f"Invalid LongMemEval-V2 repo: {v2_repo}")
    sys.path.insert(0, str(v2_repo))

    from memory_modules.memory import Memory, register_memory
    from benchmarks.longmemeval_v2.mnemonics_memory import MnemonicsV2Backend

    @register_memory
    class MnemonicsMemory(Memory):
        memory_type = "mnemonics"

        def __init__(self, memory_params: dict[str, object]) -> None:
            super().__init__(memory_params)
            merged = dict(memory_params)
            merged.setdefault("top_k", args.mnemonics_top_k)
            merged.setdefault("candidate_k", args.mnemonics_candidate_k)
            merged.setdefault("max_per_trajectory", args.mnemonics_max_per_trajectory)
            merged.setdefault("rerank", not args.mnemonics_no_rerank)
            self.backend = MnemonicsV2Backend(merged)

        def insert(self, trajectory: dict[str, object]) -> None:
            self.backend.insert(trajectory)

        def query(self, query: str, query_image: str | None = None):
            # Frozen first pass is text-only; query_image is intentionally not inspected.
            return [{"type": "text", "value": t} for t in self.backend.query_texts(query)]

    # Upstream harness requires a config file. Generate a minimal temporary one
    # only when the caller did not supply --memory-config-path.
    if "--memory-config-path" not in rest:
        cfg = Path(os.environ.get("MNEMONICS_V2_CONFIG", "/tmp/mnemonics_v2_memory_config.json"))
        cfg.write_text(json.dumps({
            "memory_type": "mnemonics",
            "memory_params": {
                "top_k": args.mnemonics_top_k,
                "candidate_k": args.mnemonics_candidate_k,
                "max_per_trajectory": args.mnemonics_max_per_trajectory,
                "rerank": not args.mnemonics_no_rerank,
            },
        }, indent=2) + "\n")
        rest = ["--memory-config-path", str(cfg), *rest]

    sys.argv = [str(v2_repo / "evaluation" / "harness.py"), *rest]
    from evaluation.harness import main as upstream_main
    upstream_main()


if __name__ == "__main__":
    main()
