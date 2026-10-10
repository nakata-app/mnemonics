#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

EVREN = "https://evren-llmapi.ssyz.org.tr/v1"


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--v2-repo", required=True)
    p.add_argument("--contexts", required=True)
    p.add_argument("--reader", default="qwen3.8-flash-next")
    p.add_argument("--judge", default="glm-5.3")
    p.add_argument("--provider", choices=["evren", "nim"], default="evren")
    args, rest = p.parse_known_args()

    by_q: dict[str, list[dict]] = {}
    for line in open(args.contexts):
        r = json.loads(line)
        by_q[r["question_id"]] = r["context"]
    questions = {}
    qpath = rest[rest.index("--questions-path") + 1]
    for line in open(qpath):
        x = json.loads(line)
        q = x.get("question")
        questions[(q.get("text") if isinstance(q, dict) else q) or ""] = str(x["id"])

    sys.path.insert(0, str(Path(args.v2_repo).resolve()))
    from memory_modules.memory import Memory, register_memory

    @register_memory
    class ReplayMemory(Memory):
        memory_type = "replay"

        def insert(self, trajectory: dict[str, object]) -> None:
            pass

        def query(self, query: str, query_image: str | None = None):
            return by_q[questions[query]]

    cfg = Path(os.environ.get("TMPDIR", "/tmp")) / "replay_memory_config.json"
    cfg.write_text(json.dumps({"memory_type": "replay", "memory_params": {}}))
    if args.provider == "evren":
        os.environ["EVREN_V2_KEY"] = json.loads((Path.home() / ".svelka/keys.json").read_text())["providers"]["evren"][0]
        base, key_env = EVREN, "EVREN_V2_KEY"
    else:
        base, key_env = "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"
    sys.argv = [str(Path(args.v2_repo) / "evaluation" / "harness.py"), "--memory-config-path", str(cfg),
                "--model", args.reader, "--base-url", base, "--api-key-env", key_env,
                "--evaluator-model", args.judge, "--evaluator-base-url", base, "--evaluator-api-key-env", key_env,
                *rest]
    from evaluation.harness import main as upstream_main
    upstream_main()


if __name__ == "__main__":
    main()
