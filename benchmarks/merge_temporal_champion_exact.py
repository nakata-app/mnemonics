"""Merge two exact ca53594 Kaggle candidate halves and run temporal ablation.

This script performs validation/post-processing only. It never loads embedding
or reranker models and never executes benchmark retrieval on the Mac.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {name} under {root}, found {matches}")
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--half0",
        type=Path,
        default=Path("/Users/macmini/krun-output/mnemonics_temporal_champion_h0"),
    )
    ap.add_argument(
        "--half1",
        type=Path,
        default=Path("/Users/macmini/krun-output/mnemonics_temporal_champion_h1"),
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("/Users/macmini/krun-output/lme-data/longmemeval_s_cleaned.json"),
    )
    ap.add_argument(
        "--ablation-script",
        type=Path,
        default=Path(
            "/Users/macmini/Projects/mnemonics-pr2-temp-rebase/"
            "benchmarks/temporal_ablation_offline.py"
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/Users/macmini/krun-output/mnemonics_temporal_champion_exact_final"
        ),
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    half_rows = []
    provenance = []
    for idx, root in enumerate((args.half0, args.half1)):
        cands = find_one(root, f"cands_ca53594_half{idx}.json")
        prov_file = find_one(root, f"provenance_half{idx}.json")
        prov = json.loads(prov_file.read_text())
        rows = json.loads(cands.read_text())
        if prov["candidate_source_commit"] != "ca53594":
            raise SystemExit(f"half {idx}: wrong source commit")
        if prov["outer_half"] != idx:
            raise SystemExit(f"half {idx}: wrong outer_half")
        if prov["question_count"] != 250 or len(rows) != 250:
            raise SystemExit(f"half {idx}: expected 250 rows")
        actual_sha = sha256(cands)
        if actual_sha != prov["candidate_sha256"]:
            raise SystemExit(
                f"half {idx}: candidate sha mismatch {actual_sha} != "
                f"{prov['candidate_sha256']}"
            )
        half_rows.append(rows)
        provenance.append(prov)

    rows = half_rows[0] + half_rows[1]
    if len(rows) != 500:
        raise SystemExit(f"merged candidate count {len(rows)} != 500")

    dataset = json.loads(args.data.read_text())
    if len(dataset) != 500:
        raise SystemExit(f"dataset count {len(dataset)} != 500")
    expected_ids = [q["question_id"] for q in dataset]
    actual_ids = [r["qid"] for r in rows]
    if actual_ids != expected_ids:
        raise SystemExit("merged candidate QID order does not match dataset")
    if len(set(actual_ids)) != 500:
        raise SystemExit("merged candidate QIDs are not unique")

    merged = args.out_dir / "cands_ca53594_500.json"
    merged.write_text(json.dumps(rows))
    ablation = args.out_dir / "ablation_ca53594_500.json"

    subprocess.run(
        [
            sys.executable,
            str(args.ablation_script),
            "--candidates",
            str(merged),
            "--data",
            str(args.data),
            "--out",
            str(ablation),
            "--expected-n",
            "500",
        ],
        check=True,
    )

    result = json.loads(ablation.read_text())
    on_labeled = result["variants"]["on_labeled"]["R@1"]
    final_prov = {
        "candidate_source_commit": "ca53594",
        "ablation_code_commit": "120210a",
        "candidate_sha256": sha256(merged),
        "candidate_count": 500,
        "half_provenance": provenance,
        "reproduced_champion_r1": on_labeled,
    }
    result["provenance"] = final_prov
    ablation.write_text(json.dumps(result, indent=2))
    (args.out_dir / "provenance_ca53594_500.json").write_text(
        json.dumps(final_prov, indent=2)
    )

    if abs(on_labeled - 0.958) > 1e-12:
        raise SystemExit(
            f"REJECT: on_labeled R@1={on_labeled} does not reproduce 0.958"
        )

    print("HISTORICAL_CHAMPION_REPRO_OK")
    for variant in ("off", "on_labeled", "on_no_label"):
        row = result["variants"][variant]
        print(
            variant,
            f"R@1={row['R@1']:.4f}",
            f"R@5={row['R@5']:.4f}",
            f"R@10={row['R@10']:.4f}",
        )
    for name, row in result["paired_R@1"].items():
        print(
            name,
            f"delta={row['delta']:+.4f}",
            f"ci95={row['ci95']}",
            f"helped={row['helped']}",
            f"hurt={row['hurt']}",
        )


if __name__ == "__main__":
    main()
