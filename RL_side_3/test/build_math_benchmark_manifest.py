#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/home/work/DDAI_revised/verl")
CKPT_ROOT = ROOT / "checkpoints" / "m2-replay-exp"


def checkpoint_entry(name: str, run: str, step: int, tag: str) -> dict:
    hf_path = CKPT_ROOT / run / f"global_step_{step}" / "actor" / "huggingface"
    return {
        "name": name,
        "run": run,
        "step": step,
        "tag": tag,
        "hf_path": str(hf_path),
        "exists": hf_path.exists(),
    }


def dataset_entry(name: str, rel_path: str) -> dict:
    path = ROOT / rel_path
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
    }


def build_manifest() -> dict:
    return {
        "notes": [
            "Recommended shortlist for the math benchmark.",
            "The fair baseline is included twice: once for best-per-run comparison and once for matched-step comparison.",
            "baseline256 step 250 is the most natural held-out peak based on logged MATH-500 validation.",
        ],
        "checkpoints": [
            checkpoint_entry(
                name="baseline256-step250",
                run="grpo-qwen25-math-1.5b-s8-baseline",
                step=250,
                tag="standard_baseline",
            ),
            checkpoint_entry(
                name="replay-step350",
                run="main-grpo-qwen25-math-1.5b-s8-m2-replay-tau0025-b1024-ndeg-plain-quarter",
                step=350,
                tag="main_replay",
            ),
            checkpoint_entry(
                name="replay-step450",
                run="main-grpo-qwen25-math-1.5b-s8-m2-replay-tau0025-b1024-ndeg-plain-quarter",
                step=450,
                tag="main_replay_backup",
            ),
            checkpoint_entry(
                name="fair384-step250",
                run="main-grpo-qwen25-math-1.5b-s8-baseline-fair-b384",
                step=250,
                tag="fair_baseline_best",
            ),
            checkpoint_entry(
                name="fair384-step350",
                run="main-grpo-qwen25-math-1.5b-s8-baseline-fair-b384",
                step=350,
                tag="fair_baseline_matched_phase",
            ),
        ],
        "datasets": [
            dataset_entry("math500", "data/math500/test.parquet"),
            dataset_entry("aime2024x4", "data/aime2024x4/test.parquet"),
            dataset_entry("aime2025x4", "data/aime2025x4/test.parquet"),
            dataset_entry("minerva", "data/minerva/test.parquet"),
            dataset_entry("amc23", "data/amc23/test.parquet"),
            dataset_entry("olympiadbench", "data/olympiadbench/test.parquet"),
        ],
        "generation_defaults": {
            "n": 16,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": 4096,
            "backend": "vllm",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manifest for the RL_side_3 math benchmark.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    args = parser.parse_args()

    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest to {args.output}")


if __name__ == "__main__":
    main()
