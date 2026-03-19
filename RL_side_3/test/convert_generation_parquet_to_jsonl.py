#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def to_jsonable(value: Any):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert verl main_generation_server parquet output to benchmark JSONL.")
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--source-dataset-path", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.input_parquet)
    required = {"data_source", "prompt", "reward_model", "responses"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input parquet missing required columns: {sorted(missing)}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as f:
        for idx, (_, row) in enumerate(df.iterrows()):
            payload = {
                "row_index": idx,
                "checkpoint_name": args.checkpoint_name,
                "dataset_name": args.dataset_name,
                "source_dataset_path": args.source_dataset_path,
                "data_source": row["data_source"],
                "prompt": to_jsonable(row["prompt"]),
                "reward_model": to_jsonable(row["reward_model"]),
                "extra_info": to_jsonable(row.get("extra_info", {})),
                "responses": to_jsonable(row["responses"]),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"Wrote benchmark JSONL to {args.output_jsonl}")


if __name__ == "__main__":
    main()
