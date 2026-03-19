#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CPU-scored benchmark metrics.")
    parser.add_argument("--inputs", nargs="+", required=True, help="metrics.json files from score_math_benchmark_cpu.py")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--sort-key", default="best@16", help="Metric key used for sorting rows.")
    args = parser.parse_args()

    rows = []
    for raw_path in args.inputs:
        path = Path(raw_path)
        payload = json.loads(path.read_text())
        rows.extend(payload)

    rows.sort(
        key=lambda row: (
            row.get("checkpoint_name", ""),
            row.get("dataset_name") == "__overall__",
            -(row.get(args.sort_key, -1)),
        )
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2))

    if rows:
        keys = sorted({k for row in rows for k in row.keys()})
        with args.output_csv.open("w") as f:
            f.write(",".join(keys) + "\n")
            for row in rows:
                vals = []
                for key in keys:
                    val = row.get(key, "")
                    vals.append(str(val))
                f.write(",".join(vals) + "\n")

    print(f"Wrote aggregate JSON to {args.output_json}")
    print(f"Wrote aggregate CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
