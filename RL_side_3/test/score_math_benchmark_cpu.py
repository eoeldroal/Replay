#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path("/home/work/DDAI_revised/verl")
REWARD_ROOT = ROOT / "verl" / "utils" / "reward_score"


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE_CACHE: dict[str, Any] = {}


def get_module(name: str):
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]
    mapping = {
        "math_reward": REWARD_ROOT / "math_reward.py",
        "math_dapo": REWARD_ROOT / "math_dapo.py",
        "geo3k": REWARD_ROOT / "geo3k.py",
        "search_r1": REWARD_ROOT / "search_r1_like_qa_em.py",
    }
    module = _load_module(f"{name}_local", mapping[name])
    _MODULE_CACHE[name] = module
    return module


def parse_k_values(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def default_compute_score_local(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
) -> float:
    extra_info = extra_info or {}

    if data_source in {"openai/gsm8k", "lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"}:
        return float(get_module("math_reward").compute_score(solution_str, ground_truth))
    if data_source in {"math_dapo", "math", "math_dapo_reasoning"} or str(data_source).startswith("aime"):
        return float(get_module("math_dapo").compute_score(solution_str, ground_truth))
    if data_source == "hiyouga/geometry3k":
        return float(get_module("geo3k").compute_score(solution_str, ground_truth))
    if data_source in {
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    }:
        return float(get_module("search_r1").compute_score(solution_str, ground_truth))
    raise NotImplementedError(f"Unsupported data_source for CPU scorer: {data_source}")


def score_responses(data_source: str, responses: list[str], reward_model: dict, extra_info: dict) -> list[float]:
    gt = reward_model["ground_truth"]
    return [default_compute_score_local(data_source, resp, gt, extra_info=extra_info) for resp in responses]


def prefix_metrics(scores: list[float], ks: list[int]) -> dict[str, float]:
    metrics = {}
    binary = all(abs(s - round(s)) < 1e-9 and s in (0.0, 1.0) for s in scores)
    for k in ks:
        prefix = scores[: min(k, len(scores))]
        if not prefix:
            continue
        metrics[f"mean@{k}"] = sum(prefix) / len(prefix)
        metrics[f"best@{k}"] = max(prefix)
        if binary:
            metrics[f"pass@{k}"] = 1.0 if max(prefix) > 0 else 0.0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only scorer for RL_side_3 benchmark generations.")
    parser.add_argument("--input", type=Path, required=True, help="Generated JSONL file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write scored outputs.")
    parser.add_argument("--k-values", default="1,2,4,8,16")
    args = parser.parse_args()

    ks = parse_k_values(args.k_values)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_prompt_path = args.output_dir / "per_prompt_scores.jsonl"
    metrics_json_path = args.output_dir / "metrics.json"
    metrics_csv_path = args.output_dir / "metrics.csv"

    grouped = defaultdict(list)
    total_rows = []

    with args.input.open() as f_in, per_prompt_path.open("w") as f_out:
        for line in f_in:
            row = json.loads(line)
            dataset_name = row.get("dataset_name")
            if not dataset_name:
                raise ValueError(
                    "Each generated row must contain a non-empty 'dataset_name'. "
                    "Do not rely on 'data_source' for benchmark identity."
                )
            scores = score_responses(
                data_source=row["data_source"],
                responses=row["responses"],
                reward_model=row["reward_model"],
                extra_info=row.get("extra_info", {}),
            )
            metrics = prefix_metrics(scores, ks)
            prompt_result = {
                "row_index": row["row_index"],
                "checkpoint_name": row["checkpoint_name"],
                "dataset_name": dataset_name,
                "source_dataset_path": row.get("source_dataset_path", ""),
                "data_source": row["data_source"],
                "num_responses": len(row["responses"]),
                "scores": scores,
                "metrics": metrics,
            }
            f_out.write(json.dumps(prompt_result, ensure_ascii=False) + "\n")
            total_rows.append(prompt_result)
            grouped[(row["checkpoint_name"], dataset_name)].append(prompt_result)

    summary_rows = []
    for (checkpoint_name, dataset_name), rows in grouped.items():
        merged = defaultdict(list)
        scoring_sources = sorted({row["data_source"] for row in rows})
        source_paths = sorted({row.get("source_dataset_path", "") for row in rows if row.get("source_dataset_path", "")})
        for row in rows:
            for key, value in row["metrics"].items():
                merged[key].append(value)
        summary = {
            "checkpoint_name": checkpoint_name,
            "dataset_name": dataset_name,
            "num_prompts": len(rows),
            "scoring_data_source": "|".join(scoring_sources),
            "source_dataset_path": "|".join(source_paths),
        }
        for key, values in merged.items():
            summary[key] = sum(values) / len(values)
        summary_rows.append(summary)

    if total_rows:
        merged = defaultdict(list)
        checkpoint_name = total_rows[0]["checkpoint_name"]
        scoring_sources = sorted({row["data_source"] for row in total_rows})
        for row in total_rows:
            for key, value in row["metrics"].items():
                merged[key].append(value)
        overall = {
            "checkpoint_name": checkpoint_name,
            "dataset_name": "__overall__",
            "num_prompts": len(total_rows),
            "scoring_data_source": "|".join(scoring_sources),
            "source_dataset_path": "__multiple__",
        }
        for key, values in merged.items():
            overall[key] = sum(values) / len(values)
        summary_rows.append(overall)

    metrics_json_path.write_text(json.dumps(summary_rows, indent=2))

    if summary_rows:
        keys = sorted({k for row in summary_rows for k in row.keys()})
        with metrics_csv_path.open("w") as f:
            f.write(",".join(keys) + "\n")
            for row in summary_rows:
                vals = []
                for key in keys:
                    val = row.get(key, "")
                    if isinstance(val, str):
                        vals.append(val)
                    else:
                        vals.append(str(val))
                f.write(",".join(vals) + "\n")

    print(f"Wrote per-prompt scores to {per_prompt_path}")
    print(f"Wrote metrics JSON to {metrics_json_path}")
    print(f"Wrote metrics CSV to {metrics_csv_path}")


if __name__ == "__main__":
    main()
