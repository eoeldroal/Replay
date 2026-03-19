#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_prompt(prompt_obj: Any) -> list[dict[str, Any]]:
    if hasattr(prompt_obj, "tolist"):
        prompt_obj = prompt_obj.tolist()
    if isinstance(prompt_obj, dict):
        return [prompt_obj]
    if isinstance(prompt_obj, list):
        return prompt_obj
    raise TypeError(f"Unsupported prompt type: {type(prompt_obj).__name__}")


def render_prompts(df: pd.DataFrame, tokenizer) -> list[str]:
    rendered = []
    for prompt_obj in df["prompt"]:
        messages = normalize_prompt(prompt_obj)
        rendered.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    return rendered


def run_vllm(args, prompts: list[str]) -> list[list[str]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.checkpoint_path),
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
    )
    outputs = llm.generate(prompts, params, use_tqdm=True)
    outputs = sorted(outputs, key=lambda x: x.request_id)
    return [[candidate.text for candidate in item.outputs] for item in outputs]


def run_transformers(args, prompts: list[str]) -> list[list[str]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    all_responses = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        responses = []
        for _ in range(args.n):
            output_ids = model.generate(
                **encoded,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
            )
            gen_ids = output_ids[0][encoded["input_ids"].shape[1] :]
            responses.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
        all_responses.append(responses)
    return all_responses


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark outputs from a checkpoint on a remote GPU server.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    df = pd.read_parquet(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    prompts = render_prompts(df, tokenizer)

    if args.backend == "vllm":
        grouped_responses = run_vllm(args, prompts)
    else:
        grouped_responses = run_transformers(args, prompts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for idx, (_, row) in enumerate(df.iterrows()):
            payload = {
                "row_index": idx,
                "checkpoint_name": args.checkpoint_name,
                "dataset_name": args.dataset_name,
                "source_dataset_path": str(args.dataset_path),
                "data_source": row["data_source"],
                "prompt": normalize_prompt(row["prompt"]),
                "reward_model": row["reward_model"],
                "extra_info": row.get("extra_info", {}),
                "responses": grouped_responses[idx],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"Wrote generations to {args.output}")


if __name__ == "__main__":
    main()
