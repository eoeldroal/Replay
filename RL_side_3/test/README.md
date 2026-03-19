# RL_side_3 Benchmark Utilities

This directory contains a lightweight benchmark pipeline for the math-side
checkpoints used in `RL_side_3`.

The workflow is intentionally split into two parts:

1. Remote generation on a separate server with available GPUs.
2. Local scoring and aggregation on CPU only.

This separation is deliberate so we never touch the GPUs on the main training
server while other reinforcement-learning jobs are running.

## Files

- `build_math_benchmark_manifest.py`
  - Creates a JSON manifest of recommended math checkpoints and validation
    datasets.
- `run_math_benchmark_vllm_server.sh`
  - Remote-only generation launcher using `verl.trainer.main_generation_server`.
  - This is the recommended vLLM path because it follows verl's tested
    generation-server workflow.
- `generate_math_benchmark_remote.py`
  - Runs generation from a selected checkpoint on a selected dataset.
  - Intended to be run on a different server with GPUs.
- `convert_generation_parquet_to_jsonl.py`
  - Converts the parquet output of `main_generation_server` into the JSONL
    schema expected by the CPU scorer.
- `score_math_benchmark_cpu.py`
  - CPU-only offline scorer for generated JSONL outputs.
  - Reuses `verl.utils.reward_score.default_compute_score`.
- `aggregate_math_benchmark_cpu.py`
  - Combines multiple scored runs into a comparison CSV/JSON.

## Recommended workflow

### 1. Build a manifest locally

```bash
python /home/work/DDAI_revised/verl/RL_side_3/test/build_math_benchmark_manifest.py \
  --output /home/work/DDAI_revised/verl/RL_side_3/test/math_benchmark_manifest.json
```

### 2. Copy to a benchmark server

Copy the following to the benchmark server:

- selected checkpoint `actor/huggingface` folders
- the dataset parquet files
- `generate_math_benchmark_remote.py`
- the generated manifest if useful

### 3A. Run generation remotely through verl's vLLM server path

This is the recommended path when the remote benchmark server has vLLM
installed.

```bash
CHECKPOINT_PATH=/path/to/global_step_350/actor/huggingface \
DATASET_PATH=/path/to/math500/test.parquet \
OUTPUT_PATH=/path/to/outputs/replay_step350_math500.parquet \
NGPUS_PER_NODE=1 \
GEN_TP=1 \
N_SAMPLES=16 \
PROMPT_LENGTH=1536 \
RESPONSE_LENGTH=4096 \
GPU_MEMORY_UTILIZATION=0.85 \
bash /home/work/DDAI_revised/verl/RL_side_3/test/run_math_benchmark_vllm_server.sh
```

Then convert the resulting parquet:

```bash
python /home/work/DDAI_revised/verl/RL_side_3/test/convert_generation_parquet_to_jsonl.py \
  --input-parquet /path/to/outputs/replay_step350_math500.parquet \
  --checkpoint-name replay-step350 \
  --dataset-name math500 \
  --source-dataset-path /path/to/math500/test.parquet \
  --output-jsonl /path/to/outputs/replay_step350_math500.jsonl
```

### 3B. Alternative: direct Python generation remotely

Example:

```bash
python /path/to/generate_math_benchmark_remote.py \
  --checkpoint-path /path/to/global_step_350/actor/huggingface \
  --checkpoint-name replay-step350 \
  --dataset-path /path/to/math500/test.parquet \
  --dataset-name math500 \
  --output /path/to/outputs/replay_step350_math500.jsonl \
  --backend vllm \
  --n 16 \
  --temperature 1.0 \
  --top-p 1.0 \
  --top-k -1 \
  --max-new-tokens 4096
```

Use this path only if you specifically want a direct Python runner instead of
the verl generation server route.

### 4. Bring generated JSONL files back and score them locally on CPU

```bash
python /home/work/DDAI_revised/verl/RL_side_3/test/score_math_benchmark_cpu.py \
  --input /path/to/outputs/replay_step350_math500.jsonl \
  --output-dir /path/to/scored/replay_step350_math500
```

### 5. Aggregate results

```bash
python /home/work/DDAI_revised/verl/RL_side_3/test/aggregate_math_benchmark_cpu.py \
  --inputs /path/to/scored/*/metrics.json \
  --output-csv /path/to/scored/math_benchmark_summary.csv \
  --output-json /path/to/scored/math_benchmark_summary.json
```

## Generation output schema

Each generation JSONL row is expected to look like:

```json
{
  "row_index": 0,
  "checkpoint_name": "replay-step350",
  "dataset_name": "math500",
  "data_source": "HuggingFaceH4/MATH-500",
  "reward_model": {"ground_truth": "...", "style": "rule"},
  "extra_info": {"index": 0, "name": "math", "split": "test"},
  "responses": ["...", "..."]
}
```

The CPU scorer only needs `data_source`, `reward_model`, `extra_info`, and
`responses`.

## Important note about dataset identity

Several math benchmark parquet files in this project share the same
`data_source` value, often `HuggingFaceH4/MATH-500`, even when they correspond
to different benchmark splits such as `math500`, `aime2024x4`, `minerva`, or
`olympiadbench`.

Because of this, benchmark identity in this pipeline is determined by
`dataset_name`, not by `data_source`.

The scripts in this folder enforce that rule:

- generation writes `dataset_name` explicitly into every JSONL row
- scoring groups by `checkpoint_name + dataset_name`
- aggregate preserves per-dataset rows and does not collapse them into a single
  `data_source` bucket

This prevents the previous failure mode where multiple math benchmarks were
accidentally merged under a shared `MATH-500` label.

## vLLM compatibility note

The current local server does not have `vllm` installed, so vLLM execution is
not tested here. Instead, the remote launcher is written to follow verl's own
tested generation-server pattern:

- `tests/special_e2e/generation/run_gen_qwen05_server.sh`
- `verl.trainer.main_generation_server`

This is intentionally more conservative than opening a custom vLLM engine path
from scratch.
