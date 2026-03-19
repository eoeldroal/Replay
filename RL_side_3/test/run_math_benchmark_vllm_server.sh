#!/usr/bin/env bash
set -euo pipefail

# Remote-only benchmark generation launcher.
# This script is intended to run on a separate server with GPUs and vLLM installed.
# It follows verl's tested main_generation_server workflow rather than directly
# instantiating a custom vLLM engine.

ROOT_DIR="${ROOT_DIR:-/home/work/DDAI_revised/verl}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to actor/huggingface directory}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to benchmark parquet}"
OUTPUT_PATH="${OUTPUT_PATH:?Set OUTPUT_PATH to generated parquet path}"

NGPUS_PER_NODE="${NGPUS_PER_NODE:-1}"
NNODES="${NNODES:-1}"
GEN_TP="${GEN_TP:-1}"
N_SAMPLES="${N_SAMPLES:-16}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
PROMPT_LENGTH="${PROMPT_LENGTH:-1536}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

python - <<'PY'
try:
    import vllm
    version = getattr(vllm, "__version__", "unknown")
    print(f"[vllm] detected version: {version}")
except Exception as exc:
    raise SystemExit(f"vLLM is required on the remote benchmark server: {exc}")
PY

# Known-good reference:
# - tests/special_e2e/generation/run_gen_qwen05_server.sh
# - examples/grpo_trainer/run_qwen3-8b.sh (tested on vllm0.8.4 image)

python3 -m verl.trainer.main_generation_server \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    actor_rollout_ref.model.path="${CHECKPOINT_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n="${N_SAMPLES}" \
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${TOP_P}" \
    actor_rollout_ref.rollout.top_k="${TOP_K}" \
    actor_rollout_ref.rollout.prompt_length="${PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    data.train_files="${DATASET_PATH}" \
    data.prompt_key=prompt \
    +data.output_path="${OUTPUT_PATH}" \
    +data.batch_size="${BATCH_SIZE}"
