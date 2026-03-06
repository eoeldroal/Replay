#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

project_name=m2-replay-exp
experiment_name=grpo-qwen25-math-1.5b-s8-m2-replay-rev2

deepscaler_preview_train_path="$ROOT_DIR/data/deepscaler_preview/train.parquet"
train_files="['$deepscaler_preview_train_path']"

math500_test_path="$ROOT_DIR/data/math500/test.parquet"
aime2024_test_path="$ROOT_DIR/data/aime2024x4/test.parquet"
aime2025_test_path="$ROOT_DIR/data/aime2025x4/test.parquet"
minerva_test_path="$ROOT_DIR/data/minerva/test.parquet"
amc23_test_path="$ROOT_DIR/data/amc23/test.parquet"
olympiadbench_test_path="$ROOT_DIR/data/olympiadbench/test.parquet"
test_files="['$math500_test_path', '$aime2024_test_path', '$aime2025_test_path', '$minerva_test_path', '$amc23_test_path', '$olympiadbench_test_path']"

model_path="$ROOT_DIR/data/models/Qwen2.5-Math-1.5B"
log_dir="$ROOT_DIR/data-log/$project_name/$experiment_name"
mkdir -p "$log_dir"

export CUDA_VISIBLE_DEVICES=4,5,6,7

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=1536 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.clip_ratio_c=100000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=100000 \
    actor_rollout_ref.rollout.max_model_len=5632 \
    actor_rollout_ref.rollout.max_num_seqs=4096 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1 \
    algorithm.use_kl_in_reward=False \
    +algorithm.m2_replay.enable=true \
    +algorithm.m2_replay.tau=0.001 \
    +algorithm.m2_replay.buffer.max_query_groups=1024 \
    +algorithm.m2_replay.schedule.replay_target_groups=128 \
    +algorithm.m2_replay.selection.mode=recency_only \
    +algorithm.m2_replay.selection.ingress_filter_mode=rlvr_halfband \
    +algorithm.m2_replay.schedule.floor_to_micro_multiple=true \
    +algorithm.m2_replay.logging.prefix=m2_replay \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=30 \
    trainer.test_freq=50 \
    trainer.total_training_steps=1201 \
    trainer.total_epochs=100 2>&1 | tee "$log_dir/$experiment_name.log"
