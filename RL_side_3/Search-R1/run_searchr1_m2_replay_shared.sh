#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SEARCHR1_DIR="$SCRIPT_DIR"

project_name=search-r1-m2-replay-exp
experiment_name="${EXPERIMENT_NAME:-search-r1-qwen25-3b-m2-replay-revised-10-quarter}"
start_mode="${START_MODE:-quarter}"

TRAIN_DATA="${TRAIN_DATA:-$ROOT_DIR/data/searchr1_processed_direct/train.parquet}"
VAL_DATA="${VAL_DATA:-$ROOT_DIR/data/searchr1_processed_direct/test.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
TOOL_CONFIG="$SEARCHR1_DIR/tool_config/search_r1_qwen_text_tool_config.yaml"

LOG_DIR="$ROOT_DIR/data-log/$project_name/$experiment_name"
VAL_DUMP_DIR="$LOG_DIR/validation_jsonl"
ROLLOUT_DUMP_DIR="$LOG_DIR/rollout_jsonl"
mkdir -p "$LOG_DIR" "$VAL_DUMP_DIR" "$ROLLOUT_DUMP_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$SEARCHR1_DIR${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT_DIR"

python3 -u -m verl.trainer.main_ppo \
  --config-path="$ROOT_DIR/examples/sglang_multiturn/config" \
  --config-name='search_multiturn_grpo' \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.train_batch_size=128 \
  data.max_prompt_length=512 \
  data.max_response_length=16384 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.return_raw_chat=True \
  data.filter_overlong_prompts_workers=64 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
  actor_rollout_ref.rollout.multi_turn.format=qwen \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  +algorithm.m2_replay.enable=true \
  +algorithm.m2_replay.tau=0.0025 \
  +algorithm.m2_replay.buffer.max_query_groups=512 \
  +algorithm.m2_replay.schedule.replay_target_groups=64 \
  +algorithm.m2_replay.schedule.start_mode="$start_mode" \
  +algorithm.m2_replay.selection.mode=zvp_recency \
  +algorithm.m2_replay.selection.zvp_use_recency=false \
  +algorithm.m2_replay.selection.zvp_ema_alpha=1.0 \
  +algorithm.m2_replay.selection.logprob_groups_per_chunk=64 \
  +algorithm.m2_replay.selection.log_prob_micro_batch_size_per_gpu=4 \
  +algorithm.m2_replay.selection.log_prob_use_dynamic_bsz=true \
  +algorithm.m2_replay.selection.log_prob_max_token_len_per_gpu=70000 \
  +algorithm.m2_replay.selection.gpu_m2_fastpath=true \
  +algorithm.m2_replay.selection.compute_runtime_zvp=false \
  +algorithm.m2_replay.selection.ingress_filter_mode=rlvr_non_degenerate \
  +algorithm.m2_replay.selection.zvp_mode=sign_only \
  +algorithm.m2_replay.schedule.floor_to_micro_multiple=true \
  +algorithm.m2_replay.logging.prefix=m2_replay \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="$project_name" \
  trainer.experiment_name="$experiment_name" \
  trainer.validation_data_dir="$VAL_DUMP_DIR" \
  trainer.rollout_data_dir="$ROLLOUT_DUMP_DIR" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.use_legacy_worker_impl=enable \
  trainer.val_before_train=False \
  trainer.save_freq=10 \
  trainer.test_freq=50 \
  trainer.total_training_steps=500 \
  trainer.total_epochs=1 "$@" 2>&1 | tee "$LOG_DIR/$experiment_name.log"
