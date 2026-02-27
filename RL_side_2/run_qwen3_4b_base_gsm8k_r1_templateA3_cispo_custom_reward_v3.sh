#!/usr/bin/env bash

# CISPO + DeepSeek-R1-style 3-head custom reward on GSM8K R1 TemplateA v3.
# New script only; existing scripts are left untouched.

set -x

PROJECT_DIR=/home/work/DDAI_revised/verl
REWARD_FN=${PROJECT_DIR}/RL_side_2/r1_deepseek_reward.py

R1_TRAIN_DATA=${R1_TRAIN_DATA:-${PROJECT_DIR}/data/gsm8k/train_r1_templateA_v3.parquet}
R1_VAL_DATA=${R1_VAL_DATA:-${PROJECT_DIR}/data/gsm8k/test_r1_templateA_v3.parquet}

export CUDA_VISIBLE_DEVICES=0,1,3
export MAX_RESPONSE_LENGTH=16384

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${R1_TRAIN_DATA} \
    data.val_files=${R1_VAL_DATA} \
    data.train_batch_size=48 \
    data.max_prompt_length=512 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${PROJECT_DIR}/data/models/Qwen3-4B-Base \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=12 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.policy_loss.loss_mode=cispo \
    actor_rollout_ref.actor.clip_ratio_low=10 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    +actor_rollout_ref.actor.policy_diag_dir=/home/work/DDAI_revised/verl/logs/CISPO_qwen3_4b_base_gsm8k_r1tmplA3_customreward_v3_diag \
    +actor_rollout_ref.actor.policy_diag_flush_freq=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len=$((MAX_RESPONSE_LENGTH + 1024)) \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=naive \
    custom_reward_function.path=$REWARD_FN \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.w_answer=1.0 \
    +custom_reward_function.reward_kwargs.w_format=0.1 \
    +custom_reward_function.reward_kwargs.w_length=1.0 \
    +custom_reward_function.reward_kwargs.think_min_chars=32 \
    +custom_reward_function.reward_kwargs.require_format_for_answer=True \
    +custom_reward_function.reward_kwargs.max_len=16384 \
    +custom_reward_function.reward_kwargs.pass_len=15360 \
    +custom_reward_function.reward_kwargs.length_fallback_mode=word_count \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=Exp.Custom.Algo \
    trainer.experiment_name=CISPO_qwen3_4b_base_gsm8k_r1tmplA3_customreward_v3 \
    trainer.rollout_data_dir=/home/work/DDAI_revised/verl/logs/CISPO_qwen3_4b_base_gsm8k_r1tmplA3_customreward_v3_rollout \
    trainer.validation_data_dir=/home/work/DDAI_revised/verl/logs/CISPO_qwen3_4b_base_gsm8k_r1tmplA3_customreward_v3_validation \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.total_epochs=5 "$@"
