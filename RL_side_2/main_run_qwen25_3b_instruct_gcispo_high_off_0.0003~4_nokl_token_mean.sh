#!/usr/bin/env bash
# GCISPO training with Qwen2.5-3B-Instruct on DAPO-Math-17k-Processed (full dataset).
#
# Comparable baseline: CISPO_qwen25_3b_instruct_3_high_off_policy_v2
#
# Algorithm-specific settings (DO NOT CHANGE for fair comparison):
#   loss_mode=gcispo, clip_ratio_low=0.0003, clip_ratio_high=0.0004
#   loss_agg_mode=token-mean
#
# Differences vs CISPO baseline:
#   - loss_mode: gcispo (vs cispo)
#   - clip_ratio_high: 0.0004 (vs 0.2)
#   - use_kl_loss: False (vs True with coef=0.001)
#
# Reward config (weight-based, same effective values as CISPO baseline):
#   - answer_score_cfg: base(0/1) * weight=1.0
#   - overlong_buffer_cfg: base(0/1) * weight=1.0, buffer_len=7168

set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MAX_RESPONSE_LENGTH=8192

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=/home/work/DDAI_revised/verl/data/dapo-math-17k-processed/en_verl.parquet \
    data.val_files=/home/work/DDAI_revised/verl/data/aime-2024-repo/data/aime-2024.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/home/work/DDAI_revised/verl/data/models/Qwen2.5-3B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gcispo \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.clip_ratio_low=0.0003 \
    actor_rollout_ref.actor.clip_ratio_high=0.0004 \
    +actor_rollout_ref.actor.policy_diag_dir=/home/work/DDAI_revised/verl/logs/GCISPO_qwen25_3b_instruct_diag_6_high_off_0.0003_0.0004_nokl_token_mean \
    +actor_rollout_ref.actor.policy_diag_flush_freq=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len=$((MAX_RESPONSE_LENGTH + 1024)) \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.trace.backend=weave \
    actor_rollout_ref.rollout.trace.token2text=true \
    actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.answer_score_cfg.enable=True \
    +reward_model.reward_kwargs.answer_score_cfg.weight=1.0 \
    +reward_model.reward_kwargs.answer_score_cfg.log=True \
    +reward_model.reward_kwargs.max_resp_len=$MAX_RESPONSE_LENGTH \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=$((MAX_RESPONSE_LENGTH - 1024)) \
    +reward_model.reward_kwargs.overlong_buffer_cfg.weight=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='Exp.Custom.Algo' \
    trainer.experiment_name='GCISPO_qwen25_3b_instruct_6_high_off_0.0003_0.0004_nokl_token_mean' \
    trainer.rollout_data_dir='/home/work/DDAI_revised/verl/logs/GCISPO_qwen25_3b_instruct_rollout_6_high_off_0.0003_0.0004_nokl_token_mean' \
    trainer.validation_data_dir='/home/work/DDAI_revised/verl/logs/GCISPO_qwen25_3b_instruct_validation_6_high_off_0.0003_0.0004_nokl_token_mean' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.total_epochs=10 \
    "$@"
