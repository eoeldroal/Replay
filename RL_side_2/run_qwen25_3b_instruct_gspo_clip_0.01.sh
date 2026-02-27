#!/usr/bin/env bash
# GSPO training with Qwen2.5-3B-Instruct on DAPO-Math-17k-Processed (competition-level math)
# Based on run_qwen3-8b_4_gspo.sh, adapted to the current Qwen2.5-3B-Instruct setup.
# NOTE: Uses 25% random subset (3,529 of 14,116) from the English partition (en_verl).
#       Sampled with random_state=42 for reproducibility. See en_verl_25pct.parquet.
#
# GSPO = sequence-level IS ratio with tight clipping.
# This variant matches clip bounds with GCISPO for a fair clip-controlled comparison.
#
# ┌───────────────────┬──────────────┬──────────────┬──────────────┬──────────────────┐
# │ Parameter         │ GSPO         │ CISPO        │ GCISPO       │ Note             │
# ├───────────────────┼──────────────┼──────────────┼──────────────┼──────────────────┤
# │ loss_mode         │ gspo         │              │              │                  │
# │ loss_agg_mode     │ seq-mean-..  │              │              │                  │
# │ clip_ratio_low    │ 0.01         │              │              │ matched to GCISPO│
# │ clip_ratio_high   │ 0.01         │              │              │ matched to GCISPO│
# │ clip_ratio_c      │ 10.0         │              │              │ GSPO default     │
# │ use_kl_loss       │ False        │              │              │ KL removed       │
# └───────────────────┴──────────────┴──────────────┴──────────────┴──────────────────┘
#
# Diagnostics: policy_diag_dir/policy_diag_flush_freq for .pt logging.

set -x

export CUDA_VISIBLE_DEVICES=4,5,6,7
export MAX_RESPONSE_LENGTH=8192

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=/home/work/DDAI_revised/verl/data/dapo-math-17k-processed/en_verl_25pct.parquet \
    data.val_files=/home/work/DDAI_revised/verl/data/aime-2024-repo/data/aime-2024.parquet \
    data.train_batch_size=64 \
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
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean" \
    actor_rollout_ref.actor.clip_ratio_low=0.01 \
    actor_rollout_ref.actor.clip_ratio_high=0.01 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    +actor_rollout_ref.actor.policy_diag_dir=/home/work/DDAI_revised/verl/logs/GSPO_qwen25_3b_instruct_diag_clip_0.01_0.01_nokl_seqmean \
    +actor_rollout_ref.actor.policy_diag_flush_freq=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
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
    +reward_model.reward_kwargs.answer_score_cfg.mode=one_or_zero \
    +reward_model.reward_kwargs.answer_score_cfg.log=True \
    +reward_model.reward_kwargs.max_resp_len=$MAX_RESPONSE_LENGTH \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=$((MAX_RESPONSE_LENGTH - 1024)) \
    +reward_model.reward_kwargs.overlong_buffer_cfg.mode=binary \
    +reward_model.reward_kwargs.overlong_buffer_cfg.binary_combine_mode=plus_one_or_zero \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='Exp.Custom.Algo' \
    trainer.experiment_name='GSPO_qwen25_3b_instruct_clip_0.01_0.01_nokl_seqmean' \
    trainer.rollout_data_dir='/home/work/DDAI_revised/verl/logs/GSPO_qwen25_3b_instruct_rollout_clip_0.01_0.01_nokl_seqmean' \
    trainer.validation_data_dir='/home/work/DDAI_revised/verl/logs/GSPO_qwen25_3b_instruct_validation_clip_0.01_0.01_nokl_seqmean' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.total_epochs=10 \
    "$@"
