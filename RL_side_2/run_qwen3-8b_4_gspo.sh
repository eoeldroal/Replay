# GSPO training with Qwen3-8B on DAPO-Math-17k-Processed (competition-level math)
# Based on run_qwen3-8b_4.sh (GRPO baseline), switched to GSPO policy loss.
# NOTE: Uses 25% random subset (3,529 of 14,116) from the English partition (en_verl).
#       Sampled with random_state=42 for reproducibility. See en_verl_25pct.parquet.
#
# Key differences from GRPO baseline:
#   - policy_loss.loss_mode: vanilla -> gspo
#   - loss_agg_mode: token-mean (default) -> seq-mean-token-mean
#   - clip_ratio_low/high: 0.2 (default) -> 0.0003/0.0004 (GSPO-tight clipping)
#   - clip_ratio_c: 3.0 (default) -> 10.0
#   - kl_loss_coef explicitly set to 0.0

set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MAX_RESPONSE_LENGTH=8192

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files=/home/work/DDAI_revised/verl/data/dapo-math-17k-processed/en_verl_25pct.parquet \
    data.val_files=/home/work/DDAI_revised/verl/data/aime-2024-repo/data/aime-2024.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/home/work/DDAI_revised/verl/data/models/Qwen3-8B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean" \
    actor_rollout_ref.actor.clip_ratio_low=0.0003 \
    actor_rollout_ref.actor.clip_ratio_high=0.0004 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.trace.backend=weave \
    actor_rollout_ref.rollout.trace.token2text=true \
    actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.max_resp_len=$MAX_RESPONSE_LENGTH \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=$((MAX_RESPONSE_LENGTH / 2)) \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='Exp.Custom.Algo' \
    trainer.experiment_name='GSPO.test' \
    trainer.rollout_data_dir='/home/work/DDAI_revised/verl/logs/GSPO_test_rollout' \
    trainer.validation_data_dir='/home/work/DDAI_revised/verl/logs/GSPO_test_validation' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=58 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.total_epochs=3 $@
