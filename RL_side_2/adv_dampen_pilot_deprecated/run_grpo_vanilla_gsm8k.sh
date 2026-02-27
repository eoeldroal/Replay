# Vanilla GRPO baseline on GSM8K (for comparison with dampened version)
# Model: Qwen3-4B-Base, Dataset: GSM8K (7,473 train), Group size: G=16
#
# ONLY DIFFERENCES from run_grpo_dampened_gsm8k.sh:
#   algorithm.adv_estimator=grpo      (instead of grpo_dampened)
#   trainer.experiment_name            (grpo_vanilla)
#   trainer.rollout_data_dir           (separate log dir)
#   trainer.validation_data_dir        (separate log dir)
#
# All other settings are IDENTICAL to ensure fair comparison.
#
# Prerequisites:
#   huggingface-cli download Qwen/Qwen3-4B-Base \
#     --local-dir /home/work/DDAI_revised/verl/data/models/Qwen3-4B-Base

set -x

export CUDA_VISIBLE_DEVICES=4,5,7
export MAX_RESPONSE_LENGTH=1024

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.adv_dampen_diag_dir=/home/work/DDAI_revised/verl/logs/AdvDamp_vanilla_adv_diag \
    +algorithm.adv_dampen_diag_flush_freq=32 \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files=/home/work/DDAI_revised/verl/data/gsm8k/train.parquet \
    data.val_files=/home/work/DDAI_revised/verl/data/gsm8k/test.parquet \
    data.train_batch_size=384 \
    data.max_prompt_length=256 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/home/work/DDAI_revised/verl/data/models/Qwen3-4B-Base \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=96 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.policy_diag_dir=/home/work/DDAI_revised/verl/logs/AdvDamp_vanilla_diag \
    +actor_rollout_ref.actor.policy_diag_flush_freq=256 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.trace.backend=weave \
    actor_rollout_ref.rollout.trace.token2text=true \
    actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path=/home/work/DDAI_revised/verl/RL_side_2/adv_dampen_pilot/reward_gsm8k_with_format.py \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.max_resp_len=$MAX_RESPONSE_LENGTH \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=256 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='AdvDampen.Pilot' \
    trainer.experiment_name='grpo_vanilla' \
    trainer.rollout_data_dir='/home/work/DDAI_revised/verl/logs/AdvDamp_vanilla_rollout' \
    trainer.validation_data_dir='/home/work/DDAI_revised/verl/logs/AdvDamp_vanilla_validation' \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.val_before_train=True \
    trainer.total_epochs=5 $@
