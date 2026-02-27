# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# GRPO training with Qwen3-8B on DAPO-Math-17k-Processed (competition-level math)
# NOTE: Uses 25% random subset (3,529 of 14,116) from the English partition (en_verl).
#       Sampled with random_state=42 for reproducibility. See en_verl_25pct.parquet.

set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3
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
    actor_rollout_ref.model.path=/home/work/DDAI_revised/verl/data/models/Qwen3-8B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
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
    trainer.experiment_name='Baseline.test' \
    trainer.rollout_data_dir='/home/work/DDAI_revised/verl/logs/Baseline_test_rollout' \
    trainer.validation_data_dir='/home/work/DDAI_revised/verl/logs/Baseline_test_validation' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=58 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.total_epochs=3 $@
