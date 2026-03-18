# Copyright 2026

from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.m2_replay import QueryGroupReplayBuffer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _make_onpolicy_batch() -> DataProto:
    tensors = {
        "responses": torch.arange(12).reshape(4, 3),
        "response_mask": torch.ones(4, 3, dtype=torch.long),
        "input_ids": torch.arange(24).reshape(4, 6),
        "attention_mask": torch.ones(4, 6, dtype=torch.long),
        "position_ids": torch.arange(6).repeat(4, 1),
        "old_log_probs": torch.zeros(4, 3, dtype=torch.float32),
        "advantages": torch.ones(4, 3, dtype=torch.float32),
        "token_level_scores": torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    }
    non_tensors = {"uid": np.array(["q1", "q1", "q2", "q2"], dtype=object)}
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def _make_trainer() -> SimpleNamespace:
    return SimpleNamespace(
        m2_replay_enabled=True,
        m2_replay_buffer=QueryGroupReplayBuffer(max_query_groups=8),
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(
                    n=2,
                    log_prob_micro_batch_size_per_gpu=16,
                    log_prob_use_dynamic_bsz=True,
                    log_prob_max_token_len_per_gpu=4096,
                ),
                actor=SimpleNamespace(
                    ppo_mini_batch_size=1,
                    ppo_micro_batch_size_per_gpu=1,
                ),
            ),
            data={"train_batch_size": 2},
        ),
        global_steps=5,
        m2_replay_ingress_filter_mode="rlvr_non_degenerate",
        m2_replay_ingress_tool_response_pair_fraction_min=None,
        m2_replay_selection_mode="zvp_recency",
        m2_replay_training_mode="legacy_bonus",
        m2_replay_zvp_lambda_neg=1.0,
        m2_replay_zvp_mode="sign_only",
        m2_replay_adv_zero_eps=1e-8,
        m2_replay_fixed_total_groups=2,
        m2_replay_fixed_total_floor_groups=0,
        m2_replay_replay_target_groups=1,
        m2_replay_log_prob_micro_batch_size_per_gpu=4,
        m2_replay_log_prob_use_dynamic_bsz=True,
        m2_replay_log_prob_max_token_len_per_gpu=8192,
        m2_replay_compute_runtime_zvp=False,
        m2_replay_gpu_m2_fastpath=True,
        m2_replay_start_mode="quarter",
        m2_replay_one_turnover_gate=False,
        m2_replay_entropy_area_return_enabled=False,
        m2_replay_entropy_lingering_enabled=False,
        m2_replay_recent_learning_beta=1.0,
        m2_replay_recent_learning_decay_lambda=1.0,
        m2_replay_zvp_weight=1.0,
        m2_replay_zvp_ema_alpha=1.0,
        m2_replay_zvp_adv_pos_eps=1e-8,
        m2_replay_zvp_use_recency=False,
        m2_replay_needs_buffer_zvp_backfill=False,
        m2_replay_prefix="m2",
        m2_replay_tau=0.01,
        m2_replay_logprob_groups_per_chunk=2,
        m2_replay_dump_full_scores=False,
        m2_replay_full_scores_dir=None,
        m2_replay_floor_to_micro=False,
        m2_replay_query_use_count=defaultdict(int),
        actor_rollout_wg=SimpleNamespace(),
        use_legacy_worker_impl="enable",
        resource_pool_manager=SimpleNamespace(get_n_gpus=lambda: 1),
        _get_dp_size=lambda worker_group, role: 1,
        _m2_key=lambda key: f"m2/{key}",
        _m2_dump_full_scores=lambda **kwargs: None,
    )


def test_m2_metric_surface_keeps_core_metrics_only(capsys: pytest.CaptureFixture[str]):
    trainer = _make_trainer()
    metrics: dict[str, float] = {}
    timing_raw: dict[str, float] = {}

    RayPPOTrainer._m2_build_actor_batch(
        trainer,
        batch=_make_onpolicy_batch(),
        metrics=metrics,
        timing_raw=timing_raw,
    )

    required_keys = {
        "m2/buffer/new_groups",
        "m2/buffer/skipped_incomplete",
        "m2/buffer/skipped_missing_train_keys",
        "m2/buffer/skipped_by_success_band",
        "m2/schedule/onpolicy_ingress_groups",
        "m2/pass1/scanned_groups",
        "m2/pass1/accepted_groups",
        "m2/pass1/rejected_by_tau",
        "m2/pass1/acceptance_rate",
        "m2/pass1/accepted_m2_mean",
        "m2/selection/replay_used",
        "m2/selection/used_zvp_mean",
        "m2/buffer/size",
    }
    removed_keys = {
        "m2/selection/runtime_zvp_enabled",
        "m2/selection/replay_trimmed_for_floor",
        "m2/selection/log_prob_use_dynamic_bsz",
        "m2/selection/log_prob_micro_batch_size_per_gpu",
        "m2/selection/log_prob_max_token_len_per_gpu",
        "m2/selection/gpu_m2_fastpath_enabled",
        "m2/selection/fixed_total_target_applied",
        "m2/schedule/replay_target_groups",
        "m2/schedule/onpolicy_train_groups",
        "m2/pass2/selected_groups",
        "m2/pass1/target_groups",
        "m2/pass1/rejected_missing_fields",
        "m2/pass1/rejected_eval_error",
        "m2/gating/two_turnovers_ready",
        "m2/gating/start_mode_two_turnovers",
        "m2/gating/start_mode_quarter",
        "m2/gating/start_mode_immediate",
        "m2/gating/start_mode_half",
        "m2/gating/start_mode_buffer_full",
        "m2/gating/selection_ready",
        "m2/gating/quarter_threshold_groups",
        "m2/gating/quarter_ready",
        "m2/gating/one_turnover_ready",
        "m2/gating/one_turnover_enabled",
        "m2/gating/half_threshold_groups",
        "m2/gating/half_ready",
        "m2/gating/buffer_size_pre_select",
        "m2/gating/buffer_inserted_pre_select",
        "m2/gating/buffer_full_ready",
        "m2/gating/buffer_capacity",
    }

    assert required_keys.issubset(metrics.keys())
    assert removed_keys.isdisjoint(metrics.keys())

    output = capsys.readouterr().out
    assert "[m2_replay]" in output
    assert "[m2_replay_scores]" not in output
    assert "onpolicy_query_ids(" not in output
    assert "top_replayed_query_ids" not in output
    assert "ingress=" in output
    assert "rejected_by_tau=" in output
