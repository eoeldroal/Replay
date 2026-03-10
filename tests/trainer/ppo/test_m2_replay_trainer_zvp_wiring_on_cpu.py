# Copyright 2026

from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo import ray_trainer as ray_trainer_module
from verl.trainer.ppo.m2_replay import QueryGroupReplayBuffer, compute_zvp_stats_from_full_batch as compute_zvp_stats_from_full_batch_real
from verl.trainer.ppo.m2_replay_adapter import build_query_groups_from_onpolicy_batch as build_query_groups_real
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _make_onpolicy_batch() -> DataProto:
    tensors = {
        "responses": torch.arange(12).reshape(4, 3),
        "response_mask": torch.ones(4, 3, dtype=torch.long),
        "input_ids": torch.arange(12).reshape(4, 3),
        "attention_mask": torch.ones(4, 3, dtype=torch.long),
        "position_ids": torch.arange(3).repeat(4, 1),
        "old_log_probs": torch.log(torch.full((4, 3), 0.9, dtype=torch.float32)),
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
    non_tensors = {
        "uid": np.array(["q1", "q1", "q2", "q2"], dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def _make_trainer(*, training_mode: str, selection_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        m2_replay_enabled=True,
        m2_replay_buffer=QueryGroupReplayBuffer(max_query_groups=8),
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(
                    n=2,
                    log_prob_micro_batch_size_per_gpu=16,
                    log_prob_use_dynamic_bsz=False,
                    log_prob_max_token_len_per_gpu=4096,
                ),
                actor=SimpleNamespace(
                    ppo_mini_batch_size=1,
                    ppo_micro_batch_size_per_gpu=1,
                ),
            ),
            data={"train_batch_size": 2},
        ),
        global_steps=11,
        m2_replay_ingress_filter_mode="none",
        m2_replay_selection_mode=selection_mode,
        m2_replay_training_mode=training_mode,
        m2_replay_zvp_lambda_neg=1.0,
        m2_replay_zvp_mode="sign_only",
        m2_replay_adv_zero_eps=1e-8,
        m2_replay_fixed_total_groups=2,
        m2_replay_fixed_total_floor_groups=0,
        m2_replay_replay_target_groups=0,
        m2_replay_log_prob_micro_batch_size_per_gpu=None,
        m2_replay_log_prob_use_dynamic_bsz=None,
        m2_replay_log_prob_max_token_len_per_gpu=None,
        m2_replay_compute_runtime_zvp=False,
        m2_replay_gpu_m2_fastpath=False,
        m2_replay_start_mode="buffer_full",
        m2_replay_one_turnover_gate=False,
        m2_replay_recent_learning_beta=1.0,
        m2_replay_recent_learning_decay_lambda=1.0,
        m2_replay_zvp_weight=1.0,
        m2_replay_zvp_ema_alpha=0.5,
        m2_replay_zvp_adv_pos_eps=1e-8,
        m2_replay_zvp_use_recency=True,
        m2_replay_needs_buffer_zvp_backfill=False,
        m2_replay_prefix="m2",
        m2_replay_tau=0.01,
        m2_replay_logprob_groups_per_chunk=2,
        m2_replay_dump_full_scores=False,
        m2_replay_full_scores_dir=None,
        m2_replay_floor_to_micro=False,
        m2_replay_query_use_count=defaultdict(int),
        actor_rollout_wg=SimpleNamespace(),
        use_legacy_worker_impl="legacy",
        _get_dp_size=lambda worker_group, role: 1,
        _m2_key=lambda key: f"m2/{key}",
        _m2_dump_full_scores=lambda **kwargs: None,
    )


@pytest.mark.parametrize(
    (
        "training_mode",
        "selection_mode",
        "expected_compute_zvp_stats",
        "expected_build_update_count",
        "expected_full_batch_calls",
    ),
    [
        ("legacy_bonus", "zvp_recency", True, 1, 0),
        ("legacy_bonus", "legacy_uncertainty_recency", False, 0, 0),
        ("fixed_total_with_adv0_drop", "zvp_recency", False, 0, 1),
        ("fixed_total_with_adv0_drop", "legacy_uncertainty_recency", False, 0, 0),
    ],
)
def test_m2_build_actor_batch_wires_onpolicy_zvp_init_through_group_build(
    monkeypatch: pytest.MonkeyPatch,
    training_mode: str,
    selection_mode: str,
    expected_compute_zvp_stats: bool,
    expected_build_update_count: int,
    expected_full_batch_calls: int,
):
    captured: dict[str, object] = {}

    def _fake_build_query_groups_from_onpolicy_batch(**kwargs):
        captured["compute_zvp_stats"] = kwargs["compute_zvp_stats"]
        result = build_query_groups_real(**kwargs)
        captured["zvp_update_counts"] = [group.zvp_update_count for group in result.groups]
        return result

    def _fake_compute_zvp_stats_from_full_batch(**kwargs):
        captured["full_batch_calls"] = int(captured.get("full_batch_calls", 0)) + 1
        captured["full_batch_group_ids"] = [group.query_id for group in kwargs["groups"]]
        return compute_zvp_stats_from_full_batch_real(**kwargs)

    def _unexpected_update_query_groups_zvp_stats(*args, **kwargs):
        raise AssertionError("trainer should not call update_query_groups_zvp_stats for on-policy ZVP init")

    monkeypatch.setattr(
        ray_trainer_module,
        "build_query_groups_from_onpolicy_batch",
        _fake_build_query_groups_from_onpolicy_batch,
    )
    monkeypatch.setattr(
        ray_trainer_module,
        "update_query_groups_zvp_stats",
        _unexpected_update_query_groups_zvp_stats,
    )
    monkeypatch.setattr(
        ray_trainer_module,
        "compute_zvp_stats_from_full_batch",
        _fake_compute_zvp_stats_from_full_batch,
    )

    trainer = _make_trainer(training_mode=training_mode, selection_mode=selection_mode)
    metrics: dict[str, float] = {}
    timing_raw: dict[str, float] = {}

    actor_batch = RayPPOTrainer._m2_build_actor_batch(
        trainer,
        batch=_make_onpolicy_batch(),
        metrics=metrics,
        timing_raw=timing_raw,
    )

    assert captured["compute_zvp_stats"] is expected_compute_zvp_stats
    assert captured["zvp_update_counts"] == [expected_build_update_count, expected_build_update_count]
    assert int(captured.get("full_batch_calls", 0)) == expected_full_batch_calls
    if expected_full_batch_calls:
        assert captured["full_batch_group_ids"] == ["q1", "q2"]
        assert "m2_prepare_zvp_init" in timing_raw
    else:
        assert "m2_prepare_zvp_init" not in timing_raw
    assert len(trainer.m2_replay_buffer) == 2
    assert metrics["m2/buffer/new_groups"] == 2.0
    assert actor_batch.batch["m2_replay_source"].tolist() == [False, False, False, False]
