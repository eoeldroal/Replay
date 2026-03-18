from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo import ray_trainer as ray_trainer_module
from verl.trainer.ppo.m2_replay import QueryGroupReplayBuffer
from verl.trainer.ppo.m2_replay_adapter import build_query_groups_from_onpolicy_batch as build_query_groups_real
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _make_threshold_batch() -> DataProto:
    tensors = {
        "responses": torch.arange(4).reshape(4, 1),
        "response_mask": torch.ones(4, 1, dtype=torch.long),
        "input_ids": torch.arange(4).reshape(4, 1),
        "attention_mask": torch.ones(4, 1, dtype=torch.long),
        "position_ids": torch.zeros(4, 1, dtype=torch.long),
        "old_log_probs": torch.zeros(4, 1, dtype=torch.float32),
        "advantages": torch.ones(4, 1, dtype=torch.float32),
        "token_level_scores": torch.tensor([[0.1], [0.1], [1.0], [0.1]], dtype=torch.float32),
    }
    non_tensors = {
        "uid": np.array(["format_only", "format_only", "answer_mixed", "answer_mixed"], dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def test_build_query_groups_supports_optional_ingress_success_score_threshold():
    batch = _make_threshold_batch()

    legacy_result = build_query_groups_real(
        batch=batch,
        expected_group_size=2,
        insertion_step=3,
    )
    threshold_result = build_query_groups_real(
        batch=batch,
        expected_group_size=2,
        insertion_step=3,
        success_score_threshold=0.9,
    )
    threshold_filtered = build_query_groups_real(
        batch=batch,
        expected_group_size=2,
        insertion_step=3,
        success_count_min=1,
        success_count_max=1,
        success_score_threshold=0.9,
    )

    legacy_counts = {group.query_id: group.success_count for group in legacy_result.groups}
    threshold_counts = {group.query_id: group.success_count for group in threshold_result.groups}

    assert legacy_counts == {"format_only": 2, "answer_mixed": 2}
    assert threshold_counts == {"format_only": 0, "answer_mixed": 1}
    assert [group.query_id for group in threshold_filtered.groups] == ["answer_mixed"]
    assert threshold_filtered.skipped_by_success_band == 1



def _make_trainer(*, ingress_success_score_threshold):
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
        global_steps=5,
        m2_replay_ingress_filter_mode="rlvr_non_degenerate",
        m2_replay_ingress_success_score_threshold=ingress_success_score_threshold,
        m2_replay_ingress_tool_response_pair_fraction_min=None,
        m2_replay_selection_mode="legacy_uncertainty_recency",
        m2_replay_training_mode="legacy_bonus",
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
        m2_replay_entropy_area_return_enabled=False,
        m2_replay_entropy_lingering_enabled=False,
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


@pytest.mark.parametrize("threshold", [None, 0.9])
def test_m2_build_actor_batch_passes_ingress_success_score_threshold(monkeypatch: pytest.MonkeyPatch, threshold):
    captured = {}

    def _fake_build_query_groups_from_onpolicy_batch(**kwargs):
        captured["success_score_threshold"] = kwargs.get("success_score_threshold")
        return build_query_groups_real(**kwargs)

    monkeypatch.setattr(
        ray_trainer_module,
        "build_query_groups_from_onpolicy_batch",
        _fake_build_query_groups_from_onpolicy_batch,
    )

    trainer = _make_trainer(ingress_success_score_threshold=threshold)
    metrics = {}
    timing_raw = {}

    RayPPOTrainer._m2_build_actor_batch(
        trainer,
        batch=_make_threshold_batch(),
        metrics=metrics,
        timing_raw=timing_raw,
    )

    assert captured["success_score_threshold"] == threshold
