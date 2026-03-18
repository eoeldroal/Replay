from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo import ray_trainer as ray_trainer_module
from verl.trainer.ppo.m2_replay import QueryGroupReplayBuffer, ReplaySelectionResult
from verl.trainer.ppo.ray_trainer import (
    M2ReplayEntropyAreaReturnState,
    RayPPOTrainer,
    update_m2_replay_entropy_area_return_state,
)


def _make_batch() -> DataProto:
    tensors = {
        "responses": torch.arange(4).reshape(4, 1),
        "response_mask": torch.ones(4, 1, dtype=torch.long),
        "input_ids": torch.arange(4).reshape(4, 1),
        "attention_mask": torch.ones(4, 1, dtype=torch.long),
        "position_ids": torch.zeros(4, 1, dtype=torch.long),
        "old_log_probs": torch.zeros(4, 1, dtype=torch.float32),
        "advantages": torch.ones(4, 1, dtype=torch.float32),
        "token_level_scores": torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float32),
    }
    non_tensors = {
        "uid": np.array(["q0", "q0", "q1", "q1"], dtype=object),
        "multi_modal_inputs": np.array([{}, {}, {}, {}], dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def _make_trainer(*, replay_target_groups: int, area_enabled: bool, area_multiplier: float, area_state):
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
        m2_replay_ingress_filter_mode="none",
        m2_replay_ingress_success_score_threshold=None,
        m2_replay_selection_mode="legacy_uncertainty_recency",
        m2_replay_training_mode="legacy_bonus",
        m2_replay_zvp_lambda_neg=1.0,
        m2_replay_zvp_mode="sign_only",
        m2_replay_adv_zero_eps=1e-8,
        m2_replay_fixed_total_groups=2,
        m2_replay_fixed_total_floor_groups=0,
        m2_replay_replay_target_groups=replay_target_groups,
        m2_replay_log_prob_micro_batch_size_per_gpu=None,
        m2_replay_log_prob_use_dynamic_bsz=None,
        m2_replay_log_prob_max_token_len_per_gpu=None,
        m2_replay_compute_runtime_zvp=False,
        m2_replay_gpu_m2_fastpath=False,
        m2_replay_start_mode="immediate",
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
        m2_replay_entropy_lingering_enabled=False,
        m2_replay_entropy_lingering_state=None,
        m2_replay_entropy_area_return_enabled=area_enabled,
        m2_replay_entropy_area_return_multiplier=area_multiplier,
        m2_replay_entropy_area_return_state=area_state,
        actor_rollout_wg=SimpleNamespace(),
        use_legacy_worker_impl="legacy",
        _m2_compute_new_log_probs=lambda group_data: torch.zeros_like(group_data.batch["old_log_probs"]),
        _m2_compute_new_log_probs_batch=lambda group_data_list: [
            torch.zeros_like(group_data.batch["old_log_probs"]) for group_data in group_data_list
        ],
        _m2_compute_group_m2_batch=lambda group_data_list: [0.0 for _ in group_data_list],
        _get_dp_size=lambda worker_group, role: 1,
        _m2_key=lambda key: f"m2/{key}",
        _m2_dump_full_scores=lambda **kwargs: None,
    )


def test_entropy_area_return_gate_opens_once_recovery_mass_reaches_multiplier_times_delta():
    state = M2ReplayEntropyAreaReturnState()
    seq = [1.0, 1.2, 1.5, 1.3, 1.1, 0.9, 0.8]
    evals = []
    for step, entropy in enumerate(seq, start=1):
        state, gate_eval = update_m2_replay_entropy_area_return_state(
            state,
            entropy,
            current_step=step,
            gate_eligible=True,
            multiplier=2.0,
        )
        evals.append(gate_eval)

    assert evals[5].gate_open is False
    assert evals[6].gate_open is True
    assert evals[6].required_recovery_mass == pytest.approx((1.5 - (1.0 + 1.2 + 1.5) / 3.0) * 2.0, rel=1e-6)


def test_entropy_area_return_gate_respects_eligibility():
    state = M2ReplayEntropyAreaReturnState()
    for step, entropy in enumerate([1.0, 1.2, 1.5, 1.1, 0.9, 0.8], start=1):
        state, gate_eval = update_m2_replay_entropy_area_return_state(
            state,
            entropy,
            current_step=step,
            gate_eligible=False,
            multiplier=2.0,
        )

    assert gate_eval.gate_open is False


def test_m2_build_actor_batch_uses_area_return_gate_when_enabled(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def _fake_select_replay_groups(**kwargs):
        captured["target_groups"] = kwargs["target_groups"]
        return ReplaySelectionResult(
            selected_groups=[],
            scanned_groups=0,
            target_groups=int(kwargs["target_groups"]),
            rejected_missing_fields=0,
            rejected_by_tau=0,
            rejected_eval_error=0,
        )

    monkeypatch.setattr(ray_trainer_module, "select_replay_groups", _fake_select_replay_groups)

    trainer = _make_trainer(
        replay_target_groups=18,
        area_enabled=True,
        area_multiplier=2.0,
        area_state=M2ReplayEntropyAreaReturnState(
            prefix_sum=3.7,
            prefix_count=4,
            baseline_sum_at_peak=3.7,
            baseline_count_at_peak=4,
            peak_entropy=1.2,
            peak_step=3,
            recovery_mass=0.7,
            gate_open=False,
        ),
    )
    metrics = {"actor/entropy": 0.7}
    timing_raw = {}

    RayPPOTrainer._m2_build_actor_batch(
        trainer,
        batch=_make_batch(),
        metrics=metrics,
        timing_raw=timing_raw,
    )

    assert captured["target_groups"] == 18
    assert metrics["m2/schedule/replay_target_groups_base"] == 18.0
    assert metrics["m2/schedule/replay_target_groups_effective"] == 18.0
    assert metrics["m2/schedule/entropy_area_return_enabled"] == 1.0
    assert metrics["m2/schedule/entropy_area_return_multiplier"] == 2.0
    assert metrics["m2/schedule/entropy_area_return_gate_open"] == 1.0
