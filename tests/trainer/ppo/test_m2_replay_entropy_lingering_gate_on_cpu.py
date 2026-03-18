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
    M2ReplayEntropyLingeringState,
    RayPPOTrainer,
    update_m2_replay_entropy_lingering_state,
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


def _make_trainer(*, replay_target_groups: int, enabled: bool, state: M2ReplayEntropyLingeringState):
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
        m2_replay_entropy_lingering_enabled=enabled,
        m2_replay_entropy_lingering_state=state,
        m2_replay_entropy_area_return_enabled=False,
        m2_replay_entropy_area_return_multiplier=2.0,
        m2_replay_entropy_area_return_state=M2ReplayEntropyAreaReturnState(),
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


def test_entropy_lingering_gate_stays_off_for_geo_like_monotonic_decay():
    state = M2ReplayEntropyLingeringState()
    gates = []
    for entropy in [0.7219, 0.6731, 0.6847, 0.6199, 0.4670]:
        state, gate_eval = update_m2_replay_entropy_lingering_state(state, entropy, gate_eligible=True)
        gates.append(gate_eval.gate_open)

    assert gates == [False, False, False, False, False]


def test_entropy_lingering_gate_opens_only_after_search_like_peak_turns_over():
    state = M2ReplayEntropyLingeringState()
    seq = [0.3378, 0.3952, 0.5016, 0.7107, 0.6978, 0.6150, 0.5523]
    elig = [False, False, False, False, True, True, True]
    evals = []
    for entropy, gate_eligible in zip(seq, elig, strict=True):
        state, gate_eval = update_m2_replay_entropy_lingering_state(state, entropy, gate_eligible=gate_eligible)
        evals.append(gate_eval)

    assert evals[0].gate_open is False
    assert evals[3].gate_open is False
    assert evals[4].decline_streak == 0
    assert evals[-1].gate_open is False

    for entropy in [0.4860, 0.4700, 0.4620]:
        state, gate_eval = update_m2_replay_entropy_lingering_state(state, entropy, gate_eligible=True)

    assert gate_eval.gate_open is True


def test_entropy_lingering_gate_does_not_open_before_eligible():
    state = M2ReplayEntropyLingeringState()
    for entropy in [0.7173, 0.7512, 0.7113]:
        state, gate_eval = update_m2_replay_entropy_lingering_state(state, entropy, gate_eligible=False)

    assert gate_eval.gate_open is False


def test_m2_build_actor_batch_scales_replay_target_groups_when_gate_enabled(monkeypatch: pytest.MonkeyPatch):
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
        enabled=True,
        state=M2ReplayEntropyLingeringState(
            entropy_initial=1.0,
            cumulative_relative_overshoot=1.0,
            observed_steps=2,
            lingering_peak=0.6,
            decline_streak=1,
        ),
    )
    metrics = {"actor/entropy": 1.0}
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
    assert metrics["m2/schedule/entropy_lingering_enabled"] == 1.0
    assert metrics["m2/schedule/entropy_lingering_gate_open"] == 1.0
