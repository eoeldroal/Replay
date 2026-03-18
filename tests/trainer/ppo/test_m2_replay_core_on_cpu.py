import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.m2_replay import (
    QueryGroup,
    QueryGroupReplayBuffer,
    compute_group_m2,
    compute_group_m2_batched,
    compute_group_runtime_zvp_stats_batched,
    compute_group_zvp_sufficient_stats_batched,
    compute_group_zvp_stats_batched,
    compute_group_zvp_stats,
    derive_micro_group_multiple,
    update_query_groups_zvp_stats,
    select_replay_groups,
)


def _make_group(query_id: str, delta: float, success_prob: float, insertion_step: int) -> QueryGroup:
    old = torch.zeros(2, 3, dtype=torch.float32)
    mask = torch.ones(2, 3, dtype=torch.float32)
    data = DataProto.from_dict(
        tensors={
            "old_log_probs": old,
            "response_mask": mask,
            "delta": torch.full_like(old, fill_value=delta),
        },
        non_tensors={"uid": np.array([query_id, query_id], dtype=object)},
    )
    return QueryGroup(
        query_id=query_id,
        data=data,
        success_prob=success_prob,
        insertion_step=insertion_step,
    )


def test_replay_buffer_fifo_behavior():
    buffer = QueryGroupReplayBuffer(max_query_groups=2, seed=7)

    buffer.append_group(_make_group("q1", delta=0.0, success_prob=0.1, insertion_step=1))
    buffer.append_group(_make_group("q2", delta=0.0, success_prob=0.2, insertion_step=2))
    buffer.append_group(_make_group("q3", delta=0.0, success_prob=0.3, insertion_step=3))

    assert len(buffer) == 2
    assert [g.query_id for g in buffer.items()] == ["q2", "q3"]


def test_select_replay_groups_tau_and_target():
    buffer = QueryGroupReplayBuffer(max_query_groups=8, seed=123)
    buffer.append_group(_make_group("qA", delta=0.0, success_prob=0.49, insertion_step=10))
    buffer.append_group(_make_group("qB", delta=1.0, success_prob=0.50, insertion_step=11))
    buffer.append_group(_make_group("qC", delta=0.1, success_prob=0.40, insertion_step=12))

    def compute_new(data: DataProto) -> torch.Tensor:
        return data.batch["old_log_probs"] + data.batch["delta"]

    result = select_replay_groups(
        buffer=buffer,
        target_groups=2,
        tau=0.2,
        compute_new_log_probs_fn=compute_new,
        max_scan=len(buffer),
    )

    # qA (m2=0) and qC (m2=0.01) should pass; qB (m2=1) should be rejected.
    assert len(result.selected_groups) == 2
    assert set(g.query_id for g in result.selected_groups) == {"qA", "qC"}
    assert result.rejected_by_tau == 1


def test_select_replay_groups_m2_only_prefers_smallest_m2_over_insertion_order():
    buffer = QueryGroupReplayBuffer(max_query_groups=8, seed=123)
    buffer.append_group(_make_group("q1", delta=0.2, success_prob=0.5, insertion_step=1))
    buffer.append_group(_make_group("q2", delta=0.0, success_prob=0.5, insertion_step=2))
    buffer.append_group(_make_group("q3", delta=0.1, success_prob=0.5, insertion_step=3))

    def compute_new(data: DataProto) -> torch.Tensor:
        return data.batch["old_log_probs"] + data.batch["delta"]

    result = select_replay_groups(
        buffer=buffer,
        target_groups=2,
        tau=0.2,
        compute_new_log_probs_fn=compute_new,
        max_scan=len(buffer),
        selection_mode="m2_only",
    )

    assert [g.query_id for g in result.selected_groups] == ["q2", "q3"]
    assert result.accepted_m2 == pytest.approx([0.0, 0.01], rel=1e-6)


def test_derive_micro_group_multiple():
    assert derive_micro_group_multiple(micro_batch_size_per_gpu=8, n_gpus=8, rollout_n=8) == 8
    assert derive_micro_group_multiple(micro_batch_size_per_gpu=None, n_gpus=8, rollout_n=8) == 1
    assert derive_micro_group_multiple(micro_batch_size_per_gpu=8, n_gpus=0, rollout_n=8) == 1


def test_select_replay_groups_prefers_less_recently_trained_group():
    buffer = QueryGroupReplayBuffer(max_query_groups=8, seed=123)
    old_group = _make_group("old_q", delta=0.0, success_prob=0.5, insertion_step=10)
    recent_group = _make_group("recent_q", delta=0.0, success_prob=0.5, insertion_step=11)

    old_group.last_training_step = 1
    recent_group.last_training_step = 9

    buffer.append_group(old_group)
    buffer.append_group(recent_group)

    def compute_new(data: DataProto) -> torch.Tensor:
        return data.batch["old_log_probs"] + data.batch["delta"]

    result = select_replay_groups(
        buffer=buffer,
        target_groups=1,
        tau=0.2,
        compute_new_log_probs_fn=compute_new,
        max_scan=len(buffer),
        current_step=10,
        recency_beta=1.0,
        recency_decay_lambda=4.0,
    )

    assert len(result.selected_groups) == 1
    assert result.selected_groups[0].query_id == "old_q"


def test_compute_group_zvp_stats_adv_magnitude_changes_score():
    probs = torch.tensor([[0.2, 0.9, 0.9]], dtype=torch.float32)
    log_probs = torch.log(probs)
    advantages = torch.tensor([[10.0, -1.0, -1.0]], dtype=torch.float32)
    response_mask = torch.ones_like(advantages)

    sign_only_score, _, neg_frac, pos_frac = compute_group_zvp_stats(
        log_probs=log_probs,
        advantages=advantages,
        response_mask=response_mask,
        zvp_mode="sign_only",
    )
    adv_magnitude_score, _, neg_frac_weighted, pos_frac_weighted = compute_group_zvp_stats(
        log_probs=log_probs,
        advantages=advantages,
        response_mask=response_mask,
        zvp_mode="adv_magnitude",
    )

    assert sign_only_score == pytest.approx((0.8 + 0.9 + 0.9) / 3.0, rel=1e-6)
    assert adv_magnitude_score == pytest.approx((0.8 * 10.0 + 0.9 + 0.9) / 12.0, rel=1e-6)
    assert adv_magnitude_score < sign_only_score
    assert neg_frac == pytest.approx(2.0 / 3.0)
    assert pos_frac == pytest.approx(1.0 / 3.0)
    assert neg_frac_weighted == pytest.approx(neg_frac)
    assert pos_frac_weighted == pytest.approx(pos_frac)


def test_compute_group_zvp_stats_batched_matches_single_group_results():
    probs = torch.tensor(
        [
            [[0.2, 0.9, 0.9], [0.2, 0.9, 0.9]],
            [[0.8, 0.3, 0.7], [0.8, 0.3, 0.7]],
        ],
        dtype=torch.float32,
    )
    log_probs = torch.log(probs)
    advantages = torch.tensor(
        [
            [[10.0, -1.0, -1.0], [10.0, -1.0, -1.0]],
            [[-2.0, -2.0, 1.0], [-2.0, -2.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    response_mask = torch.ones_like(advantages)

    batched = compute_group_zvp_stats_batched(
        log_probs=log_probs,
        advantages=advantages,
        response_mask=response_mask,
        zvp_mode="adv_magnitude",
    )

    singles = [
        compute_group_zvp_stats(
            log_probs=log_probs[idx],
            advantages=advantages[idx],
            response_mask=response_mask[idx],
            zvp_mode="adv_magnitude",
        )
        for idx in range(log_probs.shape[0])
    ]

    for idx, single in enumerate(singles):
        assert batched[0][idx].item() == pytest.approx(single[0], rel=1e-6)
        assert batched[1][idx].item() == pytest.approx(single[1], rel=1e-6)
        assert batched[2][idx].item() == pytest.approx(single[2], rel=1e-6)
        assert batched[3][idx].item() == pytest.approx(single[3], rel=1e-6)


def test_compute_group_zvp_sufficient_stats_batched_matches_final_scores():
    probs = torch.tensor(
        [
            [[0.2, 0.9, 0.9], [0.2, 0.9, 0.9]],
            [[0.8, 0.3, 0.7], [0.8, 0.3, 0.7]],
        ],
        dtype=torch.float32,
    )
    log_probs = torch.log(probs)
    advantages = torch.tensor(
        [
            [[10.0, -1.0, -1.0], [10.0, -1.0, -1.0]],
            [[-2.0, -2.0, 1.0], [-2.0, -2.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    response_mask = torch.ones_like(advantages)
    score_num, score_denom, mean_surprisal_num, token_count, neg_count, pos_count = (
        compute_group_zvp_sufficient_stats_batched(
            log_probs=log_probs,
            advantages=advantages,
            response_mask=response_mask,
            zvp_mode="adv_magnitude",
        )
    )
    score, mean_surprisal, neg_frac, pos_frac = compute_group_zvp_stats_batched(
        log_probs=log_probs,
        advantages=advantages,
        response_mask=response_mask,
        zvp_mode="adv_magnitude",
    )

    assert torch.allclose(score, score_num / score_denom)
    assert torch.allclose(mean_surprisal, mean_surprisal_num / token_count)
    assert torch.allclose(neg_frac, neg_count / token_count)
    assert torch.allclose(pos_frac, pos_count / token_count)


def test_compute_group_m2_batched_matches_single_group_results():
    old_log_probs = torch.zeros(2, 2, 3, dtype=torch.float32)
    new_log_probs = torch.tensor(
        [
            [[0.0, 0.1, 0.1], [0.0, 0.1, 0.1]],
            [[0.0, 0.2, 0.2], [0.0, 0.2, 0.2]],
        ],
        dtype=torch.float32,
    )
    response_mask = torch.ones_like(old_log_probs)

    batched = compute_group_m2_batched(
        old_log_probs=old_log_probs,
        new_log_probs=new_log_probs,
        response_mask=response_mask,
    )
    singles = [
        compute_group_m2(
            old_log_probs=old_log_probs[idx],
            new_log_probs=new_log_probs[idx],
            response_mask=response_mask[idx],
        )
        for idx in range(old_log_probs.shape[0])
    ]

    assert batched[0].item() == pytest.approx(singles[0], rel=1e-6)
    assert batched[1].item() == pytest.approx(singles[1], rel=1e-6)


def test_update_query_groups_zvp_stats_stores_sufficient_statistics():
    probs = torch.tensor([[0.2, 0.9, 0.9]], dtype=torch.float32)
    log_probs = torch.log(probs)
    advantages = torch.tensor([[10.0, -1.0, -1.0]], dtype=torch.float32)
    response_mask = torch.ones_like(advantages)
    data = DataProto.from_dict(
        tensors={
            "old_log_probs": log_probs,
            "advantages": advantages,
            "response_mask": response_mask,
        },
        non_tensors={"uid": np.array(["q1"], dtype=object)},
    )
    group = QueryGroup(query_id="q1", data=data, success_prob=0.5, insertion_step=3)

    processed = update_query_groups_zvp_stats([group], update_step=7, zvp_mode="adv_magnitude")

    assert processed == 1
    assert group.zvp_score == pytest.approx((0.8 * 10.0 + 0.9 + 0.9) / 12.0, rel=1e-6)
    assert group.zvp_score_num == pytest.approx(9.8, rel=1e-6)
    assert group.zvp_score_denom == pytest.approx(12.0, rel=1e-6)
    assert group.zvp_token_count == 3
    assert group.zvp_neg_count == 2
    assert group.zvp_pos_count == 1
    assert group.zvp_update_count == 1
    assert group.zvp_last_update_step == 7


def test_compute_group_runtime_zvp_stats_batched_matches_single_group_results():
    old = torch.log(torch.tensor([[0.2, 0.9, 0.9]], dtype=torch.float32))
    advantages = torch.tensor([[10.0, -1.0, -1.0]], dtype=torch.float32)
    mask = torch.ones_like(advantages)

    groups = []
    new_log_probs_list = []
    for idx, delta in enumerate([0.0, 0.1]):
        data = DataProto.from_dict(
            tensors={
                "old_log_probs": old,
                "response_mask": mask,
                "advantages": advantages,
            },
            non_tensors={"uid": np.array([f"q{idx}"], dtype=object)},
        )
        groups.append(QueryGroup(query_id=f"q{idx}", data=data, success_prob=0.5, insertion_step=1))
        new_log_probs_list.append(old + delta)

    batched = compute_group_runtime_zvp_stats_batched(
        groups,
        new_log_probs_list,
        zvp_mode="adv_magnitude",
    )
    singles = [
        compute_group_zvp_stats(
            log_probs=new_log_probs,
            advantages=advantages,
            response_mask=mask,
            zvp_mode="adv_magnitude",
        )
        for new_log_probs in new_log_probs_list
    ]

    for batched_stats, single_stats in zip(batched, singles, strict=True):
        assert batched_stats[0] == pytest.approx(single_stats[0], rel=1e-6)
        assert batched_stats[1] == pytest.approx(single_stats[1], rel=1e-6)
        assert batched_stats[2] == pytest.approx(single_stats[2], rel=1e-6)
        assert batched_stats[3] == pytest.approx(single_stats[3], rel=1e-6)


def test_select_replay_groups_runtime_zvp_uses_requested_mode():
    old = torch.log(torch.tensor([[0.2, 0.9, 0.9]], dtype=torch.float32))
    advantages = torch.tensor([[10.0, -1.0, -1.0]], dtype=torch.float32)
    mask = torch.ones_like(advantages)
    data = DataProto.from_dict(
        tensors={
            "old_log_probs": old,
            "response_mask": mask,
            "advantages": advantages,
            "delta": torch.zeros_like(old),
        },
        non_tensors={"uid": np.array(["q1"], dtype=object)},
    )
    group = QueryGroup(query_id="q1", data=data, success_prob=0.0625, insertion_step=1)
    buffer = QueryGroupReplayBuffer(max_query_groups=4, seed=0)
    buffer.append_group(group)

    def compute_new(data: DataProto) -> torch.Tensor:
        return data.batch["old_log_probs"] + data.batch["delta"]

    sign_only = select_replay_groups(
        buffer=buffer,
        target_groups=1,
        tau=1e-6,
        compute_new_log_probs_fn=compute_new,
        max_scan=1,
        selection_mode="zvp_recency",
        zvp_mode="sign_only",
    )
    adv_magnitude = select_replay_groups(
        buffer=buffer,
        target_groups=1,
        tau=1e-6,
        compute_new_log_probs_fn=compute_new,
        max_scan=1,
        selection_mode="zvp_recency",
        zvp_mode="adv_magnitude",
    )

    assert len(sign_only.selected_groups) == 1
    assert len(adv_magnitude.selected_groups) == 1
    sign_only_runtime = sign_only.selected_priority_debug[0]["zvp_runtime_score"]
    adv_magnitude_runtime = adv_magnitude.selected_priority_debug[0]["zvp_runtime_score"]
    assert sign_only_runtime == pytest.approx((0.8 + 0.9 + 0.9) / 3.0, rel=1e-6)
    assert adv_magnitude_runtime == pytest.approx((0.8 * 10.0 + 0.9 + 0.9) / 12.0, rel=1e-6)
    assert adv_magnitude_runtime < sign_only_runtime


def test_select_replay_groups_chunked_matches_scalar_path():
    buffer = QueryGroupReplayBuffer(max_query_groups=8, seed=123)
    deltas = {"qA": 0.0, "qB": 1.0, "qC": 0.1}
    for step, (query_id, delta) in enumerate(deltas.items(), start=10):
        group = _make_group(query_id, delta=delta, success_prob=0.5, insertion_step=step)
        group.data.batch["advantages"] = torch.ones_like(group.data.batch["old_log_probs"])
        buffer.append_group(group)

    def compute_new(data: DataProto) -> torch.Tensor:
        return data.batch["old_log_probs"] + data.batch["delta"]

    def compute_new_batch(datas: list[DataProto]) -> list[torch.Tensor]:
        return [compute_new(data) for data in datas]

    scalar = select_replay_groups(
        buffer=buffer,
        target_groups=2,
        tau=0.2,
        compute_new_log_probs_fn=compute_new,
        max_scan=len(buffer),
        selection_mode="zvp_recency",
        zvp_mode="sign_only",
    )
    chunked = select_replay_groups(
        buffer=buffer,
        target_groups=2,
        tau=0.2,
        compute_new_log_probs_fn=compute_new,
        compute_new_log_probs_batch_fn=compute_new_batch,
        groups_per_chunk=2,
        max_scan=len(buffer),
        selection_mode="zvp_recency",
        zvp_mode="sign_only",
    )

    assert [group.query_id for group in chunked.selected_groups] == [group.query_id for group in scalar.selected_groups]
    assert chunked.rejected_by_tau == scalar.rejected_by_tau
    assert chunked.accepted_m2 == pytest.approx(scalar.accepted_m2, rel=1e-6)
    assert len(chunked.selected_priority_debug) == len(scalar.selected_priority_debug)
    for chunked_debug, scalar_debug in zip(chunked.selected_priority_debug, scalar.selected_priority_debug, strict=True):
        assert chunked_debug["query_id"] == scalar_debug["query_id"]
        assert chunked_debug["m2"] == pytest.approx(scalar_debug["m2"], rel=1e-6)
        assert chunked_debug["zvp_runtime_score"] == pytest.approx(scalar_debug["zvp_runtime_score"], rel=1e-6)
