import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.m2_replay import QueryGroup, QueryGroupReplayBuffer, select_replay_groups


def _make_group(query_id: str, delta: float, insertion_step: int) -> QueryGroup:
    old = torch.zeros(2, 3, dtype=torch.float32)
    mask = torch.ones(2, 3, dtype=torch.float32)
    data = DataProto.from_dict(
        tensors={
            "old_log_probs": old,
            "response_mask": mask,
            "advantages": torch.ones_like(old),
            "delta": torch.full_like(old, fill_value=delta),
        },
        non_tensors={"uid": np.array([query_id, query_id], dtype=object)},
    )
    return QueryGroup(
        query_id=query_id,
        data=data,
        success_prob=0.5,
        insertion_step=insertion_step,
    )


def _compute_new(data: DataProto) -> torch.Tensor:
    return data.batch["old_log_probs"] + data.batch["delta"]


def _compute_new_batch(data_list: list[DataProto]) -> list[torch.Tensor]:
    return [_compute_new(data) for data in data_list]


def test_chunked_replay_selection_matches_per_group_path():
    buffer = QueryGroupReplayBuffer(max_query_groups=16, seed=42)
    # m2 = delta^2, so tau=0.05 accepts first three, rejects the rest.
    deltas = [0.0, 0.1, 0.2, 0.3, 0.6, 1.0]
    for i, delta in enumerate(deltas, start=1):
        buffer.append_group(_make_group(f"q{i}", delta=delta, insertion_step=i))

    result_per_group = select_replay_groups(
        buffer=buffer,
        target_groups=3,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        max_scan=len(buffer),
        selection_mode="recency_only",
        groups_per_chunk=1,
    )

    result_chunked = select_replay_groups(
        buffer=buffer,
        target_groups=3,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        compute_new_log_probs_batch_fn=_compute_new_batch,
        max_scan=len(buffer),
        selection_mode="recency_only",
        groups_per_chunk=4,
    )

    assert [g.query_id for g in result_chunked.selected_groups] == [
        g.query_id for g in result_per_group.selected_groups
    ]
    assert len(result_chunked.selected_groups) == 3
    assert all(g.query_id in {"q1", "q2", "q3"} for g in result_chunked.selected_groups)


def test_chunked_path_never_exceeds_target_groups():
    buffer = QueryGroupReplayBuffer(max_query_groups=16, seed=7)
    # All candidates pass tau.
    for i in range(1, 13):
        buffer.append_group(_make_group(f"q{i}", delta=0.0, insertion_step=i))

    result_chunked = select_replay_groups(
        buffer=buffer,
        target_groups=5,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        compute_new_log_probs_batch_fn=_compute_new_batch,
        max_scan=len(buffer),
        selection_mode="recency_only",
        groups_per_chunk=8,
    )

    assert len(result_chunked.selected_groups) == 5
    assert [g.query_id for g in result_chunked.selected_groups] == [f"q{i}" for i in range(1, 6)]


def test_chunked_path_falls_back_to_per_group_when_batch_eval_fails():
    buffer = QueryGroupReplayBuffer(max_query_groups=16, seed=21)
    deltas = [0.0, 0.1, 0.2, 0.8]
    for i, delta in enumerate(deltas, start=1):
        buffer.append_group(_make_group(f"q{i}", delta=delta, insertion_step=i))

    def _broken_batch_eval(_: list[DataProto]) -> list[torch.Tensor]:
        raise RuntimeError("intentional failure")

    result_fallback = select_replay_groups(
        buffer=buffer,
        target_groups=3,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        compute_new_log_probs_batch_fn=_broken_batch_eval,
        max_scan=len(buffer),
        selection_mode="recency_only",
        groups_per_chunk=4,
    )

    result_reference = select_replay_groups(
        buffer=buffer,
        target_groups=3,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        max_scan=len(buffer),
        selection_mode="recency_only",
        groups_per_chunk=1,
    )

    assert [g.query_id for g in result_fallback.selected_groups] == [
        g.query_id for g in result_reference.selected_groups
    ]
    assert len(result_fallback.selected_groups) == 3


def test_chunked_path_falls_back_when_batch_result_count_mismatch():
    buffer = QueryGroupReplayBuffer(max_query_groups=16, seed=9)
    for i in range(1, 6):
        buffer.append_group(_make_group(f"q{i}", delta=0.0, insertion_step=i))

    def _short_batch_eval(data_list: list[DataProto]) -> list[torch.Tensor]:
        # Intentionally return fewer tensors than requested.
        return [_compute_new(data_list[0])]

    result_fallback = select_replay_groups(
        buffer=buffer,
        target_groups=4,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        compute_new_log_probs_batch_fn=_short_batch_eval,
        max_scan=len(buffer),
        selection_mode="recency_only",
        groups_per_chunk=4,
    )

    result_reference = select_replay_groups(
        buffer=buffer,
        target_groups=4,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        max_scan=len(buffer),
        selection_mode="recency_only",
        groups_per_chunk=1,
    )

    assert [g.query_id for g in result_fallback.selected_groups] == [
        g.query_id for g in result_reference.selected_groups
    ]
    assert len(result_fallback.selected_groups) == 4


def test_chunked_path_respects_max_scan_limit():
    buffer = QueryGroupReplayBuffer(max_query_groups=32, seed=19)
    for i in range(1, 13):
        buffer.append_group(_make_group(f"q{i}", delta=0.0, insertion_step=i))

    # max_scan=6 means only first six recency-ordered groups can be considered.
    result_chunked = select_replay_groups(
        buffer=buffer,
        target_groups=10,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        compute_new_log_probs_batch_fn=_compute_new_batch,
        max_scan=6,
        selection_mode="recency_only",
        groups_per_chunk=4,
    )

    assert result_chunked.scanned_groups == 6
    assert [g.query_id for g in result_chunked.selected_groups] == [f"q{i}" for i in range(1, 7)]


def test_chunked_m2_only_prefers_lowest_m2_candidates():
    buffer = QueryGroupReplayBuffer(max_query_groups=16, seed=42)
    deltas = [0.2, 0.0, 0.1, 0.3]
    for i, delta in enumerate(deltas, start=1):
        buffer.append_group(_make_group(f"q{i}", delta=delta, insertion_step=i))

    result_chunked = select_replay_groups(
        buffer=buffer,
        target_groups=2,
        tau=0.2,
        compute_new_log_probs_fn=_compute_new,
        compute_new_log_probs_batch_fn=_compute_new_batch,
        max_scan=len(buffer),
        selection_mode="m2_only",
        groups_per_chunk=4,
    )

    assert [g.query_id for g in result_chunked.selected_groups] == ["q2", "q3"]
    assert result_chunked.accepted_m2 == pytest.approx([0.0, 0.01], rel=1e-6)
