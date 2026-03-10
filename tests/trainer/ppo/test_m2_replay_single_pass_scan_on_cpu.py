# Copyright 2026

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


def test_single_pass_chunked_path_keeps_later_accepts_after_earlier_rejects():
    buffer = QueryGroupReplayBuffer(max_query_groups=16, seed=11)
    for step, delta in enumerate([0.0, 1.0, 0.1, 0.2], start=1):
        buffer.append_group(_make_group(f"q{step}", delta=delta, insertion_step=step))

    scalar = select_replay_groups(
        buffer=buffer,
        target_groups=3,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        max_scan=len(buffer),
        selection_mode="recency_only",
        compute_runtime_zvp=False,
        groups_per_chunk=1,
    )

    def _compute_new_batch(data_list: list[DataProto]) -> list[torch.Tensor]:
        return [_compute_new(data) for data in data_list]

    single_pass = select_replay_groups(
        buffer=buffer,
        target_groups=3,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        compute_new_log_probs_batch_fn=_compute_new_batch,
        max_scan=len(buffer),
        selection_mode="recency_only",
        compute_runtime_zvp=False,
        groups_per_chunk=4,
    )

    assert [group.query_id for group in single_pass.selected_groups] == [group.query_id for group in scalar.selected_groups]
    assert [group.query_id for group in single_pass.selected_groups] == ["q1", "q3", "q4"]
    assert single_pass.rejected_by_tau == scalar.rejected_by_tau == 1
    assert single_pass.accepted_m2 == pytest.approx(scalar.accepted_m2, rel=1e-6)


def test_single_pass_chunked_path_calls_batch_logprob_once_for_all_valid_groups():
    buffer = QueryGroupReplayBuffer(max_query_groups=16, seed=5)
    for step, delta in enumerate([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], start=1):
        buffer.append_group(_make_group(f"q{step}", delta=delta, insertion_step=step))

    captured_group_counts: list[int] = []

    def _compute_new_batch(data_list: list[DataProto]) -> list[torch.Tensor]:
        captured_group_counts.append(len(data_list))
        return [_compute_new(data) for data in data_list]

    select_replay_groups(
        buffer=buffer,
        target_groups=2,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        compute_new_log_probs_batch_fn=_compute_new_batch,
        max_scan=len(buffer),
        selection_mode="recency_only",
        compute_runtime_zvp=False,
        groups_per_chunk=4,
    )

    assert captured_group_counts == [len(buffer)]


def test_single_pass_gpu_fastpath_calls_group_m2_once_for_all_valid_groups():
    buffer = QueryGroupReplayBuffer(max_query_groups=16, seed=13)
    for step, delta in enumerate([0.0, 0.1, 0.2, 0.8, 1.0], start=1):
        buffer.append_group(_make_group(f"q{step}", delta=delta, insertion_step=step))

    captured_group_counts: list[int] = []

    def _compute_new_batch(data_list: list[DataProto]) -> list[torch.Tensor]:
        return [_compute_new(data) for data in data_list]

    def _compute_group_m2_batch(data_list: list[DataProto]) -> list[float]:
        captured_group_counts.append(len(data_list))
        m2_values: list[float] = []
        for data in data_list:
            old = data.batch["old_log_probs"].float()
            new = _compute_new(data).float()
            mask = data.batch["response_mask"].float()
            denom = mask.sum()
            m2_values.append(float((((new - old).square() * mask).sum() / denom).item()))
        return m2_values

    result = select_replay_groups(
        buffer=buffer,
        target_groups=3,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        compute_new_log_probs_batch_fn=_compute_new_batch,
        compute_group_m2_batch_fn=_compute_group_m2_batch,
        max_scan=len(buffer),
        selection_mode="recency_only",
        compute_runtime_zvp=False,
        groups_per_chunk=4,
    )

    assert captured_group_counts == [len(buffer)]
    assert [group.query_id for group in result.selected_groups] == ["q1", "q2", "q3"]
