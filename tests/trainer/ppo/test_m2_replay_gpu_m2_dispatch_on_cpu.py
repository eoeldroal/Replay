# Copyright 2026

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _make_group_data(rows: int, delta: float) -> DataProto:
    return DataProto.from_dict(
        tensors={
            "old_log_probs": torch.zeros(rows, 3, dtype=torch.float32),
            "response_mask": torch.ones(rows, 3, dtype=torch.float32),
            "delta": torch.full((rows, 3), fill_value=delta, dtype=torch.float32),
        }
    )


def _make_group_data_with_workload(rows: int, delta: float, prompt_len: int) -> DataProto:
    total_len = 16
    attention_mask = torch.zeros(rows, total_len, dtype=torch.long)
    attention_mask[:, : prompt_len + 3] = 1
    return DataProto.from_dict(
        tensors={
            "old_log_probs": torch.zeros(rows, 3, dtype=torch.float32),
            "response_mask": torch.ones(rows, 3, dtype=torch.float32),
            "delta": torch.full((rows, 3), fill_value=delta, dtype=torch.float32),
            "attention_mask": attention_mask,
        },
        non_tensors={"uid": np.array([f"p{prompt_len}"] * rows, dtype=object)},
    )


def _compute_group_m2_from_chunk(batch: DataProto) -> DataProto:
    group_lengths = [int(length) for length in batch.meta_info["m2_group_lengths"]]
    batch_rows = int(batch.batch["old_log_probs"].shape[0])
    assert sum(group_lengths) == batch_rows

    if batch_rows == 0:
        return DataProto.from_dict(tensors={"m2": torch.empty(0, dtype=torch.float32)})

    old = batch.batch["old_log_probs"].float()
    new = old + batch.batch["delta"].float()
    mask = batch.batch["response_mask"].float()
    row_num = ((new - old).square() * mask).sum(dim=1)
    row_denom = mask.sum(dim=1)

    values = []
    start = 0
    for group_length in group_lengths:
        end = start + group_length
        values.append(float((row_num[start:end].sum() / row_denom[start:end].sum()).item()))
        start = end
    return DataProto.from_dict(tensors={"m2": torch.tensor(values, dtype=torch.float32)})


class _FakeActorRolloutWG:
    def __init__(self, workers: int):
        self.workers = workers
        self.chunk_rows: list[int] = []
        self.chunk_group_lengths: list[list[int]] = []

    def compute_log_prob_m2(self, batch: DataProto) -> DataProto:
        worker_batches = batch.chunk(self.workers)
        self.chunk_rows = [len(worker_batch) for worker_batch in worker_batches]
        self.chunk_group_lengths = [
            [int(length) for length in worker_batch.meta_info["m2_group_lengths"]] for worker_batch in worker_batches
        ]
        return DataProto.concat([_compute_group_m2_from_chunk(worker_batch) for worker_batch in worker_batches])


def test_m2_compute_group_m2_batch_preserves_group_boundaries_across_worker_chunks():
    actor_rollout_wg = _FakeActorRolloutWG(workers=3)
    trainer = SimpleNamespace(
        actor_rollout_wg=actor_rollout_wg,
        m2_replay_log_prob_micro_batch_size_per_gpu=None,
        m2_replay_log_prob_use_dynamic_bsz=None,
        m2_replay_log_prob_max_token_len_per_gpu=None,
    )

    values = RayPPOTrainer._m2_compute_group_m2_batch(
        trainer,
        [
            _make_group_data(rows=8, delta=0.1),
            _make_group_data(rows=8, delta=0.2),
            _make_group_data(rows=8, delta=0.3),
            _make_group_data(rows=8, delta=0.4),
            _make_group_data(rows=8, delta=0.5),
        ],
    )

    assert values == pytest.approx([0.01, 0.04, 0.09, 0.16, 0.25], rel=1e-6)
    assert actor_rollout_wg.chunk_rows == [16, 16, 8]
    assert actor_rollout_wg.chunk_group_lengths == [[8, 8], [8, 8], [8]]


def test_m2_compute_group_m2_batch_handles_more_workers_than_groups():
    actor_rollout_wg = _FakeActorRolloutWG(workers=4)
    trainer = SimpleNamespace(
        actor_rollout_wg=actor_rollout_wg,
        m2_replay_log_prob_micro_batch_size_per_gpu=None,
        m2_replay_log_prob_use_dynamic_bsz=None,
        m2_replay_log_prob_max_token_len_per_gpu=None,
    )

    values = RayPPOTrainer._m2_compute_group_m2_batch(
        trainer,
        [
            _make_group_data(rows=2, delta=0.25),
            _make_group_data(rows=3, delta=0.5),
        ],
    )

    assert values == pytest.approx([0.0625, 0.25], rel=1e-6)
    assert actor_rollout_wg.chunk_rows == [2, 3, 0, 0]
    assert actor_rollout_wg.chunk_group_lengths == [[2], [3], [], []]


def test_m2_compute_group_m2_batch_restores_original_order_after_reorder_and_dispatch():
    actor_rollout_wg = _FakeActorRolloutWG(workers=3)
    trainer = SimpleNamespace(
        actor_rollout_wg=actor_rollout_wg,
        m2_replay_log_prob_micro_batch_size_per_gpu=None,
        m2_replay_log_prob_use_dynamic_bsz=None,
        m2_replay_log_prob_max_token_len_per_gpu=None,
    )

    group_data_list = [
        _make_group_data_with_workload(rows=9, delta=0.5, prompt_len=9),
        _make_group_data_with_workload(rows=1, delta=0.1, prompt_len=1),
        _make_group_data_with_workload(rows=8, delta=0.4, prompt_len=8),
        _make_group_data_with_workload(rows=2, delta=0.2, prompt_len=2),
        _make_group_data_with_workload(rows=7, delta=0.3, prompt_len=7),
    ]

    values = RayPPOTrainer._m2_compute_group_m2_batch(trainer, group_data_list)

    assert values == pytest.approx([0.25, 0.01, 0.16, 0.04, 0.09], rel=1e-6)
    assert sum(actor_rollout_wg.chunk_rows) == sum(len(group_data) for group_data in group_data_list)
    flattened_chunk_lengths = [length for chunk_lengths in actor_rollout_wg.chunk_group_lengths for length in chunk_lengths]
    assert sorted(flattened_chunk_lengths) == [1, 2, 7, 8, 9]
