# Copyright 2026

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.m2_replay_adapter import (
    M2_REPLAY_SOURCE_KEY,
    build_balanced_group_eval_order,
    compute_actor_batch_metrics,
    rebalance_batch_by_attention,
    reorder_group_data_for_logprob_eval,
    restore_group_output_order,
)


def _make_group_data(
    query_id: str,
    prompt_len: int,
    delta: float,
    response_len: int = 3,
    max_prompt_len: int = 9,
) -> DataProto:
    total_len = max_prompt_len + response_len
    attention_mask = torch.zeros(1, total_len, dtype=torch.long)
    attention_mask[:, : prompt_len + response_len] = 1
    response_mask = torch.ones(1, response_len, dtype=torch.long)
    old_log_probs = torch.zeros(1, response_len, dtype=torch.float32)
    delta_tensor = torch.full((1, response_len), fill_value=delta, dtype=torch.float32)
    tensors = {
        "responses": torch.arange(response_len).reshape(1, response_len),
        "response_mask": response_mask,
        "input_ids": torch.arange(total_len).reshape(1, total_len),
        "attention_mask": attention_mask,
        "position_ids": torch.arange(total_len).reshape(1, total_len),
        "old_log_probs": old_log_probs,
        "advantages": torch.ones_like(old_log_probs),
        "delta": delta_tensor,
    }
    non_tensors = {"uid": np.array([query_id], dtype=object)}
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def _make_actor_batch(lengths: list[int], replay_flags: list[bool], response_len: int = 3) -> DataProto:
    max_total_len = max(lengths)
    batch_size = len(lengths)
    attention_mask = torch.zeros(batch_size, max_total_len, dtype=torch.long)
    for idx, seq_len in enumerate(lengths):
        attention_mask[idx, :seq_len] = 1

    tensors = {
        "responses": torch.arange(batch_size * response_len).reshape(batch_size, response_len),
        "response_mask": torch.ones(batch_size, response_len, dtype=torch.long),
        "input_ids": torch.arange(batch_size * max_total_len).reshape(batch_size, max_total_len),
        "attention_mask": attention_mask,
        "position_ids": torch.arange(max_total_len).repeat(batch_size, 1),
        "old_log_probs": torch.zeros(batch_size, response_len),
        "advantages": torch.ones(batch_size, response_len),
        M2_REPLAY_SOURCE_KEY: torch.tensor(replay_flags, dtype=torch.bool),
    }
    non_tensors = {
        "uid": np.array([f"q{idx}" for idx in range(batch_size)], dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def test_reorder_group_data_for_logprob_eval_restores_original_outputs():
    group_data_list = [
        _make_group_data("q1", prompt_len=9, delta=0.1),
        _make_group_data("q2", prompt_len=1, delta=0.2),
        _make_group_data("q3", prompt_len=8, delta=0.3),
        _make_group_data("q4", prompt_len=2, delta=0.4),
    ]

    eval_order = build_balanced_group_eval_order(group_data_list)
    assert eval_order != list(range(len(group_data_list)))

    reordered_group_data, eval_order_again = reorder_group_data_for_logprob_eval(group_data_list)
    assert eval_order_again == eval_order

    reordered_outputs = [
        group_data.batch["old_log_probs"] + group_data.batch["delta"] for group_data in reordered_group_data
    ]
    restored_outputs = restore_group_output_order(reordered_outputs, eval_order)

    for original_group, restored in zip(group_data_list, restored_outputs, strict=True):
        expected = original_group.batch["old_log_probs"] + original_group.batch["delta"]
        assert torch.equal(restored, expected)


def test_rebalance_batch_by_attention_preserves_source_flags_and_reduces_spread():
    actor_batch = _make_actor_batch(
        lengths=[12, 12, 12, 12, 4, 4, 4, 4],
        replay_flags=[True, True, False, False, True, False, False, False],
    )

    before_flags = actor_batch.batch[M2_REPLAY_SOURCE_KEY].clone()
    before_token_count = int(actor_batch.batch["attention_mask"].sum().item())

    metrics = rebalance_batch_by_attention(actor_batch, k_partitions=2, logging_prefix="actor_batch_seqlen")

    assert metrics["actor_batch_seqlen/applied"] == 1.0
    balanced_diff = metrics["actor_batch_seqlen/balanced_max"] - metrics["actor_batch_seqlen/balanced_min"]
    assert balanced_diff <= metrics["actor_batch_seqlen/minmax_diff"]
    assert int(actor_batch.batch["attention_mask"].sum().item()) == before_token_count
    assert int(actor_batch.batch[M2_REPLAY_SOURCE_KEY].sum().item()) == int(before_flags.sum().item())


def test_compute_actor_batch_metrics_reports_replay_and_onpolicy_splits():
    actor_batch = _make_actor_batch(
        lengths=[12, 9, 6, 4],
        replay_flags=[True, True, False, False],
    )

    metrics = compute_actor_batch_metrics(actor_batch, prefix="actor_batch")

    assert metrics["actor_batch/groups"] == 4.0
    assert metrics["actor_batch/replay_groups"] == 2.0
    assert metrics["actor_batch/onpolicy_groups"] == 2.0
    assert metrics["actor_batch/replay_seq_frac"] == pytest.approx(0.5, rel=1e-6)
    assert metrics["actor_batch/token_count"] == pytest.approx(float(sum([12, 9, 6, 4])), rel=1e-6)
    assert metrics["actor_batch/replay_token_count"] == pytest.approx(21.0, rel=1e-6)
    assert metrics["actor_batch/onpolicy_token_count"] == pytest.approx(10.0, rel=1e-6)
    assert metrics["actor_batch/replay_token_frac"] == pytest.approx(21.0 / 31.0, rel=1e-6)
    assert metrics["actor_batch/onpolicy_response_length_mean"] == pytest.approx(3.0, rel=1e-6)
    assert metrics["actor_batch/replay_response_length_mean"] == pytest.approx(3.0, rel=1e-6)
