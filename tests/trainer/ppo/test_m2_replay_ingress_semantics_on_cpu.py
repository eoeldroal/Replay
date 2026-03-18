# Copyright 2026

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.m2_replay_adapter import build_query_groups_from_onpolicy_batch, filter_query_groups_for_ingress
from verl.utils.reward_score import geo3k


def _make_batch_for_group_scores(group_scores: dict[str, list[float]]) -> DataProto:
    rows = []
    uids = []
    for query_id, scores in group_scores.items():
        for score in scores:
            rows.append(float(score))
            uids.append(query_id)

    batch_size = len(rows)
    response_len = 2
    total_len = 4
    score_tensor = torch.tensor(rows, dtype=torch.float32).reshape(batch_size, 1).repeat(1, response_len)
    tensors = {
        "responses": torch.arange(batch_size * response_len).reshape(batch_size, response_len),
        "response_mask": torch.ones(batch_size, response_len, dtype=torch.long),
        "input_ids": torch.arange(batch_size * total_len).reshape(batch_size, total_len),
        "attention_mask": torch.ones(batch_size, total_len, dtype=torch.long),
        "position_ids": torch.arange(total_len).repeat(batch_size, 1),
        "old_log_probs": torch.zeros(batch_size, response_len, dtype=torch.float32),
        "advantages": torch.ones(batch_size, response_len, dtype=torch.float32),
        "token_level_scores": score_tensor,
    }
    non_tensors = {"uid": np.array(uids, dtype=object)}
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def _make_batch_for_group_scores_and_pairs(
    group_scores: dict[str, list[float]],
    group_pair_counts: dict[str, list[int]],
) -> DataProto:
    rows = []
    pair_counts = []
    uids = []
    for query_id, scores in group_scores.items():
        counts = group_pair_counts[query_id]
        for score, pair_count in zip(scores, counts, strict=True):
            rows.append(float(score))
            pair_counts.append(int(pair_count))
            uids.append(query_id)

    batch_size = len(rows)
    response_len = 2
    total_len = 4
    score_tensor = torch.tensor(rows, dtype=torch.float32).reshape(batch_size, 1).repeat(1, response_len)
    tensors = {
        "responses": torch.arange(batch_size * response_len).reshape(batch_size, response_len),
        "response_mask": torch.ones(batch_size, response_len, dtype=torch.long),
        "input_ids": torch.arange(batch_size * total_len).reshape(batch_size, total_len),
        "attention_mask": torch.ones(batch_size, total_len, dtype=torch.long),
        "position_ids": torch.arange(total_len).repeat(batch_size, 1),
        "old_log_probs": torch.zeros(batch_size, response_len, dtype=torch.float32),
        "advantages": torch.ones(batch_size, response_len, dtype=torch.float32),
        "token_level_scores": score_tensor,
    }
    non_tensors = {
        "uid": np.array(uids, dtype=object),
        "tool_response_pair_counts": np.array(pair_counts, dtype=np.int32),
        "tool_call_counts": np.array(pair_counts, dtype=np.int32),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def test_geo3k_format_only_reward_is_positive(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(geo3k, "acc_reward", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(geo3k, "format_reward", lambda *args, **kwargs: 1.0)

    assert geo3k.compute_score("dummy", "dummy") == pytest.approx(0.1, rel=1e-6)


def test_non_degenerate_ingress_counts_any_positive_reward_as_success():
    batch = _make_batch_for_group_scores(
        {
            "all_zero": [0.0] * 8,
            "all_format_only": [0.1] * 8,
            "mixed_partial": [0.1, 0.1, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
        }
    )

    result = build_query_groups_from_onpolicy_batch(
        batch=batch,
        expected_group_size=8,
        insertion_step=3,
        compute_zvp_stats=False,
    )

    groups = {group.query_id: group for group in result.groups}
    assert groups["all_zero"].success_count == 0
    assert groups["all_format_only"].success_count == 8
    assert groups["mixed_partial"].success_count == 3

    filtered, skipped, skipped_tool = filter_query_groups_for_ingress(
        groups=list(groups.values()),
        rollout_n=8,
        ingress_filter_mode="rlvr_non_degenerate",
    )

    assert [group.query_id for group in filtered] == ["mixed_partial"]
    assert skipped == 2
    assert skipped_tool == 0


def test_tool_response_pair_fraction_ingress_filters_groups():
    batch = _make_batch_for_group_scores_and_pairs(
        {
            "low_pair": [0.1, 0.1, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
            "high_pair": [0.1, 0.1, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
        },
        {
            "low_pair": [1, 1, 1, 0, 0, 0, 0, 0],
            "high_pair": [1, 1, 1, 1, 1, 0, 0, 0],
        },
    )

    result = build_query_groups_from_onpolicy_batch(
        batch=batch,
        expected_group_size=8,
        insertion_step=3,
        tool_response_pair_fraction_min=0.5,
        compute_zvp_stats=False,
    )

    assert [group.query_id for group in result.groups] == ["high_pair"]
    assert result.skipped_by_tool_pair_fraction == 1
