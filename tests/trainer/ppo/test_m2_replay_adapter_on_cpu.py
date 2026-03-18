import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.m2_replay import QueryGroup
from verl.trainer.ppo.m2_replay_adapter import (
    M2_REPLAY_SOURCE_KEY,
    build_actor_batch_with_replay,
    build_query_groups_from_onpolicy_batch,
    filter_query_groups_for_ingress,
)


def _make_onpolicy_batch() -> DataProto:
    batch_size = 4
    seq_len = 3

    tensors = {
        "responses": torch.arange(batch_size * seq_len).reshape(batch_size, seq_len),
        "response_mask": torch.ones(batch_size, seq_len),
        "input_ids": torch.arange(batch_size * seq_len).reshape(batch_size, seq_len),
        "attention_mask": torch.ones(batch_size, seq_len),
        "position_ids": torch.arange(seq_len).repeat(batch_size, 1),
        "old_log_probs": torch.zeros(batch_size, seq_len),
        "advantages": torch.ones(batch_size, seq_len),
        "token_level_scores": torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    }
    non_tensors = {
        "uid": np.array(["q1", "q1", "q2", "q2"], dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def test_build_query_groups_from_onpolicy_batch():
    batch = _make_onpolicy_batch()

    result = build_query_groups_from_onpolicy_batch(
        batch=batch,
        expected_group_size=2,
        insertion_step=42,
    )

    assert len(result.groups) == 2
    assert result.skipped_incomplete == 0
    assert result.skipped_missing_train_keys == 0

    group_map = {g.query_id: g for g in result.groups}
    assert group_map["q1"].success_prob == 1.0
    assert group_map["q2"].success_prob == 0.0


def test_build_actor_batch_with_replay_replay_first():
    onpolicy = _make_onpolicy_batch()
    replay_result = build_query_groups_from_onpolicy_batch(onpolicy, expected_group_size=2, insertion_step=1)
    replay_group = replay_result.groups[0]

    replay_group = QueryGroup(
        query_id=replay_group.query_id,
        data=replay_group.data,
        success_prob=replay_group.success_prob,
        insertion_step=replay_group.insertion_step,
    )

    merged = build_actor_batch_with_replay(onpolicy_batch=onpolicy, replay_groups=[replay_group])

    assert len(merged) == 6
    # replay-first: first two samples come from replay q1
    assert merged.non_tensor_batch["uid"][0] == "q1"
    assert merged.non_tensor_batch["uid"][1] == "q1"
    assert M2_REPLAY_SOURCE_KEY in merged.batch.keys()
    assert merged.batch[M2_REPLAY_SOURCE_KEY][:2].tolist() == [True, True]
    assert merged.batch[M2_REPLAY_SOURCE_KEY][2:].tolist() == [False, False, False, False]


def test_filter_query_groups_for_ingress_supports_non_degenerate_mode():
    groups = [
        QueryGroup(query_id=f"q{i}", data=_make_onpolicy_batch()[:1], success_prob=0.0, insertion_step=0, success_count=i)
        for i in range(5)
    ]

    halfband_groups, halfband_skipped, halfband_tool_skipped = filter_query_groups_for_ingress(
        groups=groups,
        rollout_n=4,
        ingress_filter_mode="rlvr_halfband",
    )
    non_degenerate_groups, non_degenerate_skipped, non_degenerate_tool_skipped = filter_query_groups_for_ingress(
        groups=groups,
        rollout_n=4,
        ingress_filter_mode="rlvr_non_degenerate",
    )

    assert [group.success_count for group in halfband_groups] == [1, 2]
    assert halfband_skipped == 3
    assert halfband_tool_skipped == 0
    assert [group.success_count for group in non_degenerate_groups] == [1, 2, 3]
    assert non_degenerate_skipped == 2
    assert non_degenerate_tool_skipped == 0


def test_build_query_groups_from_onpolicy_batch_supports_adv_magnitude_zvp():
    tensors = {
        "responses": torch.arange(6).reshape(2, 3),
        "response_mask": torch.ones(2, 3),
        "input_ids": torch.arange(6).reshape(2, 3),
        "attention_mask": torch.ones(2, 3),
        "position_ids": torch.arange(3).repeat(2, 1),
        "old_log_probs": torch.log(torch.tensor([[0.2, 0.9, 0.9], [0.2, 0.9, 0.9]], dtype=torch.float32)),
        "advantages": torch.tensor([[10.0, -1.0, -1.0], [10.0, -1.0, -1.0]], dtype=torch.float32),
        "token_level_scores": torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
    }
    non_tensors = {
        "uid": np.array(["q1", "q1"], dtype=object),
    }
    batch = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

    sign_only = build_query_groups_from_onpolicy_batch(
        batch=batch,
        expected_group_size=2,
        insertion_step=7,
        zvp_mode="sign_only",
    )
    adv_magnitude = build_query_groups_from_onpolicy_batch(
        batch=batch,
        expected_group_size=2,
        insertion_step=7,
        zvp_mode="adv_magnitude",
    )

    assert len(sign_only.groups) == 1
    assert len(adv_magnitude.groups) == 1
    assert sign_only.groups[0].zvp_score == pytest.approx((0.8 + 0.9 + 0.9) / 3.0, rel=1e-6)
    assert adv_magnitude.groups[0].zvp_score == pytest.approx((0.8 * 10.0 + 0.9 + 0.9) / 12.0, rel=1e-6)
    assert adv_magnitude.groups[0].zvp_score < sign_only.groups[0].zvp_score


def test_build_query_groups_from_onpolicy_batch_can_skip_initial_zvp_init():
    batch = _make_onpolicy_batch()

    result = build_query_groups_from_onpolicy_batch(
        batch=batch,
        expected_group_size=2,
        insertion_step=42,
        compute_zvp_stats=False,
    )

    assert len(result.groups) == 2
    assert all(group.zvp_update_count == 0 for group in result.groups)
    assert all(group.zvp_score == 0.0 for group in result.groups)
