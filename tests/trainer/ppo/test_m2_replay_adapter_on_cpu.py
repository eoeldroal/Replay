import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.m2_replay import QueryGroup
from verl.trainer.ppo.m2_replay_adapter import (
    build_actor_batch_with_replay,
    build_query_groups_from_onpolicy_batch,
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
