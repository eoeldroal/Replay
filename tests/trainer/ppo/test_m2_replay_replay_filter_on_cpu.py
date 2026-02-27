import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.m2_replay import (
    QueryGroup,
    QueryGroupReplayBuffer,
    derive_micro_group_multiple,
    select_replay_groups,
)


def _make_group(query_id: str, insertion_step: int) -> QueryGroup:
    old = torch.zeros(2, 3, dtype=torch.float32)
    mask = torch.ones(2, 3, dtype=torch.float32)
    data = DataProto.from_dict(
        tensors={
            "old_log_probs": old,
            "response_mask": mask,
            "delta": torch.zeros_like(old),
        },
        non_tensors={"uid": np.array([query_id, query_id], dtype=object)},
    )
    return QueryGroup(
        query_id=query_id,
        data=data,
        success_prob=0.5,
        insertion_step=insertion_step,
    )


def test_select_replay_groups_does_not_filter_by_insertion_step():
    buffer = QueryGroupReplayBuffer(max_query_groups=8, seed=123)
    buffer.append_group(_make_group("old_q", insertion_step=10))
    buffer.append_group(_make_group("cur_q", insertion_step=11))

    def compute_new(data: DataProto) -> torch.Tensor:
        return data.batch["old_log_probs"] + data.batch["delta"]

    result = select_replay_groups(
        buffer=buffer,
        target_groups=2,
        tau=0.2,
        compute_new_log_probs_fn=compute_new,
        max_scan=len(buffer),
    )

    assert len(result.selected_groups) == 2
    assert set(g.query_id for g in result.selected_groups) == {"old_q", "cur_q"}
    assert result.scanned_groups == 2


def test_derive_micro_group_multiple_requires_per_gpu():
    assert derive_micro_group_multiple(micro_batch_size_per_gpu=None, n_gpus=8, rollout_n=8) == 1
