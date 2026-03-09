import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.m2_replay import QueryGroup, QueryGroupReplayBuffer, select_replay_groups


def _make_group(query_id: str, delta: float, insertion_step: int) -> QueryGroup:
    old = torch.zeros(2, 3, dtype=torch.float32)
    mask = torch.ones(2, 3, dtype=torch.float32)
    advantages = torch.tensor([[1.0, -1.0, 1.0], [1.0, -1.0, 1.0]], dtype=torch.float32)
    data = DataProto.from_dict(
        tensors={
            "old_log_probs": old,
            "response_mask": mask,
            "advantages": advantages,
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


def _compute_group_m2_batch(data_list: list[DataProto]) -> list[float]:
    m2_values: list[float] = []
    for data in data_list:
        old = data.batch["old_log_probs"].float()
        new = _compute_new(data).float()
        mask = data.batch["response_mask"].float()
        denom = mask.sum()
        if float(denom.item()) <= 0.0:
            m2_values.append(float("inf"))
            continue
        m2_values.append(float((((new - old).square() * mask).sum() / denom).item()))
    return m2_values


def test_gpu_m2_fastpath_matches_chunked_reference_selection():
    buffer = QueryGroupReplayBuffer(max_query_groups=16, seed=17)
    for i, delta in enumerate([0.0, 0.1, 0.2, 0.3, 0.6, 1.0], start=1):
        buffer.append_group(_make_group(f"q{i}", delta=delta, insertion_step=i))

    reference = select_replay_groups(
        buffer=buffer,
        target_groups=3,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        compute_new_log_probs_batch_fn=_compute_new_batch,
        max_scan=len(buffer),
        selection_mode="recency_only",
        groups_per_chunk=4,
        compute_runtime_zvp=False,
    )
    fastpath = select_replay_groups(
        buffer=buffer,
        target_groups=3,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        compute_new_log_probs_batch_fn=_compute_new_batch,
        compute_group_m2_batch_fn=_compute_group_m2_batch,
        max_scan=len(buffer),
        selection_mode="recency_only",
        groups_per_chunk=4,
        compute_runtime_zvp=False,
    )

    assert [g.query_id for g in fastpath.selected_groups] == [g.query_id for g in reference.selected_groups]
    assert fastpath.accepted_m2 == reference.accepted_m2
    assert fastpath.rejected_by_tau == reference.rejected_by_tau


def test_gpu_m2_fastpath_omits_runtime_zvp_debug_fields_when_disabled():
    buffer = QueryGroupReplayBuffer(max_query_groups=8, seed=3)
    for i, delta in enumerate([0.0, 0.1, 0.2], start=1):
        buffer.append_group(_make_group(f"q{i}", delta=delta, insertion_step=i))

    result = select_replay_groups(
        buffer=buffer,
        target_groups=2,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        compute_new_log_probs_batch_fn=_compute_new_batch,
        compute_group_m2_batch_fn=_compute_group_m2_batch,
        max_scan=len(buffer),
        selection_mode="recency_only",
        groups_per_chunk=4,
        compute_runtime_zvp=False,
    )

    assert len(result.selected_priority_debug) == len(result.selected_groups)
    for entry in result.selected_priority_debug:
        assert "zvp_runtime_score" not in entry
        assert "zvp_runtime_mean_surprisal" not in entry
        assert "zvp_runtime_neg_frac" not in entry
        assert "zvp_runtime_pos_frac" not in entry


def test_gpu_m2_fastpath_falls_back_to_chunked_new_log_prob_path_on_error():
    buffer = QueryGroupReplayBuffer(max_query_groups=16, seed=29)
    for i, delta in enumerate([0.0, 0.1, 0.2, 0.8], start=1):
        buffer.append_group(_make_group(f"q{i}", delta=delta, insertion_step=i))

    def _broken_group_m2_batch(_: list[DataProto]) -> list[float]:
        raise RuntimeError("intentional failure")

    fallback = select_replay_groups(
        buffer=buffer,
        target_groups=3,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        compute_new_log_probs_batch_fn=_compute_new_batch,
        compute_group_m2_batch_fn=_broken_group_m2_batch,
        max_scan=len(buffer),
        selection_mode="recency_only",
        groups_per_chunk=4,
        compute_runtime_zvp=False,
    )
    reference = select_replay_groups(
        buffer=buffer,
        target_groups=3,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        compute_new_log_probs_batch_fn=_compute_new_batch,
        max_scan=len(buffer),
        selection_mode="recency_only",
        groups_per_chunk=4,
        compute_runtime_zvp=False,
    )

    assert [g.query_id for g in fallback.selected_groups] == [g.query_id for g in reference.selected_groups]
    assert fallback.accepted_m2 == reference.accepted_m2
