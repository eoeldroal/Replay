# Copyright 2026
"""Tests for ZVP update of M2-rejected (tau-exceeded) groups.

When compute_runtime_zvp=True and a group fails the tau threshold,
the already-computed new_log_probs should be used to update its ZVP
via rejected_by_tau_zvp_updates in ReplaySelectionResult.
"""

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.m2_replay import (
    QueryGroup,
    QueryGroupReplayBuffer,
    ReplaySelectionResult,
    compute_group_runtime_zvp_stats_batched,
    select_replay_groups,
)


def _make_group(
    query_id: str,
    delta: float,
    insertion_step: int,
    *,
    advantages: list[list[float]] | None = None,
) -> QueryGroup:
    old = torch.zeros(2, 3, dtype=torch.float32)
    mask = torch.ones(2, 3, dtype=torch.float32)
    tensors = {
        "old_log_probs": old,
        "response_mask": mask,
        "delta": torch.full_like(old, fill_value=delta),
    }
    if advantages is not None:
        tensors["advantages"] = torch.tensor(advantages, dtype=torch.float32)
    else:
        tensors["advantages"] = torch.tensor(
            [[1.0, -1.0, 0.5], [1.0, -1.0, 0.5]], dtype=torch.float32
        )
    data = DataProto.from_dict(
        tensors=tensors,
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


# ---------------------------------------------------------------------------
# 1. compute_runtime_zvp=False -> rejected_by_tau_zvp_updates is empty
# ---------------------------------------------------------------------------
def test_rejected_zvp_empty_when_compute_runtime_zvp_false():
    buffer = QueryGroupReplayBuffer(max_query_groups=8, seed=42)
    buffer.append_group(_make_group("pass", delta=0.0, insertion_step=1))
    buffer.append_group(_make_group("fail", delta=1.0, insertion_step=2))  # M2 >> tau

    result = select_replay_groups(
        buffer=buffer,
        target_groups=2,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        max_scan=len(buffer),
        selection_mode="recency_only",
        compute_runtime_zvp=False,
        groups_per_chunk=1,
    )
    assert result.rejected_by_tau == 1
    assert result.rejected_by_tau_zvp_updates == []


# ---------------------------------------------------------------------------
# 2. Serial path: rejected group gets ZVP update
# ---------------------------------------------------------------------------
def test_serial_path_rejected_group_gets_zvp_update():
    buffer = QueryGroupReplayBuffer(max_query_groups=8, seed=42)
    passing = _make_group("pass", delta=0.0, insertion_step=1)
    failing = _make_group("fail", delta=1.0, insertion_step=2)
    buffer.append_group(passing)
    buffer.append_group(failing)

    result = select_replay_groups(
        buffer=buffer,
        target_groups=2,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        max_scan=len(buffer),
        selection_mode="recency_only",
        compute_runtime_zvp=True,
        groups_per_chunk=1,  # serial path
    )

    assert result.rejected_by_tau == 1
    assert len(result.rejected_by_tau_zvp_updates) == 1
    group, (zvp_score, mean_surprisal, neg_frac, pos_frac) = result.rejected_by_tau_zvp_updates[0]
    assert group.query_id == "fail"
    assert zvp_score > 0.0


# ---------------------------------------------------------------------------
# 3. Serial path: ZVP values match reference computation
# ---------------------------------------------------------------------------
def test_serial_path_rejected_zvp_matches_reference():
    buffer = QueryGroupReplayBuffer(max_query_groups=8, seed=42)
    failing = _make_group("fail", delta=1.0, insertion_step=1)
    buffer.append_group(failing)

    result = select_replay_groups(
        buffer=buffer,
        target_groups=1,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        max_scan=len(buffer),
        selection_mode="recency_only",
        compute_runtime_zvp=True,
        groups_per_chunk=1,
    )

    assert len(result.rejected_by_tau_zvp_updates) == 1
    _, actual_stats = result.rejected_by_tau_zvp_updates[0]

    # Reference: compute directly
    new_log_probs = _compute_new(failing.data)
    ref_stats = compute_group_runtime_zvp_stats_batched(
        [failing], [new_log_probs], zvp_mode="sign_only",
    )[0]

    assert actual_stats[0] == pytest.approx(ref_stats[0], rel=1e-6)
    assert actual_stats[1] == pytest.approx(ref_stats[1], rel=1e-6)
    assert actual_stats[2] == pytest.approx(ref_stats[2], rel=1e-6)
    assert actual_stats[3] == pytest.approx(ref_stats[3], rel=1e-6)


# ---------------------------------------------------------------------------
# 4. Chunked path: rejected group gets ZVP update
# ---------------------------------------------------------------------------
def test_chunked_path_rejected_group_gets_zvp_update():
    buffer = QueryGroupReplayBuffer(max_query_groups=8, seed=42)
    buffer.append_group(_make_group("pass", delta=0.0, insertion_step=1))
    buffer.append_group(_make_group("fail", delta=1.0, insertion_step=2))

    result = select_replay_groups(
        buffer=buffer,
        target_groups=2,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        compute_new_log_probs_batch_fn=_compute_new_batch,
        max_scan=len(buffer),
        selection_mode="recency_only",
        compute_runtime_zvp=True,
        groups_per_chunk=4,  # chunked path
    )

    assert result.rejected_by_tau == 1
    assert len(result.rejected_by_tau_zvp_updates) == 1
    group, stats = result.rejected_by_tau_zvp_updates[0]
    assert group.query_id == "fail"
    assert stats[0] > 0.0


# ---------------------------------------------------------------------------
# 5. Chunked path matches serial path for rejected ZVP values
# ---------------------------------------------------------------------------
def test_chunked_path_rejected_zvp_matches_serial_path():
    buffer = QueryGroupReplayBuffer(max_query_groups=8, seed=42)
    buffer.append_group(_make_group("pass", delta=0.0, insertion_step=1))
    buffer.append_group(_make_group("fail1", delta=0.8, insertion_step=2))
    buffer.append_group(_make_group("fail2", delta=1.2, insertion_step=3))

    common_kwargs = dict(
        buffer=buffer,
        target_groups=3,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        max_scan=len(buffer),
        selection_mode="recency_only",
        compute_runtime_zvp=True,
    )

    serial = select_replay_groups(**common_kwargs, groups_per_chunk=1)
    chunked = select_replay_groups(
        **common_kwargs,
        compute_new_log_probs_batch_fn=_compute_new_batch,
        groups_per_chunk=4,
    )

    serial_by_qid = {g.query_id: s for g, s in serial.rejected_by_tau_zvp_updates}
    chunked_by_qid = {g.query_id: s for g, s in chunked.rejected_by_tau_zvp_updates}

    assert set(serial_by_qid.keys()) == set(chunked_by_qid.keys())
    for qid in serial_by_qid:
        for i in range(4):
            assert serial_by_qid[qid][i] == pytest.approx(chunked_by_qid[qid][i], rel=1e-6), (
                f"Mismatch for {qid} stat[{i}]"
            )


# ---------------------------------------------------------------------------
# 6. Accepted groups do NOT appear in rejected_by_tau_zvp_updates
# ---------------------------------------------------------------------------
def test_accepted_groups_not_in_rejected_zvp_updates():
    buffer = QueryGroupReplayBuffer(max_query_groups=8, seed=42)
    buffer.append_group(_make_group("pass1", delta=0.0, insertion_step=1))
    buffer.append_group(_make_group("pass2", delta=0.01, insertion_step=2))
    buffer.append_group(_make_group("fail", delta=1.0, insertion_step=3))

    result = select_replay_groups(
        buffer=buffer,
        target_groups=3,
        tau=0.5,
        compute_new_log_probs_fn=_compute_new,
        max_scan=len(buffer),
        selection_mode="recency_only",
        compute_runtime_zvp=True,
        groups_per_chunk=1,
    )

    accepted_ids = {g.query_id for g in result.selected_groups}
    rejected_zvp_ids = {g.query_id for g, _ in result.rejected_by_tau_zvp_updates}
    assert accepted_ids & rejected_zvp_ids == set()


# ---------------------------------------------------------------------------
# 7. Groups missing 'advantages' are excluded from rejected ZVP updates
# ---------------------------------------------------------------------------
def test_groups_missing_advantages_excluded():
    buffer = QueryGroupReplayBuffer(max_query_groups=8, seed=42)

    # Group without advantages
    old = torch.zeros(2, 3, dtype=torch.float32)
    mask = torch.ones(2, 3, dtype=torch.float32)
    no_adv_data = DataProto.from_dict(
        tensors={
            "old_log_probs": old,
            "response_mask": mask,
            "delta": torch.full_like(old, fill_value=1.0),
        },
        non_tensors={"uid": np.array(["no_adv", "no_adv"], dtype=object)},
    )
    no_adv_group = QueryGroup(
        query_id="no_adv", data=no_adv_data, success_prob=0.5, insertion_step=1,
    )
    buffer.append_group(no_adv_group)

    result = select_replay_groups(
        buffer=buffer,
        target_groups=1,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        max_scan=len(buffer),
        selection_mode="recency_only",
        compute_runtime_zvp=True,
        groups_per_chunk=1,
    )

    assert result.rejected_by_tau == 1
    assert result.rejected_by_tau_zvp_updates == []


# ---------------------------------------------------------------------------
# 8. Simulated EMA application: last_training_step NOT updated
# ---------------------------------------------------------------------------
def test_ema_does_not_update_last_training_step():
    """Simulate what ray_trainer.py should do: apply ZVP overwrite
    but NOT update last_training_step for rejected groups."""
    buffer = QueryGroupReplayBuffer(max_query_groups=8, seed=42)
    failing = _make_group("fail", delta=1.0, insertion_step=1)
    failing.last_training_step = 5
    buffer.append_group(failing)

    result = select_replay_groups(
        buffer=buffer,
        target_groups=1,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        max_scan=len(buffer),
        selection_mode="recency_only",
        compute_runtime_zvp=True,
        groups_per_chunk=1,
    )

    assert len(result.rejected_by_tau_zvp_updates) == 1
    group, (runtime_score, runtime_ms, runtime_nf, runtime_pf) = result.rejected_by_tau_zvp_updates[0]

    # Simulate ray_trainer.py: overwrite (alpha=1.0), do NOT touch last_training_step
    old_last_training_step = int(group.last_training_step)
    group.zvp_score = runtime_score
    group.zvp_mean_surprisal = runtime_ms
    group.zvp_neg_frac = runtime_nf
    group.zvp_pos_frac = runtime_pf
    group.zvp_update_count = int(group.zvp_update_count) + 1

    assert group.last_training_step == old_last_training_step  # unchanged
    assert group.zvp_update_count == 1
    assert group.zvp_score == pytest.approx(runtime_score)


# ---------------------------------------------------------------------------
# 9. zvp_update_count incremented after EMA application
# ---------------------------------------------------------------------------
def test_zvp_update_count_incremented():
    buffer = QueryGroupReplayBuffer(max_query_groups=8, seed=42)
    failing = _make_group("fail", delta=1.0, insertion_step=1)
    failing.zvp_update_count = 3  # had prior updates
    buffer.append_group(failing)

    result = select_replay_groups(
        buffer=buffer,
        target_groups=1,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        max_scan=len(buffer),
        selection_mode="recency_only",
        compute_runtime_zvp=True,
        groups_per_chunk=1,
    )

    assert len(result.rejected_by_tau_zvp_updates) == 1
    group, stats = result.rejected_by_tau_zvp_updates[0]
    # Simulate EMA application
    group.zvp_update_count = int(group.zvp_update_count) + 1
    assert group.zvp_update_count == 4


# ---------------------------------------------------------------------------
# 10. GPU fastpath (compute_runtime_zvp=False) -> no rejected ZVP
# ---------------------------------------------------------------------------
def test_gpu_fastpath_no_rejected_zvp_updates():
    buffer = QueryGroupReplayBuffer(max_query_groups=8, seed=42)
    buffer.append_group(_make_group("pass", delta=0.0, insertion_step=1))
    buffer.append_group(_make_group("fail", delta=1.0, insertion_step=2))

    def _compute_group_m2_batch(data_list: list[DataProto]) -> list[float]:
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
        target_groups=2,
        tau=0.05,
        compute_new_log_probs_fn=_compute_new,
        compute_new_log_probs_batch_fn=_compute_new_batch,
        compute_group_m2_batch_fn=_compute_group_m2_batch,
        max_scan=len(buffer),
        selection_mode="recency_only",
        compute_runtime_zvp=False,
        groups_per_chunk=4,
    )

    assert result.rejected_by_tau == 1
    assert result.rejected_by_tau_zvp_updates == []
