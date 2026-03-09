# Copyright 2026

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.m2_replay import QueryGroup
from verl.trainer.ppo.m2_replay_adapter import (
    M2_REPLAY_SOURCE_KEY,
    _attach_replay_source_flag,
    build_actor_batch_from_groups,
    build_actor_batch_with_replay,
    build_query_groups_from_onpolicy_batch,
    select_actor_training_view,
)


def _make_onpolicy_batch(*, include_advantages: bool = True) -> DataProto:
    batch_size = 6
    response_len = 4
    prompt_len = 5
    total_len = prompt_len + response_len

    tensors = {
        "responses": torch.arange(batch_size * response_len).reshape(batch_size, response_len),
        "response_mask": torch.ones(batch_size, response_len, dtype=torch.long),
        "input_ids": torch.arange(batch_size * total_len).reshape(batch_size, total_len),
        "attention_mask": torch.ones(batch_size, total_len, dtype=torch.long),
        "position_ids": torch.arange(total_len).repeat(batch_size, 1),
        "old_log_probs": torch.zeros(batch_size, response_len),
        "token_level_scores": torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    }
    if include_advantages:
        tensors["advantages"] = torch.ones(batch_size, response_len)

    non_tensors = {
        "uid": np.array(["q1", "q1", "q2", "q2", "q3", "q3"], dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def _assert_dataproto_equal(lhs: DataProto, rhs: DataProto) -> None:
    assert lhs.batch is not None
    assert rhs.batch is not None
    assert set(lhs.batch.keys()) == set(rhs.batch.keys())
    for key in lhs.batch.keys():
        assert torch.equal(lhs.batch[key], rhs.batch[key]), f"Tensor mismatch for key={key}"

    assert set(lhs.non_tensor_batch.keys()) == set(rhs.non_tensor_batch.keys())
    for key in lhs.non_tensor_batch.keys():
        assert lhs.non_tensor_batch[key].tolist() == rhs.non_tensor_batch[key].tolist(), (
            f"Non-tensor mismatch for key={key}"
        )


def _legacy_build_query_groups_from_onpolicy_batch(
    batch: DataProto,
    *,
    expected_group_size: int | None,
    insertion_step: int,
    success_count_min: int | None = None,
    success_count_max: int | None = None,
) -> tuple[list[QueryGroup], int, int, int, list[dict[str, object]]]:
    uid_array = batch.non_tensor_batch["uid"]
    order: list[str] = []
    indices: dict[str, list[int]] = {}
    for idx, uid in enumerate(uid_array.tolist()):
        uid_str = str(uid)
        if uid_str not in indices:
            indices[uid_str] = []
            order.append(uid_str)
        indices[uid_str].append(idx)

    groups: list[QueryGroup] = []
    skipped_incomplete = 0
    skipped_missing_train_keys = 0
    skipped_by_success_band = 0
    group_debug_all: list[dict[str, object]] = []
    accepted_candidates: list[tuple[str, list[int], float, int, int, dict[str, object]]] = []

    for uid in order:
        idxs = indices[uid]
        debug_entry: dict[str, object] = {"query_id": str(uid), "observed_group_size": int(len(idxs))}
        if expected_group_size is not None and expected_group_size > 0 and len(idxs) != expected_group_size:
            skipped_incomplete += 1
            debug_entry["status"] = "skipped_incomplete"
            group_debug_all.append(debug_entry)
            continue

        score = batch.batch["token_level_scores"][idxs]
        mask = batch.batch["response_mask"][idxs].float()
        sample_score = (score.float() * mask).sum(dim=-1)
        success = (sample_score > 0).float()
        success_prob = float(success.mean().item())
        success_count = int(success.sum().item())
        group_size = int(success.numel())
        debug_entry["success_prob"] = float(success_prob)
        debug_entry["success_count"] = int(success_count)
        debug_entry["group_size"] = int(group_size)

        if success_count_min is not None and success_count < int(success_count_min):
            skipped_by_success_band += 1
            debug_entry["status"] = "skipped_by_success_band"
            group_debug_all.append(debug_entry)
            continue
        if success_count_max is not None and success_count > int(success_count_max):
            skipped_by_success_band += 1
            debug_entry["status"] = "skipped_by_success_band"
            group_debug_all.append(debug_entry)
            continue
        accepted_candidates.append((str(uid), idxs, success_prob, success_count, group_size, debug_entry))

    required_keys = {"old_log_probs", "advantages", "response_mask"}
    for uid, idxs, success_prob, success_count, group_size, debug_entry in accepted_candidates:
        raw_group = batch[idxs]
        actor_group = select_actor_training_view(raw_group)
        if actor_group.batch is None or not required_keys.issubset(set(actor_group.batch.keys())):
            skipped_missing_train_keys += 1
            debug_entry["status"] = "skipped_missing_train_keys"
            group_debug_all.append(debug_entry)
            continue
        groups.append(
            QueryGroup(
                query_id=uid,
                data=actor_group,
                success_prob=success_prob,
                insertion_step=int(insertion_step),
                success_count=int(success_count),
                group_size=int(group_size),
                last_training_step=int(insertion_step),
            )
        )
        debug_entry["status"] = "accepted"
        group_debug_all.append(debug_entry)

    return groups, skipped_incomplete, skipped_missing_train_keys, skipped_by_success_band, group_debug_all


def _legacy_build_actor_batch_with_replay(onpolicy_batch: DataProto, replay_groups: list[QueryGroup]) -> DataProto:
    onpolicy_actor_batch = _attach_replay_source_flag(select_actor_training_view(onpolicy_batch), is_replay=False)
    if not replay_groups:
        return onpolicy_actor_batch

    replay_batch = _attach_replay_source_flag(DataProto.concat([group.data for group in replay_groups]), is_replay=True)
    return DataProto.concat([replay_batch, onpolicy_actor_batch])


def _legacy_build_actor_batch_from_groups(
    onpolicy_groups: list[QueryGroup],
    replay_groups: list[QueryGroup],
) -> DataProto | None:
    chunks: list[DataProto] = []
    if replay_groups:
        replay_batch = _attach_replay_source_flag(DataProto.concat([group.data for group in replay_groups]), is_replay=True)
        chunks.append(replay_batch)
    if onpolicy_groups:
        onpolicy_batch = _attach_replay_source_flag(
            DataProto.concat([group.data for group in onpolicy_groups]),
            is_replay=False,
        )
        chunks.append(onpolicy_batch)
    if not chunks:
        return None
    if len(chunks) == 1:
        return chunks[0]
    return DataProto.concat(chunks)


def test_build_query_groups_fastpath_matches_legacy_output():
    batch = _make_onpolicy_batch()

    new_result = build_query_groups_from_onpolicy_batch(
        batch=batch,
        expected_group_size=2,
        insertion_step=17,
        success_count_min=0,
        success_count_max=2,
        compute_zvp_stats=False,
    )
    legacy_groups, skipped_incomplete, skipped_missing, skipped_by_band, debug_all = _legacy_build_query_groups_from_onpolicy_batch(
        batch=batch,
        expected_group_size=2,
        insertion_step=17,
        success_count_min=0,
        success_count_max=2,
    )

    assert new_result.skipped_incomplete == skipped_incomplete
    assert new_result.skipped_missing_train_keys == skipped_missing
    assert new_result.skipped_by_success_band == skipped_by_band
    assert len(new_result.groups) == len(legacy_groups)
    assert [group["status"] for group in new_result.group_debug_all] == [group["status"] for group in debug_all]

    for new_group, legacy_group in zip(new_result.groups, legacy_groups, strict=True):
        assert new_group.query_id == legacy_group.query_id
        assert new_group.success_prob == legacy_group.success_prob
        assert new_group.success_count == legacy_group.success_count
        assert new_group.group_size == legacy_group.group_size
        _assert_dataproto_equal(new_group.data, legacy_group.data)


def test_build_query_groups_fastpath_matches_legacy_when_actor_keys_missing():
    batch = _make_onpolicy_batch(include_advantages=False)

    new_result = build_query_groups_from_onpolicy_batch(
        batch=batch,
        expected_group_size=2,
        insertion_step=5,
        compute_zvp_stats=False,
    )
    legacy_groups, skipped_incomplete, skipped_missing, skipped_by_band, debug_all = _legacy_build_query_groups_from_onpolicy_batch(
        batch=batch,
        expected_group_size=2,
        insertion_step=5,
    )

    assert len(new_result.groups) == len(legacy_groups) == 0
    assert new_result.skipped_incomplete == skipped_incomplete
    assert new_result.skipped_missing_train_keys == skipped_missing
    assert new_result.skipped_by_success_band == skipped_by_band
    assert [group["status"] for group in new_result.group_debug_all] == [group["status"] for group in debug_all]


def test_build_actor_batch_with_replay_one_shot_concat_matches_legacy():
    onpolicy_batch = _make_onpolicy_batch()
    group_result = build_query_groups_from_onpolicy_batch(
        batch=onpolicy_batch,
        expected_group_size=2,
        insertion_step=1,
        compute_zvp_stats=False,
    )

    replay_groups = [group_result.groups[0], group_result.groups[1]]
    merged = build_actor_batch_with_replay(onpolicy_batch=onpolicy_batch, replay_groups=replay_groups)
    legacy = _legacy_build_actor_batch_with_replay(onpolicy_batch=onpolicy_batch, replay_groups=replay_groups)

    _assert_dataproto_equal(merged, legacy)
    assert M2_REPLAY_SOURCE_KEY in merged.batch.keys()


def test_fullbatch_zvp_matches_per_group_zvp():
    """compute_zvp_stats_from_full_batch must give same zvp_score as update_query_groups_zvp_stats."""
    import torch
    import numpy as np
    from verl import DataProto
    from verl.trainer.ppo.m2_replay import (
        update_query_groups_zvp_stats,
        compute_zvp_stats_from_full_batch,  # to be implemented
    )
    from verl.trainer.ppo.m2_replay_adapter import build_query_groups_from_onpolicy_batch

    torch.manual_seed(42)
    n_groups = 8
    group_size = 4
    seq_len = 32
    total_rows = n_groups * group_size

    old_log_probs = torch.randn(total_rows, seq_len)
    advantages = torch.randn(total_rows, seq_len)
    response_mask = (torch.rand(total_rows, seq_len) > 0.3).bool()
    uid_arr = np.repeat(np.arange(n_groups), group_size).astype(str)

    batch = DataProto.from_dict(
        tensors={
            "old_log_probs": old_log_probs,
            "advantages": advantages,
            "response_mask": response_mask,
            "responses": torch.zeros(total_rows, seq_len, dtype=torch.long),
            "input_ids": torch.zeros(total_rows, seq_len, dtype=torch.long),
            "attention_mask": response_mask.long(),
            "position_ids": torch.zeros(total_rows, seq_len, dtype=torch.long),
        },
        non_tensors={"uid": uid_arr},
    )

    # Method 1: build groups then update ZVP per-group (existing path)
    result = build_query_groups_from_onpolicy_batch(
        batch=batch,
        expected_group_size=group_size,
        insertion_step=1,
        compute_zvp_stats=False,
    )
    groups_ref = result.groups
    update_query_groups_zvp_stats(groups_ref, lambda_neg=1.0, zvp_mode="sign_only", update_step=1)
    scores_ref = [g.zvp_score for g in groups_ref]

    # Method 2: full-batch fast path (new function)
    result2 = build_query_groups_from_onpolicy_batch(
        batch=batch,
        expected_group_size=group_size,
        insertion_step=1,
        compute_zvp_stats=False,
    )
    groups_new = result2.groups
    compute_zvp_stats_from_full_batch(
        groups=groups_new,
        batch=batch,
        group_size=group_size,
        zvp_mode="sign_only",
        lambda_neg=1.0,
        update_step=1,
    )
    scores_new = [g.zvp_score for g in groups_new]

    for i, (ref, new) in enumerate(zip(scores_ref, scores_new)):
        assert abs(ref - new) < 1e-5, f"Group {i}: ref={ref:.6f}, new={new:.6f}"


def test_build_actor_batch_from_groups_one_shot_concat_matches_legacy():
    onpolicy_batch = _make_onpolicy_batch()
    group_result = build_query_groups_from_onpolicy_batch(
        batch=onpolicy_batch,
        expected_group_size=2,
        insertion_step=3,
        compute_zvp_stats=False,
    )

    replay_groups = [group_result.groups[0]]
    onpolicy_groups = group_result.groups[1:]
    merged = build_actor_batch_from_groups(onpolicy_groups=onpolicy_groups, replay_groups=replay_groups)
    legacy = _legacy_build_actor_batch_from_groups(onpolicy_groups=onpolicy_groups, replay_groups=replay_groups)

    assert merged is not None
    assert legacy is not None
    _assert_dataproto_equal(merged, legacy)
