# Copyright 2026

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.m2_replay import QueryGroup, normalize_zvp_mode, update_query_groups_zvp_stats

# Minimal actor training keys.
ACTOR_BATCH_KEYS = [
    "responses",
    "response_mask",
    "input_ids",
    "attention_mask",
    "position_ids",
    "old_log_probs",
    "advantages",
]
OPTIONAL_ACTOR_BATCH_KEYS = [
    "ref_log_prob",
    "rollout_is_weights",
    "rollout_log_probs",
]

# Keys used only for building replay priority from on-policy samples.
_SCORE_CANDIDATE_KEYS = ["token_level_scores", "token_level_rewards"]
M2_REPLAY_SOURCE_KEY = "m2_replay_source"


@dataclass
class GroupBuildResult:
    groups: list[QueryGroup]
    skipped_incomplete: int
    skipped_missing_train_keys: int
    skipped_by_success_band: int
    group_debug_all: list[dict[str, object]]

    def to_metrics(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}/buffer/new_groups": float(len(self.groups)),
            f"{prefix}/buffer/skipped_incomplete": float(self.skipped_incomplete),
            f"{prefix}/buffer/skipped_missing_train_keys": float(self.skipped_missing_train_keys),
            f"{prefix}/buffer/skipped_by_success_band": float(self.skipped_by_success_band),
        }


def _group_indices_by_uid(uid_array: np.ndarray) -> list[tuple[str, list[int]]]:
    order: list[str] = []
    indices: dict[str, list[int]] = {}
    for idx, uid in enumerate(uid_array.tolist()):
        uid_str = str(uid)
        if uid_str not in indices:
            indices[uid_str] = []
            order.append(uid_str)
        indices[uid_str].append(idx)
    return [(uid, indices[uid]) for uid in order]


def _masked_sample_score(group: DataProto) -> Optional[torch.Tensor]:
    if group.batch is None:
        return None

    score_key = next((k for k in _SCORE_CANDIDATE_KEYS if k in group.batch.keys()), None)
    if score_key is None or "response_mask" not in group.batch.keys():
        return None

    score = group.batch[score_key]
    mask = group.batch["response_mask"].float()
    if score.dim() == 1:
        return score.float()
    return (score.float() * mask).sum(dim=-1)


def _compute_group_success_stats(group: DataProto) -> tuple[float, int, int]:
    sample_score = _masked_sample_score(group)
    if sample_score is None or sample_score.numel() == 0:
        return (0.5, 0, 0)
    # RLVR binary success proxy.
    success = (sample_score > 0).float()
    success_count = int(success.sum().item())
    group_size = int(success.numel())
    success_prob = float(success.mean().item())
    return (success_prob, success_count, group_size)


def _compute_group_success_stats_from_batch(batch: DataProto, idxs: list[int]) -> tuple[float, int, int]:
    if batch.batch is None:
        return (0.5, 0, 0)

    score_key = next((k for k in _SCORE_CANDIDATE_KEYS if k in batch.batch.keys()), None)
    if score_key is None or "response_mask" not in batch.batch.keys():
        return (0.5, 0, 0)

    score = batch.batch[score_key][idxs]
    mask = batch.batch["response_mask"][idxs].float()
    if score.dim() == 1:
        sample_score = score.float()
    else:
        sample_score = (score.float() * mask).sum(dim=-1)

    if sample_score.numel() == 0:
        return (0.5, 0, 0)

    success = (sample_score > 0).float()
    success_count = int(success.sum().item())
    group_size = int(success.numel())
    success_prob = float(success.mean().item())
    return (success_prob, success_count, group_size)


def select_actor_training_view(data: DataProto) -> DataProto:
    if data.batch is None:
        raise ValueError("data.batch is None")

    batch_keys = [k for k in ACTOR_BATCH_KEYS if k in data.batch.keys()]
    batch_keys.extend([k for k in OPTIONAL_ACTOR_BATCH_KEYS if k in data.batch.keys()])
    non_tensor_keys = [k for k in ["uid", "multi_modal_inputs"] if k in data.non_tensor_batch.keys()]

    view = data.select(batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_keys)
    view.meta_info = {}
    return view


def _attach_replay_source_flag(data: Optional[DataProto], is_replay: bool) -> Optional[DataProto]:
    if data is None:
        return None
    if data.batch is None:
        return data

    tensors = {key: value for key, value in data.batch.items()}
    tensors[M2_REPLAY_SOURCE_KEY] = torch.full(
        (len(data),),
        fill_value=bool(is_replay),
        dtype=torch.bool,
        device=data.batch.device,
    )
    non_tensors = {key: value.copy() for key, value in data.non_tensor_batch.items()}
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=dict(data.meta_info))


def normalize_ingress_filter_mode(ingress_filter_mode: str) -> str:
    normalized_mode = str(ingress_filter_mode).strip().lower()
    if normalized_mode not in {"none", "rlvr_halfband", "rlvr_non_degenerate"}:
        return "none"
    return normalized_mode


def derive_ingress_success_band(
    rollout_n: int,
    ingress_filter_mode: str,
) -> tuple[Optional[int], Optional[int]]:
    normalized_mode = normalize_ingress_filter_mode(ingress_filter_mode)
    if normalized_mode == "none":
        return (None, None)
    if normalized_mode == "rlvr_halfband":
        max_success_count = int(rollout_n) // 2
    else:
        max_success_count = int(rollout_n) - 1
    if max_success_count < 1:
        return (1, 0)
    return (1, max_success_count)


def filter_query_groups_for_ingress(
    groups: list[QueryGroup],
    rollout_n: int,
    ingress_filter_mode: str,
) -> tuple[list[QueryGroup], int]:
    normalized_mode = normalize_ingress_filter_mode(ingress_filter_mode)
    if normalized_mode == "none":
        return list(groups), 0

    max_success_count = 0
    if normalized_mode == "rlvr_halfband":
        max_success_count = int(rollout_n) // 2
    elif normalized_mode == "rlvr_non_degenerate":
        max_success_count = int(rollout_n) - 1

    if max_success_count < 1:
        return [], len(groups)

    filtered_groups = [
        group for group in groups if 1 <= int(group.success_count) <= max_success_count
    ]
    skipped_groups = len(groups) - len(filtered_groups)
    return filtered_groups, skipped_groups


def build_query_groups_from_onpolicy_batch(
    batch: DataProto,
    expected_group_size: Optional[int],
    insertion_step: int,
    success_count_min: Optional[int] = None,
    success_count_max: Optional[int] = None,
    zvp_lambda_neg: float = 1.0,
    zvp_mode: str = "sign_only",
    compute_zvp_stats: bool = True,
    zvp_chunk_size: int = 64,
) -> GroupBuildResult:
    if "uid" not in batch.non_tensor_batch:
        return GroupBuildResult(
            groups=[],
            skipped_incomplete=0,
            skipped_missing_train_keys=0,
            skipped_by_success_band=0,
            group_debug_all=[],
        )

    uid_groups = _group_indices_by_uid(batch.non_tensor_batch["uid"])

    groups: list[QueryGroup] = []
    skipped_incomplete = 0
    skipped_missing_train_keys = 0
    skipped_by_success_band = 0
    group_debug_all: list[dict[str, object]] = []
    normalized_zvp_mode = normalize_zvp_mode(zvp_mode)
    accepted_candidates: list[tuple[str, list[int], float, int, int, dict[str, object]]] = []

    for uid, idxs in uid_groups:
        debug_entry: dict[str, object] = {
            "query_id": str(uid),
            "observed_group_size": int(len(idxs)),
        }
        if expected_group_size is not None and expected_group_size > 0 and len(idxs) != expected_group_size:
            skipped_incomplete += 1
            debug_entry["status"] = "skipped_incomplete"
            group_debug_all.append(debug_entry)
            continue

        success_prob, success_count, group_size = _compute_group_success_stats_from_batch(batch=batch, idxs=idxs)
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

    if compute_zvp_stats and groups:
        update_query_groups_zvp_stats(
            groups,
            lambda_neg=zvp_lambda_neg,
            zvp_mode=normalized_zvp_mode,
            update_step=int(insertion_step),
            chunk_size=int(zvp_chunk_size),
        )

    return GroupBuildResult(
        groups=groups,
        skipped_incomplete=skipped_incomplete,
        skipped_missing_train_keys=skipped_missing_train_keys,
        skipped_by_success_band=skipped_by_success_band,
        group_debug_all=group_debug_all,
    )


def concat_query_groups(groups: list[QueryGroup]) -> Optional[DataProto]:
    if not groups:
        return None
    sanitized = []
    for group in groups:
        dp = group.data
        dp.meta_info = {}
        sanitized.append(dp)
    return DataProto.concat(sanitized)


def build_actor_batch_with_replay(onpolicy_batch: DataProto, replay_groups: list[QueryGroup]) -> DataProto:
    onpolicy_actor_batch = _attach_replay_source_flag(select_actor_training_view(onpolicy_batch), is_replay=False)
    if not replay_groups:
        return onpolicy_actor_batch

    replay_batch = _attach_replay_source_flag(concat_query_groups(replay_groups), is_replay=True)
    if replay_batch is None:
        return onpolicy_actor_batch

    merged = DataProto.concat([replay_batch, onpolicy_actor_batch])
    return merged


def split_query_groups_by_adv_zero(
    groups: list[QueryGroup],
    eps: float = 1e-8,
) -> tuple[list[QueryGroup], list[QueryGroup]]:
    nonzero_groups: list[QueryGroup] = []
    zero_groups: list[QueryGroup] = []

    threshold = max(float(eps), 0.0)
    for group in groups:
        if group.data.batch is None or "advantages" not in group.data.batch.keys():
            nonzero_groups.append(group)
            continue

        adv = group.data.batch["advantages"].float()
        if "response_mask" in group.data.batch.keys():
            mask = group.data.batch["response_mask"].bool()
            if torch.any(mask):
                is_zero_group = bool(torch.max(torch.abs(adv[mask])) <= threshold)
            else:
                is_zero_group = True
        else:
            is_zero_group = bool(torch.max(torch.abs(adv)) <= threshold)

        if is_zero_group:
            zero_groups.append(group)
        else:
            nonzero_groups.append(group)

    return nonzero_groups, zero_groups


def build_actor_batch_from_groups(onpolicy_groups: list[QueryGroup], replay_groups: list[QueryGroup]) -> Optional[DataProto]:
    chunks: list[DataProto] = []

    replay_batch = _attach_replay_source_flag(concat_query_groups(replay_groups), is_replay=True)
    if replay_batch is not None:
        chunks.append(replay_batch)

    onpolicy_batch = _attach_replay_source_flag(concat_query_groups(onpolicy_groups), is_replay=False)
    if onpolicy_batch is not None:
        chunks.append(onpolicy_batch)

    if not chunks:
        return None
    if len(chunks) == 1:
        return chunks[0]
    return DataProto.concat(chunks)
