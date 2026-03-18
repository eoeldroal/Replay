# Copyright 2026

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.m2_replay import (
    QueryGroup,
    normalize_zvp_mode,
    update_query_groups_zvp_stats,
    compute_zvp_stats_from_full_batch,
)
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance

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
    skipped_by_tool_pair_fraction: int
    group_debug_all: list[dict[str, object]]

    def to_metrics(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}/buffer/new_groups": float(len(self.groups)),
            f"{prefix}/buffer/skipped_incomplete": float(self.skipped_incomplete),
            f"{prefix}/buffer/skipped_missing_train_keys": float(self.skipped_missing_train_keys),
            f"{prefix}/buffer/skipped_by_success_band": float(self.skipped_by_success_band),
            f"{prefix}/buffer/skipped_by_tool_pair_fraction": float(self.skipped_by_tool_pair_fraction),
        }


def _flatten_attention_lengths(data: DataProto) -> torch.Tensor:
    if data.batch is None or "attention_mask" not in data.batch.keys():
        return torch.empty(0, dtype=torch.long)
    attention_mask = data.batch["attention_mask"]
    return attention_mask.reshape(attention_mask.shape[0], -1).sum(dim=-1).long()


def _compute_group_total_workload(data: DataProto) -> int:
    seqlen = _flatten_attention_lengths(data)
    if seqlen.numel() == 0:
        return 0
    return int(calculate_workload(seqlen).sum().item())


def build_balanced_group_eval_order(group_data_list: list[DataProto]) -> list[int]:
    if len(group_data_list) <= 2:
        return list(range(len(group_data_list)))

    workloads = [_compute_group_total_workload(group_data) for group_data in group_data_list]
    if not any(workloads):
        return list(range(len(group_data_list)))

    order = sorted(range(len(group_data_list)), key=lambda idx: (workloads[idx], idx))
    return order[::2] + order[1::2][::-1]


def reorder_group_data_for_logprob_eval(group_data_list: list[DataProto]) -> tuple[list[DataProto], list[int]]:
    eval_order = build_balanced_group_eval_order(group_data_list)
    reordered = [group_data_list[idx] for idx in eval_order]
    return reordered, eval_order


def restore_group_output_order(outputs: list[torch.Tensor], eval_order: list[int]) -> list[torch.Tensor]:
    if len(outputs) != len(eval_order):
        raise ValueError(
            "restore_group_output_order requires one output per reordered group, "
            f"got {len(outputs)=} and {len(eval_order)=}."
        )

    restored: list[Optional[torch.Tensor]] = [None] * len(outputs)
    for reordered_idx, original_idx in enumerate(eval_order):
        restored[original_idx] = outputs[reordered_idx]

    if any(tensor is None for tensor in restored):
        raise ValueError("restore_group_output_order failed to reconstruct the full original group order.")

    return [tensor for tensor in restored if tensor is not None]


def rebalance_batch_by_attention(
    batch: DataProto,
    *,
    k_partitions: int,
    logging_prefix: str,
) -> dict[str, float]:
    if batch.batch is None or "attention_mask" not in batch.batch.keys():
        return {f"{logging_prefix}/applied": 0.0}

    batch_size = int(batch.batch["attention_mask"].shape[0])
    k_partitions = int(k_partitions)
    if k_partitions <= 1 or batch_size < k_partitions or batch_size % k_partitions != 0:
        return {
            f"{logging_prefix}/applied": 0.0,
            f"{logging_prefix}/batch_size": float(batch_size),
            f"{logging_prefix}/k_partitions": float(k_partitions),
        }

    global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)
    workload_lst = calculate_workload(global_seqlen_lst).tolist()
    global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=k_partitions, equal_size=True)

    for idx, partition in enumerate(global_partition_lst):
        partition.sort(key=lambda sample_idx: (workload_lst[sample_idx], sample_idx))
        ordered_partition = partition[::2] + partition[1::2][::-1]
        global_partition_lst[idx] = ordered_partition

    global_idx = torch.tensor([sample_idx for partition in global_partition_lst for sample_idx in partition])
    batch.reorder(global_idx)
    metrics = log_seqlen_unbalance(
        seqlen_list=global_seqlen_lst.tolist(),
        partitions=global_partition_lst,
        prefix=logging_prefix,
    )
    metrics[f"{logging_prefix}/applied"] = 1.0
    metrics[f"{logging_prefix}/batch_size"] = float(batch_size)
    metrics[f"{logging_prefix}/k_partitions"] = float(k_partitions)
    return metrics


def compute_actor_batch_metrics(batch: DataProto, prefix: str = "actor_batch") -> dict[str, float]:
    if batch.batch is None or "responses" not in batch.batch.keys() or "attention_mask" not in batch.batch.keys():
        return {}

    max_response_length = int(batch.batch["responses"].shape[-1])
    response_mask = batch.batch.get("response_mask", batch.batch["attention_mask"][:, -max_response_length:])
    response_length = response_mask.reshape(response_mask.shape[0], -1).sum(dim=-1).float()
    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length]
    prompt_length = prompt_mask.reshape(prompt_mask.shape[0], -1).sum(dim=-1).float()
    total_token_count = float(batch.batch["attention_mask"].sum().item())

    metrics = {
        f"{prefix}/groups": float(len(batch)),
        f"{prefix}/token_count": total_token_count,
        f"{prefix}/prompt_length_mean": float(prompt_length.mean().item()),
        f"{prefix}/response_length_mean": float(response_length.mean().item()),
        f"{prefix}/response_length_max": float(response_length.max().item()),
        f"{prefix}/response_length_min": float(response_length.min().item()),
        f"{prefix}/response_clip_ratio": float(torch.eq(response_length, max_response_length).float().mean().item()),
    }

    if M2_REPLAY_SOURCE_KEY not in batch.batch.keys():
        return metrics

    replay_mask = batch.batch[M2_REPLAY_SOURCE_KEY].bool()
    onpolicy_mask = ~replay_mask
    metrics[f"{prefix}/replay_groups"] = float(replay_mask.sum().item())
    metrics[f"{prefix}/onpolicy_groups"] = float(onpolicy_mask.sum().item())
    metrics[f"{prefix}/replay_seq_frac"] = float(replay_mask.float().mean().item())

    token_lengths = batch.batch["attention_mask"].reshape(batch.batch["attention_mask"].shape[0], -1).sum(dim=-1).float()
    replay_token_count = float(token_lengths[replay_mask].sum().item()) if torch.any(replay_mask) else 0.0
    onpolicy_token_count = float(token_lengths[onpolicy_mask].sum().item()) if torch.any(onpolicy_mask) else 0.0
    metrics[f"{prefix}/replay_token_count"] = replay_token_count
    metrics[f"{prefix}/onpolicy_token_count"] = onpolicy_token_count
    if total_token_count > 0:
        metrics[f"{prefix}/replay_token_frac"] = replay_token_count / total_token_count
        metrics[f"{prefix}/onpolicy_token_frac"] = onpolicy_token_count / total_token_count
    else:
        metrics[f"{prefix}/replay_token_frac"] = 0.0
        metrics[f"{prefix}/onpolicy_token_frac"] = 0.0

    if torch.any(replay_mask):
        metrics[f"{prefix}/replay_response_length_mean"] = float(response_length[replay_mask].mean().item())
    else:
        metrics[f"{prefix}/replay_response_length_mean"] = 0.0

    if torch.any(onpolicy_mask):
        metrics[f"{prefix}/onpolicy_response_length_mean"] = float(response_length[onpolicy_mask].mean().item())
    else:
        metrics[f"{prefix}/onpolicy_response_length_mean"] = 0.0

    return metrics


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


def _score_to_success(sample_score: torch.Tensor, success_score_threshold: Optional[float] = None) -> torch.Tensor:
    if success_score_threshold is None:
        return (sample_score > 0).float()
    return (sample_score >= float(success_score_threshold)).float()


def _compute_group_success_stats(
    group: DataProto,
    success_score_threshold: Optional[float] = None,
) -> tuple[float, int, int]:
    sample_score = _masked_sample_score(group)
    if sample_score is None or sample_score.numel() == 0:
        return (0.5, 0, 0)
    # RLVR binary success proxy.
    success = _score_to_success(sample_score, success_score_threshold=success_score_threshold)
    success_count = int(success.sum().item())
    group_size = int(success.numel())
    success_prob = float(success.mean().item())
    return (success_prob, success_count, group_size)

def _compute_group_success_stats_from_batch(
    batch: DataProto,
    idxs: list[int],
    success_score_threshold: Optional[float] = None,
) -> tuple[float, int, int]:
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

    success = _score_to_success(sample_score, success_score_threshold=success_score_threshold)
    success_count = int(success.sum().item())
    group_size = int(success.numel())
    success_prob = float(success.mean().item())
    return (success_prob, success_count, group_size)


def select_actor_training_view(data: DataProto) -> DataProto:
    if data.batch is None:
        raise ValueError("data.batch is None")

    batch_keys = [k for k in ACTOR_BATCH_KEYS if k in data.batch.keys()]
    batch_keys.extend([k for k in OPTIONAL_ACTOR_BATCH_KEYS if k in data.batch.keys()])
    non_tensor_keys = [
        k
        for k in ["uid", "multi_modal_inputs", "tool_call_counts", "tool_response_pair_counts"]
        if k in data.non_tensor_batch.keys()
    ]

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


def _attach_replay_source_flag_once(data: Optional[DataProto], replay_prefix_size: int) -> Optional[DataProto]:
    if data is None:
        return None
    if data.batch is None:
        return data

    replay_prefix_size = max(0, min(int(replay_prefix_size), len(data)))
    source_flag = torch.zeros((len(data),), dtype=torch.bool, device=data.batch.device)
    if replay_prefix_size > 0:
        source_flag[:replay_prefix_size] = True
    data.batch[M2_REPLAY_SOURCE_KEY] = source_flag
    return data


def _concat_dataprotos(chunks: list[DataProto]) -> Optional[DataProto]:
    if not chunks:
        return None

    prepared_chunks: list[DataProto] = []
    for chunk in chunks:
        if chunk.meta_info:
            prepared_chunks.append(type(chunk)(batch=chunk.batch, non_tensor_batch=chunk.non_tensor_batch, meta_info={}))
        else:
            prepared_chunks.append(chunk)
    return DataProto.concat(prepared_chunks)


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
    tool_response_pair_fraction_min: Optional[float] = None,
) -> tuple[list[QueryGroup], int, int]:
    normalized_mode = normalize_ingress_filter_mode(ingress_filter_mode)
    tool_pair_threshold = None
    if tool_response_pair_fraction_min is not None:
        tool_pair_threshold = max(float(tool_response_pair_fraction_min), 0.0)
        if tool_pair_threshold <= 0:
            tool_pair_threshold = None

    if normalized_mode == "none" and tool_pair_threshold is None:
        return list(groups), 0, 0

    max_success_count = 0
    if normalized_mode == "rlvr_halfband":
        max_success_count = int(rollout_n) // 2
    elif normalized_mode == "rlvr_non_degenerate":
        max_success_count = int(rollout_n) - 1

    if max_success_count < 1:
        return [], len(groups), 0

    filtered_groups = []
    skipped_groups = 0
    skipped_by_tool_pair_fraction = 0
    for group in groups:
        if normalized_mode != "none" and not (1 <= int(group.success_count) <= max_success_count):
            skipped_groups += 1
            continue
        if tool_pair_threshold is not None:
            pair_counts = group.data.non_tensor_batch.get("tool_response_pair_counts", None)
            if pair_counts is None:
                skipped_by_tool_pair_fraction += 1
                continue
            pair_counts = np.asarray(pair_counts)
            if pair_counts.size == 0:
                skipped_by_tool_pair_fraction += 1
                continue
            pair_fraction = float(np.mean(pair_counts.astype(np.float32) >= 1.0))
            if pair_fraction < tool_pair_threshold:
                skipped_by_tool_pair_fraction += 1
                continue
        filtered_groups.append(group)
    return filtered_groups, skipped_groups, skipped_by_tool_pair_fraction


def build_query_groups_from_onpolicy_batch(
    batch: DataProto,
    expected_group_size: Optional[int],
    insertion_step: int,
    success_count_min: Optional[int] = None,
    success_count_max: Optional[int] = None,
    success_score_threshold: Optional[float] = None,
    tool_response_pair_fraction_min: Optional[float] = None,
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
            skipped_by_tool_pair_fraction=0,
            group_debug_all=[],
        )

    uid_groups = _group_indices_by_uid(batch.non_tensor_batch["uid"])

    groups: list[QueryGroup] = []
    skipped_incomplete = 0
    skipped_missing_train_keys = 0
    skipped_by_success_band = 0
    skipped_by_tool_pair_fraction = 0
    group_debug_all: list[dict[str, object]] = []
    normalized_zvp_mode = normalize_zvp_mode(zvp_mode)
    accepted_candidates: list[tuple[str, list[int], float, int, int, dict[str, object]]] = []
    tool_pair_threshold = None
    if tool_response_pair_fraction_min is not None:
        tool_pair_threshold = max(float(tool_response_pair_fraction_min), 0.0)
        if tool_pair_threshold <= 0:
            tool_pair_threshold = None

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

        success_prob, success_count, group_size = _compute_group_success_stats_from_batch(
            batch=batch,
            idxs=idxs,
            success_score_threshold=success_score_threshold,
        )
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
        if tool_pair_threshold is not None:
            pair_counts = batch.non_tensor_batch.get("tool_response_pair_counts", None)
            if pair_counts is None:
                skipped_by_tool_pair_fraction += 1
                debug_entry["status"] = "skipped_by_tool_pair_fraction"
                group_debug_all.append(debug_entry)
                continue
            pair_counts = np.asarray(pair_counts)[idxs]
            pair_fraction = float(np.mean(pair_counts.astype(np.float32) >= 1.0)) if len(pair_counts) > 0 else 0.0
            debug_entry["tool_response_pair_fraction"] = pair_fraction
            if pair_fraction < tool_pair_threshold:
                skipped_by_tool_pair_fraction += 1
                debug_entry["status"] = "skipped_by_tool_pair_fraction"
                group_debug_all.append(debug_entry)
                continue
        accepted_candidates.append((str(uid), idxs, success_prob, success_count, group_size, debug_entry))

    required_keys = {"old_log_probs", "advantages", "response_mask"}
    if accepted_candidates:
        actor_view_batch = select_actor_training_view(batch)
        actor_view_keys = set(actor_view_batch.batch.keys()) if actor_view_batch.batch is not None else set()
        if actor_view_batch.batch is None or not required_keys.issubset(actor_view_keys):
            skipped_missing_train_keys += len(accepted_candidates)
            for _, _, _, _, _, debug_entry in accepted_candidates:
                debug_entry["status"] = "skipped_missing_train_keys"
                group_debug_all.append(debug_entry)
        else:
            for uid, idxs, success_prob, success_count, group_size, debug_entry in accepted_candidates:
                actor_group = actor_view_batch[idxs]
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
        if expected_group_size is not None and expected_group_size > 0:
            # Fast path: one vectorized pass over a contiguous group-ordered batch (no re-stacking).
            # INVARIANT: groups are built from accepted_candidates in uid-first-seen order.
            # Each group.data is actor_view_batch[idxs], so concatenating them produces a batch
            # where group i covers rows [i*expected_group_size : (i+1)*expected_group_size].
            ordered_batch = _concat_dataprotos([g.data for g in groups])
            compute_zvp_stats_from_full_batch(
                groups=groups,
                batch=ordered_batch,
                group_size=expected_group_size,
                zvp_mode=normalized_zvp_mode,
                lambda_neg=zvp_lambda_neg,
                update_step=int(insertion_step),
            )
        else:
            # Fallback: per-group stacking (non-uniform group sizes).
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
        skipped_by_tool_pair_fraction=skipped_by_tool_pair_fraction,
        group_debug_all=group_debug_all,
    )


def concat_query_groups(groups: list[QueryGroup]) -> Optional[DataProto]:
    if not groups:
        return None
    return _concat_dataprotos([group.data for group in groups])


def build_actor_batch_with_replay(onpolicy_batch: DataProto, replay_groups: list[QueryGroup]) -> DataProto:
    onpolicy_actor_batch = select_actor_training_view(onpolicy_batch)
    if not replay_groups:
        return _attach_replay_source_flag_once(onpolicy_actor_batch, replay_prefix_size=0)

    replay_chunks = [group.data for group in replay_groups]
    merged = _concat_dataprotos(replay_chunks + [onpolicy_actor_batch])
    replay_prefix_size = sum(len(group.data) for group in replay_groups)
    return _attach_replay_source_flag_once(merged, replay_prefix_size=replay_prefix_size)


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
    replay_chunks = [group.data for group in replay_groups]
    onpolicy_chunks = [group.data for group in onpolicy_groups]
    merged = _concat_dataprotos(replay_chunks + onpolicy_chunks)
    if merged is None:
        return None
    replay_prefix_size = sum(len(group.data) for group in replay_groups)
    return _attach_replay_source_flag_once(merged, replay_prefix_size=replay_prefix_size)
