# Copyright 2026

from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import torch

from verl import DataProto


@dataclass
class QueryGroup:
    query_id: str
    data: DataProto
    success_prob: float
    insertion_step: int
    success_count: int = -1
    group_size: int = -1
    last_training_step: int = -1
    insertion_order: int = -1
    # Cached ZVP statistics for replay prioritization.
    zvp_score: float = 0.0
    zvp_mean_surprisal: float = 0.0
    # Fraction of negative-advantage tokens within response tokens.
    zvp_neg_frac: float = 0.0
    # Deprecated legacy field retained for checkpoint compatibility.
    zvp_mean_pos_adv: float = 0.0
    zvp_pos_frac: float = 0.0
    zvp_score_num: float = 0.0
    zvp_score_denom: float = 0.0
    zvp_mean_surprisal_num: float = 0.0
    zvp_token_count: int = 0
    zvp_neg_count: int = 0
    zvp_pos_count: int = 0
    zvp_last_update_step: int = -1
    zvp_update_count: int = 0


@dataclass
class ReplaySelectionResult:
    selected_groups: list[QueryGroup]
    scanned_groups: int
    target_groups: int
    rejected_missing_fields: int
    rejected_by_tau: int
    rejected_eval_error: int
    accepted_m2: list[float] = field(default_factory=list)
    selected_priority_debug: list[dict[str, object]] = field(default_factory=list)
    candidate_priority_preview: list[dict[str, object]] = field(default_factory=list)
    candidate_priority_all: list[dict[str, object]] = field(default_factory=list)
    rejected_by_tau_zvp_updates: list[tuple["QueryGroup", tuple[float, float, float, float]]] = field(
        default_factory=list
    )

    def to_metrics(self, prefix: str) -> dict[str, float]:
        accepted = len(self.selected_groups)
        avg_m2 = float(sum(self.accepted_m2) / len(self.accepted_m2)) if self.accepted_m2 else 0.0
        return {
            f"{prefix}/pass1/scanned_groups": float(self.scanned_groups),
            f"{prefix}/pass1/accepted_groups": float(accepted),
            f"{prefix}/pass1/rejected_by_tau": float(self.rejected_by_tau),
            f"{prefix}/pass1/acceptance_rate": float(accepted / max(self.scanned_groups, 1)),
            f"{prefix}/pass1/accepted_m2_mean": avg_m2,
        }


class QueryGroupReplayBuffer:
    def __init__(self, max_query_groups: int, seed: int = 0):
        if max_query_groups <= 0:
            raise ValueError(f"max_query_groups must be > 0, got {max_query_groups}")
        self.max_query_groups = int(max_query_groups)
        self.seed = int(seed)
        self._groups: deque[QueryGroup] = deque()
        self._insertion_counter = 0

    def __len__(self) -> int:
        return len(self._groups)

    def items(self) -> list[QueryGroup]:
        return list(self._groups)

    def next_insertion_order(self) -> int:
        value = self._insertion_counter
        self._insertion_counter += 1
        return value

    def append_group(self, group: QueryGroup) -> None:
        if group.last_training_step < 0:
            group.last_training_step = int(group.insertion_step)
        if group.insertion_order < 0:
            group.insertion_order = self.next_insertion_order()
        if len(self._groups) >= self.max_query_groups:
            self._groups.popleft()
        self._groups.append(group)

    def append_groups(self, groups: Iterable[QueryGroup]) -> None:
        for group in groups:
            self.append_group(group)

    def state_dict(self) -> dict:
        return {
            "max_query_groups": self.max_query_groups,
            "seed": self.seed,
            "insertion_counter": self._insertion_counter,
            "groups": list(self._groups),
        }

    def load_state_dict(self, state: dict) -> None:
        self.max_query_groups = int(state.get("max_query_groups", self.max_query_groups))
        self.seed = int(state.get("seed", self.seed))
        self._insertion_counter = int(state.get("insertion_counter", 0))
        groups = state.get("groups", [])
        self._groups = deque(groups[-self.max_query_groups :])
        for group in self._groups:
            if not hasattr(group, "last_training_step"):
                group.last_training_step = int(getattr(group, "insertion_step", 0))
            if not hasattr(group, "success_count"):
                group.success_count = -1
            if not hasattr(group, "group_size"):
                group.group_size = -1
            if not hasattr(group, "zvp_score"):
                group.zvp_score = 0.0
            if not hasattr(group, "zvp_mean_surprisal"):
                group.zvp_mean_surprisal = 0.0
            if not hasattr(group, "zvp_neg_frac"):
                # Legacy snapshots may store sign-only third term in zvp_mean_pos_adv.
                group.zvp_neg_frac = float(getattr(group, "zvp_mean_pos_adv", 0.0))
            if not hasattr(group, "zvp_mean_pos_adv"):
                group.zvp_mean_pos_adv = 0.0
            if not hasattr(group, "zvp_pos_frac"):
                group.zvp_pos_frac = 0.0
            if not hasattr(group, "zvp_score_num"):
                group.zvp_score_num = 0.0
            if not hasattr(group, "zvp_score_denom"):
                group.zvp_score_denom = 0.0
            if not hasattr(group, "zvp_mean_surprisal_num"):
                group.zvp_mean_surprisal_num = 0.0
            if not hasattr(group, "zvp_token_count"):
                group.zvp_token_count = 0
            if not hasattr(group, "zvp_neg_count"):
                group.zvp_neg_count = 0
            if not hasattr(group, "zvp_pos_count"):
                group.zvp_pos_count = 0
            if not hasattr(group, "zvp_last_update_step"):
                group.zvp_last_update_step = int(getattr(group, "insertion_step", 0))
            if not hasattr(group, "zvp_update_count"):
                group.zvp_update_count = 0


def normalize_zvp_mode(zvp_mode: str) -> str:
    normalized_mode = str(zvp_mode).strip().lower()
    if normalized_mode not in {"sign_only", "adv_magnitude"}:
        return "sign_only"
    return normalized_mode


def compute_group_zvp_sufficient_stats_batched(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    adv_pos_eps: float = 1e-8,
    lambda_neg: float = 1.0,
    zvp_mode: str = "sign_only",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact sufficient statistics for batched ZVP computation."""
    if log_probs.shape != advantages.shape or log_probs.shape != response_mask.shape:
        raise ValueError(
            f"Shape mismatch in batched ZVP stats: {log_probs.shape=}, {advantages.shape=}, {response_mask.shape=}"
        )
    if log_probs.ndim < 2:
        raise ValueError(f"Batched ZVP expects ndim >= 2, got {log_probs.ndim}")

    num_groups = int(log_probs.shape[0])
    flat_log_probs = log_probs.float().reshape(num_groups, -1)
    flat_adv = advantages.float().reshape(num_groups, -1)
    flat_mask = response_mask.bool().reshape(num_groups, -1)

    mask_f = flat_mask.float()
    token_count = mask_f.sum(dim=1)
    surprisal = (-flat_log_probs).clamp_min(0.0)
    probs = torch.exp(-surprisal).clamp(0.0, 1.0)

    pos_mask = flat_mask & (flat_adv > float(adv_pos_eps))
    neg_mask = flat_mask & (flat_adv < -float(adv_pos_eps))
    pos_count = pos_mask.sum(dim=1).float()
    neg_count = neg_mask.sum(dim=1).float()

    pos_term = torch.where(pos_mask, 1.0 - probs, torch.zeros_like(probs))
    neg_term = torch.where(neg_mask, probs, torch.zeros_like(probs))
    token_score = pos_term + float(lambda_neg) * neg_term

    normalized_mode = normalize_zvp_mode(zvp_mode)
    if normalized_mode == "adv_magnitude":
        active_mask = pos_mask | neg_mask
        score_denom = torch.where(active_mask, flat_adv.abs(), torch.zeros_like(flat_adv)).sum(dim=1)
        score_num = (token_score * torch.where(active_mask, flat_adv.abs(), torch.zeros_like(flat_adv))).sum(dim=1)
    else:
        score_denom = token_count
        score_num = (token_score * mask_f).sum(dim=1)

    mean_surprisal_num = (surprisal * mask_f).sum(dim=1)
    return (score_num, score_denom, mean_surprisal_num, token_count, neg_count, pos_count)


def compute_group_zvp_stats_batched(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    adv_pos_eps: float = 1e-8,
    lambda_neg: float = 1.0,
    zvp_mode: str = "sign_only",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized ZVP statistics for a batch of groups.

    The first dimension is treated as group dimension; all remaining dimensions are
    flattened into the token dimension for each group.
    """
    score_num, score_denom, mean_surprisal_num, token_count, neg_count, pos_count = (
        compute_group_zvp_sufficient_stats_batched(
            log_probs=log_probs,
            advantages=advantages,
            response_mask=response_mask,
            adv_pos_eps=adv_pos_eps,
            lambda_neg=lambda_neg,
            zvp_mode=zvp_mode,
        )
    )
    valid = token_count > 0
    score = torch.where(
        score_denom > 0,
        score_num / score_denom.clamp_min(1e-12),
        torch.zeros_like(score_num),
    )
    mean_surprisal = torch.where(
        valid,
        mean_surprisal_num / token_count.clamp_min(1.0),
        torch.zeros_like(mean_surprisal_num),
    )
    neg_frac = torch.where(valid, neg_count / token_count.clamp_min(1.0), torch.zeros_like(neg_count))
    pos_frac = torch.where(valid, pos_count / token_count.clamp_min(1.0), torch.zeros_like(pos_count))
    return (score, mean_surprisal, neg_frac, pos_frac)


def compute_group_zvp_stats(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    adv_pos_eps: float = 1e-8,
    lambda_neg: float = 1.0,
    zvp_mode: str = "sign_only",
) -> tuple[float, float, float, float]:
    """Compute ZVP statistics from token log-probs and advantages.

    Supported modes:
    - sign_only: symmetric score that ignores advantage magnitude.
    - adv_magnitude: self-normalized |advantage|-weighted score over the same
      bounded token terms.
    """
    score, mean_surprisal, neg_frac, pos_frac = compute_group_zvp_stats_batched(
        log_probs=log_probs.unsqueeze(0),
        advantages=advantages.unsqueeze(0),
        response_mask=response_mask.unsqueeze(0),
        adv_pos_eps=adv_pos_eps,
        lambda_neg=lambda_neg,
        zvp_mode=zvp_mode,
    )
    return (
        float(score[0].item()),
        float(mean_surprisal[0].item()),
        float(neg_frac[0].item()),
        float(pos_frac[0].item()),
    )


def update_query_groups_zvp_stats(
    groups: Iterable[QueryGroup],
    *,
    adv_pos_eps: float = 1e-8,
    lambda_neg: float = 1.0,
    zvp_mode: str = "sign_only",
    update_step: Optional[int] = None,
    only_missing: bool = False,
    chunk_size: int = 64,
) -> int:
    """Refresh cached ZVP statistics for query groups in chunks."""
    required_keys = {"old_log_probs", "advantages", "response_mask"}
    candidates: list[QueryGroup] = []
    for group in groups:
        if only_missing and int(getattr(group, "zvp_update_count", 0)) > 0:
            continue
        batch = getattr(group.data, "batch", None)
        if batch is None or not required_keys.issubset(set(batch.keys())):
            continue
        candidates.append(group)

    if not candidates:
        return 0

    normalized_mode = normalize_zvp_mode(zvp_mode)
    chunk_size = max(int(chunk_size), 1)
    processed = 0
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        log_probs = torch.stack([group.data.batch["old_log_probs"] for group in chunk], dim=0)
        advantages = torch.stack([group.data.batch["advantages"] for group in chunk], dim=0)
        response_mask = torch.stack([group.data.batch["response_mask"] for group in chunk], dim=0)
        score_num, score_denom, mean_surprisal_num, token_count, neg_count, pos_count = (
            compute_group_zvp_sufficient_stats_batched(
                log_probs=log_probs,
                advantages=advantages,
                response_mask=response_mask,
                adv_pos_eps=adv_pos_eps,
                lambda_neg=lambda_neg,
                zvp_mode=normalized_mode,
            )
        )
        valid = token_count > 0
        scores = torch.where(
            score_denom > 0,
            score_num / score_denom.clamp_min(1e-12),
            torch.zeros_like(score_num),
        )
        mean_surprisal = torch.where(
            valid,
            mean_surprisal_num / token_count.clamp_min(1.0),
            torch.zeros_like(mean_surprisal_num),
        )
        neg_frac = torch.where(valid, neg_count / token_count.clamp_min(1.0), torch.zeros_like(neg_count))
        pos_frac = torch.where(valid, pos_count / token_count.clamp_min(1.0), torch.zeros_like(pos_count))
        summary = torch.stack(
            [
                scores,
                mean_surprisal,
                neg_frac,
                pos_frac,
                score_num,
                score_denom,
                mean_surprisal_num,
                token_count,
                neg_count,
                pos_count,
            ],
            dim=1,
        ).detach()
        summary_cpu = summary.cpu().tolist()
        for idx, group in enumerate(chunk):
            (
                group.zvp_score,
                group.zvp_mean_surprisal,
                group.zvp_neg_frac,
                group.zvp_pos_frac,
                group.zvp_score_num,
                group.zvp_score_denom,
                group.zvp_mean_surprisal_num,
                group_token_count,
                group_neg_count,
                group_pos_count,
            ) = summary_cpu[idx]
            group.zvp_token_count = int(group_token_count)
            group.zvp_neg_count = int(group_neg_count)
            group.zvp_pos_count = int(group_pos_count)
            if update_step is not None:
                group.zvp_last_update_step = int(update_step)
            elif int(getattr(group, "zvp_last_update_step", -1)) < 0:
                group.zvp_last_update_step = int(getattr(group, "insertion_step", 0))
            group.zvp_update_count = int(getattr(group, "zvp_update_count", 0)) + 1
        processed += len(chunk)
    return processed


def compute_zvp_stats_from_full_batch(
    groups: list["QueryGroup"],
    batch: "DataProto",
    group_size: int,
    zvp_mode: str,
    lambda_neg: float,
    adv_pos_eps: float = 1e-8,
    update_step: Optional[int] = None,
) -> None:
    """Compute ZVP sufficient stats for all groups from full batch in one vectorized pass.

    Eliminates 16x torch.stack re-stacking by using reshape on the already-contiguous
    actor_view_batch tensors.

    Mutates each group's ZVP fields in-place (same interface as update_query_groups_zvp_stats).
    Groups must be in the same order as rows in batch (i.e., group i covers rows
    [i*group_size : (i+1)*group_size]).
    """
    if not groups or group_size <= 0 or getattr(batch, "batch", None) is None:
        return

    _REQUIRED_KEYS = {"old_log_probs", "advantages", "response_mask"}
    if not _REQUIRED_KEYS.issubset(set(batch.batch.keys())):
        update_query_groups_zvp_stats(groups, lambda_neg=lambda_neg, zvp_mode=zvp_mode, update_step=update_step)
        return

    old_log_probs = batch.batch["old_log_probs"]   # (n_total_rows, seq_len)
    advantages = batch.batch["advantages"]           # (n_total_rows, seq_len)
    response_mask = batch.batch["response_mask"]     # (n_total_rows, seq_len)

    n_groups = len(groups)
    expected_rows = n_groups * group_size
    if int(old_log_probs.shape[0]) < expected_rows:
        print(
            f"[m2_replay] WARNING: compute_zvp_stats_from_full_batch: batch too small "
            f"({int(old_log_probs.shape[0])} rows, need {expected_rows}). "
            f"Falling back to per-group ZVP computation.",
            flush=True,
        )
        update_query_groups_zvp_stats(
            groups, lambda_neg=lambda_neg, zvp_mode=zvp_mode, adv_pos_eps=adv_pos_eps, update_step=update_step
        )
        return

    # INVARIANT: group i must cover rows [i*group_size : (i+1)*group_size] of `batch`.
    # If groups have been reordered (e.g., sorted by ZVP score) relative to batch rows,
    # stats will be silently mis-assigned. Only call this function immediately after
    # build_query_groups_from_onpolicy_batch() with the same batch, before any reordering.
    # Single reshape: (n_total_rows, seq_len) -> (n_groups, group_size, seq_len)
    # This is a view when tensor is contiguous — no data copy
    lp = old_log_probs[:expected_rows].reshape(n_groups, group_size, -1)
    adv = advantages[:expected_rows].reshape(n_groups, group_size, -1)
    mask = response_mask[:expected_rows].reshape(n_groups, group_size, -1)

    normalized_mode = normalize_zvp_mode(zvp_mode)

    # Single vectorized call replacing N per-group calls
    score_num, score_denom, mean_surprisal_num, token_count, neg_count, pos_count = (
        compute_group_zvp_sufficient_stats_batched(
            log_probs=lp,
            advantages=adv,
            response_mask=mask,
            adv_pos_eps=adv_pos_eps,
            lambda_neg=lambda_neg,
            zvp_mode=normalized_mode,
        )
    )  # each tensor has shape (n_groups,)

    valid = token_count > 0
    scores = torch.where(
        score_denom > 0,
        score_num / score_denom.clamp_min(1e-12),
        torch.zeros_like(score_num),
    )
    mean_surprisal = torch.where(
        valid,
        mean_surprisal_num / token_count.clamp_min(1.0),
        torch.zeros_like(mean_surprisal_num),
    )
    neg_frac = torch.where(valid, neg_count / token_count.clamp_min(1.0), torch.zeros_like(neg_count))
    pos_frac = torch.where(valid, pos_count / token_count.clamp_min(1.0), torch.zeros_like(pos_count))

    summary = torch.stack(
        [
            scores,
            mean_surprisal,
            neg_frac,
            pos_frac,
            score_num,
            score_denom,
            mean_surprisal_num,
            token_count,
            neg_count,
            pos_count,
        ],
        dim=1,
    ).detach()
    summary_cpu = summary.cpu().tolist()

    for idx, group in enumerate(groups):
        (
            group.zvp_score,
            group.zvp_mean_surprisal,
            group.zvp_neg_frac,
            group.zvp_pos_frac,
            group.zvp_score_num,
            group.zvp_score_denom,
            group.zvp_mean_surprisal_num,
            group_token_count,
            group_neg_count,
            group_pos_count,
        ) = summary_cpu[idx]
        group.zvp_token_count = int(group_token_count)
        group.zvp_neg_count = int(group_neg_count)
        group.zvp_pos_count = int(group_pos_count)
        if update_step is not None:
            group.zvp_last_update_step = int(update_step)
        elif int(getattr(group, "zvp_last_update_step", -1)) < 0:
            group.zvp_last_update_step = int(getattr(group, "insertion_step", 0))
        group.zvp_update_count = int(getattr(group, "zvp_update_count", 0)) + 1


def compute_group_m2_batched(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    if old_log_probs.shape != new_log_probs.shape:
        raise ValueError(f"Shape mismatch: {old_log_probs.shape=} vs {new_log_probs.shape=}")
    if response_mask.shape != old_log_probs.shape:
        raise ValueError(
            f"Mask shape mismatch: {response_mask.shape=} vs log_probs shape {old_log_probs.shape=}"
        )
    if old_log_probs.ndim < 2:
        raise ValueError(f"Batched M2 expects ndim >= 2, got {old_log_probs.ndim}")

    num_groups = int(old_log_probs.shape[0])
    old_flat = old_log_probs.float().reshape(num_groups, -1)
    new_flat = new_log_probs.float().reshape(num_groups, -1)
    mask_f = response_mask.float().reshape(num_groups, -1)
    denom = mask_f.sum(dim=1)
    ell = new_flat - old_flat
    m2_num = (ell.square() * mask_f).sum(dim=1)
    inf_value = torch.full_like(m2_num, float("inf"))
    return torch.where(denom > 0, m2_num / denom.clamp_min(1.0), inf_value)


def compute_group_runtime_zvp_stats_batched(
    groups: list[QueryGroup],
    new_log_probs_list: list[torch.Tensor],
    *,
    adv_pos_eps: float = 1e-8,
    lambda_neg: float = 1.0,
    zvp_mode: str = "sign_only",
) -> list[tuple[float, float, float, float]]:
    if not groups:
        return []

    runtime_stats: list[tuple[float, float, float, float]] = [(0.0, 0.0, 0.0, 0.0) for _ in groups]
    indexed_groups: list[tuple[int, QueryGroup]] = []
    indexed_log_probs: list[torch.Tensor] = []
    for idx, group in enumerate(groups):
        batch = getattr(group.data, "batch", None)
        if batch is None or "advantages" not in batch.keys() or "response_mask" not in batch.keys():
            continue
        indexed_groups.append((idx, group))
        indexed_log_probs.append(new_log_probs_list[idx])

    if not indexed_groups:
        return runtime_stats

    log_probs = torch.stack(indexed_log_probs, dim=0)
    advantages = torch.stack([group.data.batch["advantages"] for _, group in indexed_groups], dim=0)
    response_mask = torch.stack([group.data.batch["response_mask"] for _, group in indexed_groups], dim=0)
    scores, mean_surprisal, neg_frac, pos_frac = compute_group_zvp_stats_batched(
        log_probs=log_probs,
        advantages=advantages,
        response_mask=response_mask,
        adv_pos_eps=adv_pos_eps,
        lambda_neg=lambda_neg,
        zvp_mode=zvp_mode,
    )
    summary = torch.stack([scores, mean_surprisal, neg_frac, pos_frac], dim=1).detach().cpu().tolist()
    for row, (idx, _) in zip(summary, indexed_groups, strict=True):
        runtime_stats[idx] = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
    return runtime_stats


def _stable_seed_rank(query_id: str, seed: int) -> int:
    payload = f"{seed}:{query_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _candidate_sort_key(
    group: QueryGroup,
    seed: int,
    current_step: int,
    selection_mode: str,
    recency_beta: float,
    recency_decay_lambda: float,
    zvp_weight: float,
    zvp_use_recency: bool,
) -> tuple[float, int, int]:
    priority_score, _, _, _, _ = _priority_terms(
        group=group,
        current_step=current_step,
        selection_mode=selection_mode,
        recency_beta=recency_beta,
        recency_decay_lambda=recency_decay_lambda,
        zvp_weight=zvp_weight,
        zvp_use_recency=zvp_use_recency,
    )
    seed_rank = _stable_seed_rank(group.query_id, seed)
    insertion_order = int(group.insertion_order)
    return (priority_score, seed_rank, insertion_order)


def _priority_terms(
    group: QueryGroup,
    current_step: int,
    selection_mode: str,
    recency_beta: float,
    recency_decay_lambda: float,
    zvp_weight: float,
    zvp_use_recency: bool,
) -> tuple[float, float, float, int, int]:
    last_training_step = int(group.last_training_step)
    if last_training_step < 0:
        last_training_step = int(group.insertion_step)
    age = max(int(current_step) - last_training_step, 0)
    if selection_mode == "recency_only":
        # Smaller step means older sample, so ascending sort becomes oldest-first.
        priority_score = float(last_training_step)
        uncertainty = 0.0
        recency_value = float(last_training_step)
    elif selection_mode == "zvp_recency":
        # Lower score is selected first, so negate ZVP term.
        zvp_term = max(float(group.zvp_score), 0.0)
        if zvp_use_recency:
            decay_lambda = max(float(recency_decay_lambda), 1e-6)
            recency_value = float(recency_beta) * math.exp(-float(age) / decay_lambda)
        else:
            recency_value = 0.0
        priority_score = -float(zvp_weight) * zvp_term + recency_value
        uncertainty = 0.0
    else:
        uncertainty = 2.0 * abs(float(group.success_prob) - 0.5)
        decay_lambda = max(float(recency_decay_lambda), 1e-6)
        recency_value = float(recency_beta) * math.exp(-float(age) / decay_lambda)
        priority_score = uncertainty + recency_value
    return (priority_score, uncertainty, recency_value, age, last_training_step)


def compute_group_m2(old_log_probs: torch.Tensor, new_log_probs: torch.Tensor, response_mask: torch.Tensor) -> float:
    m2 = compute_group_m2_batched(
        old_log_probs=old_log_probs.unsqueeze(0),
        new_log_probs=new_log_probs.unsqueeze(0),
        response_mask=response_mask.unsqueeze(0),
    )
    return float(m2[0].item())


def select_replay_groups(
    buffer: QueryGroupReplayBuffer,
    target_groups: int,
    tau: float,
    compute_new_log_probs_fn: Callable[[DataProto], torch.Tensor],
    compute_new_log_probs_batch_fn: Optional[Callable[[list[DataProto]], list[torch.Tensor]]] = None,
    compute_group_m2_batch_fn: Optional[Callable[[list[DataProto]], list[float]]] = None,
    max_scan: Optional[int] = None,
    current_step: int = 0,
    selection_mode: str = "legacy_uncertainty_recency",
    recency_beta: float = 1.0,
    recency_decay_lambda: float = 4.0,
    zvp_weight: float = 1.0,
    adv_pos_eps: float = 1e-8,
    zvp_lambda_neg: float = 1.0,
    zvp_use_recency: bool = True,
    zvp_mode: str = "sign_only",
    compute_runtime_zvp: bool = True,
    groups_per_chunk: int = 1,
    build_candidate_priority_all: bool = True,
    timing_raw: Optional[dict[str, float]] = None,
) -> ReplaySelectionResult:
    if target_groups <= 0 or len(buffer) == 0:
        return ReplaySelectionResult(
            selected_groups=[],
            scanned_groups=0,
            target_groups=max(int(target_groups), 0),
            rejected_missing_fields=0,
            rejected_by_tau=0,
            rejected_eval_error=0,
        )

    candidates = buffer.items()
    normalized_mode = str(selection_mode).strip().lower()
    if normalized_mode not in {"legacy_uncertainty_recency", "recency_only", "zvp_recency"}:
        normalized_mode = "legacy_uncertainty_recency"
    normalized_zvp_mode = normalize_zvp_mode(zvp_mode)
    t0 = time.perf_counter()
    candidates = sorted(
        candidates,
        key=lambda group: _candidate_sort_key(
            group=group,
            seed=buffer.seed,
            current_step=current_step,
            selection_mode=normalized_mode,
            recency_beta=recency_beta,
            recency_decay_lambda=recency_decay_lambda,
            zvp_weight=zvp_weight,
            zvp_use_recency=zvp_use_recency,
        ),
    )
    if timing_raw is not None:
        timing_raw["m2_select_sort_candidates"] = timing_raw.get("m2_select_sort_candidates", 0.0) + (
            time.perf_counter() - t0
        )
    scan_limit = len(candidates) if max_scan is None else min(len(candidates), int(max_scan))
    def _build_priority_entry(group: QueryGroup) -> dict[str, object]:
        priority_score, uncertainty, recency_value, age, last_training_step = _priority_terms(
            group=group,
            current_step=current_step,
            selection_mode=normalized_mode,
            recency_beta=recency_beta,
            recency_decay_lambda=recency_decay_lambda,
            zvp_weight=zvp_weight,
            zvp_use_recency=zvp_use_recency,
        )
        return {
            "query_id": group.query_id,
            "score": priority_score,
            "uncertainty": uncertainty,
            "recency_value": recency_value,
            "age": age,
            "last_training_step": last_training_step,
            "success_prob": float(group.success_prob),
            "success_count": int(group.success_count),
            "group_size": int(group.group_size),
            "zvp_score": float(group.zvp_score),
            "zvp_mean_surprisal": float(group.zvp_mean_surprisal),
            "zvp_neg_frac": float(group.zvp_neg_frac),
            "zvp_pos_frac": float(group.zvp_pos_frac),
            "zvp_use_recency": bool(zvp_use_recency),
        }

    candidate_priority_all: list[dict[str, object]] = []
    t0 = time.perf_counter()
    if build_candidate_priority_all:
        candidate_priority_all = [_build_priority_entry(group) for group in candidates]
        candidate_priority_preview = candidate_priority_all[:10]
    else:
        candidate_priority_preview = [_build_priority_entry(group) for group in candidates[:10]]
    if timing_raw is not None:
        timing_raw["m2_select_build_priority"] = timing_raw.get("m2_select_build_priority", 0.0) + (
            time.perf_counter() - t0
        )

    selected: list[QueryGroup] = []
    accepted_m2: list[float] = []
    selected_priority_debug: list[dict[str, object]] = []
    rejected_missing_fields = 0
    rejected_by_tau = 0
    rejected_eval_error = 0
    scanned_groups = 0
    rejected_zvp_updates: list[tuple[QueryGroup, tuple[float, float, float, float]]] = []
    _fastpath_logged: list[bool] = [False]  # mutable container to track one-time warning
    required_fields = {"old_log_probs", "response_mask"}

    def _append_selected_group(
        group: QueryGroup,
        *,
        m2: float,
        runtime_stats: Optional[tuple[float, float, float, float]],
    ) -> None:
        selected.append(group)
        accepted_m2.append(m2)
        priority_score, uncertainty, recency_value, age, last_training_step = _priority_terms(
            group=group,
            current_step=current_step,
            selection_mode=normalized_mode,
            recency_beta=recency_beta,
            recency_decay_lambda=recency_decay_lambda,
            zvp_weight=zvp_weight,
            zvp_use_recency=zvp_use_recency,
        )
        debug_entry = {
            "query_id": group.query_id,
            "score": priority_score,
            "uncertainty": uncertainty,
            "recency_value": recency_value,
            "age": age,
            "last_training_step": last_training_step,
            "success_prob": float(group.success_prob),
            "success_count": int(group.success_count),
            "group_size": int(group.group_size),
            "m2": float(m2),
            "zvp_score": float(group.zvp_score),
        }
        if runtime_stats is not None:
            runtime_zvp_score, runtime_zvp_mean_surprisal, runtime_zvp_neg_frac, runtime_zvp_pos_frac = runtime_stats
            debug_entry.update(
                {
                    "zvp_runtime_score": float(runtime_zvp_score),
                    "zvp_runtime_mean_surprisal": float(runtime_zvp_mean_surprisal),
                    "zvp_runtime_neg_frac": float(runtime_zvp_neg_frac),
                    "zvp_runtime_pos_frac": float(runtime_zvp_pos_frac),
                }
            )
        selected_priority_debug.append(debug_entry)

    def _accept_if_passing_tau(group: QueryGroup, new_log_probs: torch.Tensor) -> None:
        nonlocal rejected_by_tau
        t0 = time.perf_counter()
        old_log_probs = group.data.batch["old_log_probs"]
        response_mask = group.data.batch["response_mask"]
        m2 = compute_group_m2(old_log_probs=old_log_probs, new_log_probs=new_log_probs, response_mask=response_mask)

        if m2 <= tau:
            runtime_stats = None
            if compute_runtime_zvp and "advantages" in group.data.batch.keys():
                runtime_stats = compute_group_runtime_zvp_stats_batched(
                    [group],
                    [new_log_probs],
                    adv_pos_eps=adv_pos_eps,
                    lambda_neg=zvp_lambda_neg,
                    zvp_mode=normalized_zvp_mode,
                )[0]
            _append_selected_group(group, m2=float(m2), runtime_stats=runtime_stats)
        else:
            rejected_by_tau += 1
            if compute_runtime_zvp and "advantages" in group.data.batch.keys():
                rej_zvp = compute_group_runtime_zvp_stats_batched(
                    [group],
                    [new_log_probs],
                    adv_pos_eps=adv_pos_eps,
                    lambda_neg=zvp_lambda_neg,
                    zvp_mode=normalized_zvp_mode,
                )[0]
                rejected_zvp_updates.append((group, rej_zvp))
        if timing_raw is not None:
            timing_raw["m2_select_m2_eval"] = timing_raw.get("m2_select_m2_eval", 0.0) + (time.perf_counter() - t0)

    def _build_scan_entries(groups: list[QueryGroup]) -> tuple[list[Optional[int]], list[QueryGroup]]:
        candidate_valid_indices: list[Optional[int]] = []
        valid_groups: list[QueryGroup] = []
        for group in groups:
            if group.data.batch is None or not required_fields.issubset(set(group.data.batch.keys())):
                candidate_valid_indices.append(None)
                continue
            candidate_valid_indices.append(len(valid_groups))
            valid_groups.append(group)
        return candidate_valid_indices, valid_groups

    def _collect_selected_valid_indices(
        candidate_valid_indices: list[Optional[int]],
        m2_by_valid_index: list[float],
    ) -> list[int]:
        nonlocal scanned_groups, rejected_missing_fields, rejected_by_tau
        selected_valid_indices: list[int] = []
        for valid_idx in candidate_valid_indices:
            if len(selected_valid_indices) >= target_groups:
                break
            scanned_groups += 1
            if valid_idx is None:
                rejected_missing_fields += 1
                continue
            if float(m2_by_valid_index[valid_idx]) <= float(tau):
                selected_valid_indices.append(valid_idx)
            else:
                rejected_by_tau += 1
        return selected_valid_indices

    def _append_selected_valid_indices(
        valid_groups: list[QueryGroup],
        selected_valid_indices: list[int],
        m2_by_valid_index: list[float],
        runtime_stats_map: dict[int, tuple[float, float, float, float]],
    ) -> None:
        for valid_idx in selected_valid_indices:
            _append_selected_group(
                valid_groups[valid_idx],
                m2=float(m2_by_valid_index[valid_idx]),
                runtime_stats=runtime_stats_map.get(valid_idx, None),
            )

    def _run_per_group_fallback(
        candidate_valid_indices: list[Optional[int]],
        valid_groups: list[QueryGroup],
        *,
        precomputed_new_log_probs: Optional[list[torch.Tensor]] = None,
    ) -> None:
        nonlocal scanned_groups, rejected_missing_fields, rejected_eval_error
        for valid_idx in candidate_valid_indices:
            if len(selected) >= target_groups:
                break
            scanned_groups += 1
            if valid_idx is None:
                rejected_missing_fields += 1
                continue
            group = valid_groups[valid_idx]
            try:
                if precomputed_new_log_probs is None:
                    t0 = time.perf_counter()
                    new_log_probs = compute_new_log_probs_fn(group.data)
                    if timing_raw is not None:
                        timing_raw["m2_select_logprob_eval"] = timing_raw.get("m2_select_logprob_eval", 0.0) + (
                            time.perf_counter() - t0
                        )
                else:
                    new_log_probs = precomputed_new_log_probs[valid_idx]
                _accept_if_passing_tau(group, new_log_probs)
            except Exception:
                rejected_eval_error += 1
                continue

    scan_candidates = candidates[:scan_limit]
    use_chunked_logprob = (
        compute_new_log_probs_batch_fn is not None
        and int(groups_per_chunk) > 1
    )
    use_gpu_m2_fastpath = compute_group_m2_batch_fn is not None and not compute_runtime_zvp

    loop_start = time.perf_counter()
    if not use_chunked_logprob:
        for group in scan_candidates:
            if len(selected) >= target_groups:
                break
            scanned_groups += 1

            if group.data.batch is None or not required_fields.issubset(set(group.data.batch.keys())):
                rejected_missing_fields += 1
                continue

            try:
                t0 = time.perf_counter()
                new_log_probs = compute_new_log_probs_fn(group.data)
                if timing_raw is not None:
                    timing_raw["m2_select_logprob_eval"] = timing_raw.get("m2_select_logprob_eval", 0.0) + (
                        time.perf_counter() - t0
                    )
                _accept_if_passing_tau(group, new_log_probs)
            except Exception:
                rejected_eval_error += 1
                continue
    else:
        candidate_valid_indices, valid_groups = _build_scan_entries(scan_candidates)

        if not valid_groups:
            _collect_selected_valid_indices(candidate_valid_indices, [])
        elif use_gpu_m2_fastpath:
            try:
                t0 = time.perf_counter()
                m2_cpu = [float(m2) for m2 in compute_group_m2_batch_fn([group.data for group in valid_groups])]
                if timing_raw is not None:
                    timing_raw["m2_select_logprob_eval"] = timing_raw.get("m2_select_logprob_eval", 0.0) + (
                        time.perf_counter() - t0
                    )
                if len(m2_cpu) != len(valid_groups):
                    raise ValueError(
                        "compute_group_m2_batch_fn must return one scalar per group, "
                        f"got {len(m2_cpu)} for {len(valid_groups)} groups."
                    )
            except Exception as _fastpath_exc:
                if not _fastpath_logged[0]:
                    import traceback as _tb
                    print(
                        f"[m2_replay] WARNING: GPU M2 fastpath failed, falling back to CPU path. "
                        f"Exception: {type(_fastpath_exc).__name__}: {_fastpath_exc}\n"
                        f"{_tb.format_exc()}",
                        flush=True,
                    )
                    _fastpath_logged[0] = True
            else:
                selected_valid_indices = _collect_selected_valid_indices(candidate_valid_indices, m2_cpu)
                _append_selected_valid_indices(valid_groups, selected_valid_indices, m2_cpu, runtime_stats_map={})
                if timing_raw is not None:
                    timing_raw["m2_select_scan_loop"] = timing_raw.get("m2_select_scan_loop", 0.0) + (
                        time.perf_counter() - loop_start
                    )
                return ReplaySelectionResult(
                    selected_groups=selected,
                    scanned_groups=scanned_groups,
                    target_groups=target_groups,
                    rejected_missing_fields=rejected_missing_fields,
                    rejected_by_tau=rejected_by_tau,
                    rejected_eval_error=rejected_eval_error,
                    accepted_m2=accepted_m2,
                    selected_priority_debug=selected_priority_debug,
                    candidate_priority_preview=candidate_priority_preview,
                    candidate_priority_all=candidate_priority_all,
                )

        if valid_groups:
            try:
                t0 = time.perf_counter()
                new_log_probs_list = compute_new_log_probs_batch_fn([group.data for group in valid_groups])
                if timing_raw is not None:
                    timing_raw["m2_select_logprob_eval"] = timing_raw.get("m2_select_logprob_eval", 0.0) + (
                        time.perf_counter() - t0
                    )
                if len(new_log_probs_list) != len(valid_groups):
                    raise ValueError(
                        "compute_new_log_probs_batch_fn must return one tensor per group, "
                        f"got {len(new_log_probs_list)} for {len(valid_groups)} groups."
                    )
            except Exception:
                _run_per_group_fallback(candidate_valid_indices, valid_groups)
            else:
                try:
                    t0 = time.perf_counter()
                    old_log_probs = torch.stack([group.data.batch["old_log_probs"] for group in valid_groups], dim=0)
                    response_mask = torch.stack([group.data.batch["response_mask"] for group in valid_groups], dim=0)
                    new_log_probs = torch.stack(new_log_probs_list, dim=0)
                    m2_values = compute_group_m2_batched(
                        old_log_probs=old_log_probs,
                        new_log_probs=new_log_probs,
                        response_mask=response_mask,
                    )
                    if timing_raw is not None:
                        timing_raw["m2_select_m2_eval"] = timing_raw.get("m2_select_m2_eval", 0.0) + (
                            time.perf_counter() - t0
                        )
                    m2_cpu = [float(value) for value in m2_values.detach().cpu().tolist()]
                except Exception:
                    _run_per_group_fallback(
                        candidate_valid_indices,
                        valid_groups,
                        precomputed_new_log_probs=new_log_probs_list,
                    )
                else:
                    selected_valid_indices = _collect_selected_valid_indices(candidate_valid_indices, m2_cpu)
                    runtime_stats_map: dict[int, tuple[float, float, float, float]] = {}
                    if compute_runtime_zvp and selected_valid_indices:
                        accepted_groups = [valid_groups[idx] for idx in selected_valid_indices]
                        accepted_new_log_probs = [new_log_probs_list[idx] for idx in selected_valid_indices]
                        runtime_stats = compute_group_runtime_zvp_stats_batched(
                            accepted_groups,
                            accepted_new_log_probs,
                            adv_pos_eps=adv_pos_eps,
                            lambda_neg=zvp_lambda_neg,
                            zvp_mode=normalized_zvp_mode,
                        )
                        runtime_stats_map = {
                            selected_idx: runtime_stat
                            for selected_idx, runtime_stat in zip(selected_valid_indices, runtime_stats, strict=True)
                        }
                    if compute_runtime_zvp:
                        selected_set = set(selected_valid_indices)
                        rej_indices = [
                            i
                            for i, m2_val in enumerate(m2_cpu)
                            if m2_val > tau
                            and i not in selected_set
                            and "advantages" in valid_groups[i].data.batch.keys()
                        ]
                        if rej_indices:
                            rej_groups = [valid_groups[i] for i in rej_indices]
                            rej_lp = [new_log_probs_list[i] for i in rej_indices]
                            rej_stats = compute_group_runtime_zvp_stats_batched(
                                rej_groups,
                                rej_lp,
                                adv_pos_eps=adv_pos_eps,
                                lambda_neg=zvp_lambda_neg,
                                zvp_mode=normalized_zvp_mode,
                            )
                            for grp, st in zip(rej_groups, rej_stats, strict=True):
                                rejected_zvp_updates.append((grp, st))
                    _append_selected_valid_indices(
                        valid_groups,
                        selected_valid_indices,
                        m2_cpu,
                        runtime_stats_map=runtime_stats_map,
                    )
        else:
            _collect_selected_valid_indices(candidate_valid_indices, [])
    if timing_raw is not None:
        timing_raw["m2_select_scan_loop"] = timing_raw.get("m2_select_scan_loop", 0.0) + (
            time.perf_counter() - loop_start
        )

    return ReplaySelectionResult(
        selected_groups=selected,
        scanned_groups=scanned_groups,
        target_groups=target_groups,
        rejected_missing_fields=rejected_missing_fields,
        rejected_by_tau=rejected_by_tau,
        rejected_eval_error=rejected_eval_error,
        accepted_m2=accepted_m2,
        selected_priority_debug=selected_priority_debug,
        candidate_priority_preview=candidate_priority_preview,
        candidate_priority_all=candidate_priority_all,
        rejected_by_tau_zvp_updates=rejected_zvp_updates,
    )


def derive_micro_group_multiple(
    micro_batch_size_per_gpu: Optional[int],
    n_gpus: int,
    rollout_n: int,
) -> int:
    if rollout_n <= 0:
        return 1

    if micro_batch_size_per_gpu is None or micro_batch_size_per_gpu <= 0 or n_gpus <= 0:
        return 1

    global_micro_samples = int(micro_batch_size_per_gpu) * int(n_gpus)
    return max(global_micro_samples // int(rollout_n), 1)
