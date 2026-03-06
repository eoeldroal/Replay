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

    def to_metrics(self, prefix: str) -> dict[str, float]:
        accepted = len(self.selected_groups)
        avg_m2 = float(sum(self.accepted_m2) / len(self.accepted_m2)) if self.accepted_m2 else 0.0
        return {
            f"{prefix}/pass1/target_groups": float(self.target_groups),
            f"{prefix}/pass1/scanned_groups": float(self.scanned_groups),
            f"{prefix}/pass1/accepted_groups": float(accepted),
            f"{prefix}/pass1/rejected_missing_fields": float(self.rejected_missing_fields),
            f"{prefix}/pass1/rejected_by_tau": float(self.rejected_by_tau),
            f"{prefix}/pass1/rejected_eval_error": float(self.rejected_eval_error),
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
            if not hasattr(group, "zvp_last_update_step"):
                group.zvp_last_update_step = int(getattr(group, "insertion_step", 0))
            if not hasattr(group, "zvp_update_count"):
                group.zvp_update_count = 0


def normalize_zvp_mode(zvp_mode: str) -> str:
    normalized_mode = str(zvp_mode).strip().lower()
    if normalized_mode not in {"sign_only", "adv_magnitude"}:
        return "sign_only"
    return normalized_mode


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
    if log_probs.shape != advantages.shape or log_probs.shape != response_mask.shape:
        raise ValueError(
            f"Shape mismatch in ZVP stats: {log_probs.shape=}, {advantages.shape=}, {response_mask.shape=}"
        )

    mask = response_mask.bool()
    if not torch.any(mask):
        return (0.0, 0.0, 0.0, 0.0)

    adv = advantages.float()
    surprisal = (-log_probs.float()).clamp_min(0.0)
    probs = torch.exp(-surprisal).clamp(0.0, 1.0)

    pos_mask = mask & (adv > float(adv_pos_eps))
    neg_mask = mask & (adv < -float(adv_pos_eps))
    pos_count = int(pos_mask.sum().item())
    neg_count = int(neg_mask.sum().item())
    total_count = int(mask.sum().item())
    if total_count <= 0:
        return (0.0, 0.0, 0.0, 0.0)

    pos_term = torch.zeros_like(probs)
    neg_term = torch.zeros_like(probs)
    if pos_count > 0:
        pos_term[pos_mask] = 1.0 - probs[pos_mask]
    if neg_count > 0:
        neg_term[neg_mask] = probs[neg_mask]

    token_score = pos_term + float(lambda_neg) * neg_term
    normalized_mode = normalize_zvp_mode(zvp_mode)
    if normalized_mode == "adv_magnitude":
        active_mask = pos_mask | neg_mask
        abs_adv = adv.abs()
        weights = torch.zeros_like(abs_adv)
        weights[active_mask] = abs_adv[active_mask]
        denom = float(weights.sum().item())
        if denom > 0.0:
            score = float(((token_score * weights).sum() / denom).item())
        else:
            score = 0.0
    else:
        score = float(token_score[mask].mean().item())

    mean_surprisal = float(surprisal[mask].mean().item())
    pos_frac = float(pos_count / total_count)
    neg_frac = float(neg_count / total_count)
    return (score, mean_surprisal, neg_frac, pos_frac)


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
    if old_log_probs.shape != new_log_probs.shape:
        raise ValueError(f"Shape mismatch: {old_log_probs.shape=} vs {new_log_probs.shape=}")
    if response_mask.shape != old_log_probs.shape:
        raise ValueError(
            f"Mask shape mismatch: {response_mask.shape=} vs log_probs shape {old_log_probs.shape=}"
        )

    mask = response_mask.float()
    denom = float(mask.sum().item())
    if denom <= 0:
        return float("inf")

    ell = (new_log_probs - old_log_probs).float()
    m2 = (ell.square() * mask).sum() / denom
    return float(m2.item())


def select_replay_groups(
    buffer: QueryGroupReplayBuffer,
    target_groups: int,
    tau: float,
    compute_new_log_probs_fn: Callable[[DataProto], torch.Tensor],
    compute_new_log_probs_batch_fn: Optional[Callable[[list[DataProto]], list[torch.Tensor]]] = None,
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

    def _accept_if_passing_tau(group: QueryGroup, new_log_probs: torch.Tensor) -> None:
        nonlocal rejected_by_tau
        t0 = time.perf_counter()
        old_log_probs = group.data.batch["old_log_probs"]
        response_mask = group.data.batch["response_mask"]
        m2 = compute_group_m2(old_log_probs=old_log_probs, new_log_probs=new_log_probs, response_mask=response_mask)

        if m2 <= tau:
            runtime_zvp_score = 0.0
            runtime_zvp_mean_surprisal = 0.0
            runtime_zvp_neg_frac = 0.0
            runtime_zvp_pos_frac = 0.0
            if "advantages" in group.data.batch.keys():
                (
                    runtime_zvp_score,
                    runtime_zvp_mean_surprisal,
                    runtime_zvp_neg_frac,
                    runtime_zvp_pos_frac,
                ) = compute_group_zvp_stats(
                    log_probs=new_log_probs,
                    advantages=group.data.batch["advantages"],
                    response_mask=response_mask,
                    adv_pos_eps=adv_pos_eps,
                    lambda_neg=zvp_lambda_neg,
                    zvp_mode=normalized_zvp_mode,
                )

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
            selected_priority_debug.append(
                {
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
                    "zvp_runtime_score": float(runtime_zvp_score),
                    "zvp_runtime_mean_surprisal": float(runtime_zvp_mean_surprisal),
                    "zvp_runtime_neg_frac": float(runtime_zvp_neg_frac),
                    "zvp_runtime_pos_frac": float(runtime_zvp_pos_frac),
                }
            )
        else:
            rejected_by_tau += 1
        if timing_raw is not None:
            timing_raw["m2_select_m2_eval"] = timing_raw.get("m2_select_m2_eval", 0.0) + (time.perf_counter() - t0)

    scan_candidates = candidates[:scan_limit]
    use_chunked_logprob = (
        compute_new_log_probs_batch_fn is not None
        and int(groups_per_chunk) > 1
    )

    loop_start = time.perf_counter()
    if not use_chunked_logprob:
        for group in scan_candidates:
            if len(selected) >= target_groups:
                break
            scanned_groups += 1

            required = {"old_log_probs", "response_mask"}
            if group.data.batch is None or not required.issubset(set(group.data.batch.keys())):
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
        chunk_size = max(int(groups_per_chunk), 1)
        for chunk_start in range(0, len(scan_candidates), chunk_size):
            if len(selected) >= target_groups:
                break
            chunk = scan_candidates[chunk_start : chunk_start + chunk_size]

            valid_groups: list[QueryGroup] = []
            for group in chunk:
                scanned_groups += 1
                required = {"old_log_probs", "response_mask"}
                if group.data.batch is None or not required.issubset(set(group.data.batch.keys())):
                    rejected_missing_fields += 1
                    continue
                valid_groups.append(group)

            if not valid_groups:
                continue

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
                # Fallback to per-group evaluation to preserve correctness.
                for group in valid_groups:
                    if len(selected) >= target_groups:
                        break
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

            for group, new_log_probs in zip(valid_groups, new_log_probs_list, strict=True):
                if len(selected) >= target_groups:
                    break
                try:
                    _accept_if_passing_tau(group, new_log_probs)
                except Exception:
                    rejected_eval_error += 1
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
