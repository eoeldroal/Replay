# Copyright 2026

from __future__ import annotations

import hashlib
import math
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
) -> tuple[float, int, int]:
    priority_score, _, _, _, _ = _priority_terms(
        group=group,
        current_step=current_step,
        selection_mode=selection_mode,
        recency_beta=recency_beta,
        recency_decay_lambda=recency_decay_lambda,
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
    max_scan: Optional[int] = None,
    current_step: int = 0,
    selection_mode: str = "legacy_uncertainty_recency",
    recency_beta: float = 1.0,
    recency_decay_lambda: float = 4.0,
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
    if normalized_mode not in {"legacy_uncertainty_recency", "recency_only"}:
        normalized_mode = "legacy_uncertainty_recency"
    candidates = sorted(
        candidates,
        key=lambda group: _candidate_sort_key(
            group=group,
            seed=buffer.seed,
            current_step=current_step,
            selection_mode=normalized_mode,
            recency_beta=recency_beta,
            recency_decay_lambda=recency_decay_lambda,
        ),
    )
    scan_limit = len(candidates) if max_scan is None else min(len(candidates), int(max_scan))
    candidate_priority_all: list[dict[str, object]] = []
    for group in candidates:
        priority_score, uncertainty, recency_value, age, last_training_step = _priority_terms(
            group=group,
            current_step=current_step,
            selection_mode=normalized_mode,
            recency_beta=recency_beta,
            recency_decay_lambda=recency_decay_lambda,
        )
        candidate_priority_all.append(
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
            }
        )
    candidate_priority_preview = candidate_priority_all[:10]

    selected: list[QueryGroup] = []
    accepted_m2: list[float] = []
    selected_priority_debug: list[dict[str, object]] = []
    rejected_missing_fields = 0
    rejected_by_tau = 0
    rejected_eval_error = 0
    scanned_groups = 0

    for group in candidates[:scan_limit]:
        if len(selected) >= target_groups:
            break
        scanned_groups += 1

        required = {"old_log_probs", "response_mask"}
        if group.data.batch is None or not required.issubset(set(group.data.batch.keys())):
            rejected_missing_fields += 1
            continue

        try:
            old_log_probs = group.data.batch["old_log_probs"]
            response_mask = group.data.batch["response_mask"]
            new_log_probs = compute_new_log_probs_fn(group.data)
            m2 = compute_group_m2(old_log_probs=old_log_probs, new_log_probs=new_log_probs, response_mask=response_mask)
        except Exception:
            rejected_eval_error += 1
            continue

        if m2 <= tau:
            selected.append(group)
            accepted_m2.append(m2)
            priority_score, uncertainty, recency_value, age, last_training_step = _priority_terms(
                group=group,
                current_step=current_step,
                selection_mode=normalized_mode,
                recency_beta=recency_beta,
                recency_decay_lambda=recency_decay_lambda,
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
                }
            )
        else:
            rejected_by_tau += 1

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
