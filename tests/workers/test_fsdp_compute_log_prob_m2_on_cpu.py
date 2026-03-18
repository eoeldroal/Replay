# Copyright 2026

from types import SimpleNamespace

import torch

from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def _unwrap(func):
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func


def _make_worker():
    return SimpleNamespace(
        _is_actor=True,
        _is_offload_param=False,
        config=SimpleNamespace(
            rollout=SimpleNamespace(
                log_prob_micro_batch_size_per_gpu=4,
                log_prob_max_token_len_per_gpu=8192,
                log_prob_use_dynamic_bsz=True,
                temperature=1.0,
            ),
            ref=SimpleNamespace(
                log_prob_micro_batch_size_per_gpu=4,
                log_prob_max_token_len_per_gpu=8192,
                log_prob_use_dynamic_bsz=True,
            ),
        ),
        tokenizer=SimpleNamespace(pad_token_id=0),
    )


def test_compute_log_prob_m2_returns_empty_output_for_empty_shard():
    compute_log_prob_m2 = _unwrap(ActorRolloutRefWorker.compute_log_prob_m2)
    worker = _make_worker()
    batch = DataProto.from_dict(
        tensors={
            "old_log_probs": torch.zeros(0, 3, dtype=torch.float32),
            "response_mask": torch.ones(0, 3, dtype=torch.float32),
        },
        meta_info={"m2_group_lengths": []},
    )

    output = compute_log_prob_m2(worker, batch)

    assert tuple(output.batch["m2"].shape) == (0,)
    assert float(output.meta_info["temperature"]) == 1.0


def test_compute_log_prob_m2_requires_group_lengths_meta():
    compute_log_prob_m2 = _unwrap(ActorRolloutRefWorker.compute_log_prob_m2)
    worker = _make_worker()
    batch = DataProto.from_dict(
        tensors={
            "old_log_probs": torch.zeros(2, 3, dtype=torch.float32),
            "response_mask": torch.ones(2, 3, dtype=torch.float32),
        }
    )

    try:
        compute_log_prob_m2(worker, batch)
    except ValueError as exc:
        assert "requires meta_info['m2_group_lengths']" in str(exc)
    else:
        raise AssertionError("expected compute_log_prob_m2 to reject missing m2_group_lengths")


def test_compute_log_prob_m2_rejects_nonpositive_group_lengths():
    compute_log_prob_m2 = _unwrap(ActorRolloutRefWorker.compute_log_prob_m2)
    worker = _make_worker()
    batch = DataProto.from_dict(
        tensors={
            "old_log_probs": torch.zeros(2, 3, dtype=torch.float32),
            "response_mask": torch.ones(2, 3, dtype=torch.float32),
        },
        meta_info={"m2_group_lengths": [2, 0]},
    )

    try:
        compute_log_prob_m2(worker, batch)
    except ValueError as exc:
        assert "must all be > 0" in str(exc)
    else:
        raise AssertionError("expected compute_log_prob_m2 to reject nonpositive group lengths")


def test_compute_log_prob_m2_rejects_group_length_sum_mismatch():
    compute_log_prob_m2 = _unwrap(ActorRolloutRefWorker.compute_log_prob_m2)
    worker = _make_worker()
    batch = DataProto.from_dict(
        tensors={
            "old_log_probs": torch.zeros(2, 3, dtype=torch.float32),
            "response_mask": torch.ones(2, 3, dtype=torch.float32),
        },
        meta_info={"m2_group_lengths": [1]},
    )

    try:
        compute_log_prob_m2(worker, batch)
    except ValueError as exc:
        assert "group length mismatch" in str(exc)
    else:
        raise AssertionError("expected compute_log_prob_m2 to reject group-length sum mismatch")
