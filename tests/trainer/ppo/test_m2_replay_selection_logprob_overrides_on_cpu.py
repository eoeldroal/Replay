from types import SimpleNamespace

import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.workers.fsdp_workers import _resolve_log_prob_batching_settings


def _make_group_data(rows: int, cols: int = 3) -> DataProto:
    return DataProto.from_dict(
        tensors={
            "old_log_probs": torch.zeros(rows, cols, dtype=torch.float32),
            "response_mask": torch.ones(rows, cols, dtype=torch.float32),
        }
    )


def test_resolve_log_prob_batching_settings_prefers_selection_overrides():
    config_source = SimpleNamespace(
        log_prob_micro_batch_size_per_gpu=32,
        log_prob_max_token_len_per_gpu=16384,
        log_prob_use_dynamic_bsz=False,
    )
    meta_info = {
        "log_prob_micro_batch_size_override": 48,
        "log_prob_use_dynamic_bsz_override": True,
        "log_prob_max_token_len_override": 24576,
    }

    micro_batch_size, max_token_len, use_dynamic_bsz = _resolve_log_prob_batching_settings(meta_info, config_source)

    assert micro_batch_size == 48
    assert max_token_len == 24576
    assert use_dynamic_bsz is True
    assert "log_prob_micro_batch_size_override" not in meta_info
    assert "log_prob_use_dynamic_bsz_override" not in meta_info
    assert "log_prob_max_token_len_override" not in meta_info


def test_resolve_log_prob_batching_settings_falls_back_to_rollout_defaults():
    config_source = SimpleNamespace(
        log_prob_micro_batch_size_per_gpu=24,
        log_prob_max_token_len_per_gpu=12288,
        log_prob_use_dynamic_bsz=True,
    )
    meta_info = {}

    micro_batch_size, max_token_len, use_dynamic_bsz = _resolve_log_prob_batching_settings(meta_info, config_source)

    assert micro_batch_size == 24
    assert max_token_len == 12288
    assert use_dynamic_bsz is True


def test_m2_compute_new_log_probs_forwards_selection_only_logprob_overrides():
    captured = {}

    def _fake_compute_old_log_prob(
        batch,
        calculate_entropy,
        log_prob_micro_batch_size_override=None,
        log_prob_use_dynamic_bsz_override=None,
        log_prob_max_token_len_override=None,
    ):
        captured["calculate_entropy"] = calculate_entropy
        captured["log_prob_micro_batch_size_override"] = log_prob_micro_batch_size_override
        captured["log_prob_use_dynamic_bsz_override"] = log_prob_use_dynamic_bsz_override
        captured["log_prob_max_token_len_override"] = log_prob_max_token_len_override
        return DataProto.from_dict(tensors={"old_log_probs": torch.ones_like(batch.batch["old_log_probs"])}), 0.0

    trainer = SimpleNamespace(
        m2_replay_log_prob_micro_batch_size_per_gpu=64,
        m2_replay_log_prob_use_dynamic_bsz=True,
        m2_replay_log_prob_max_token_len_per_gpu=16384,
        _compute_old_log_prob=_fake_compute_old_log_prob,
    )

    output = RayPPOTrainer._m2_compute_new_log_probs(trainer, _make_group_data(rows=2))

    assert torch.equal(output, torch.ones(2, 3))
    assert captured == {
        "calculate_entropy": False,
        "log_prob_micro_batch_size_override": 64,
        "log_prob_use_dynamic_bsz_override": True,
        "log_prob_max_token_len_override": 16384,
    }


def test_m2_compute_new_log_probs_batch_forwards_selection_only_logprob_overrides():
    captured = {}

    def _fake_compute_old_log_prob(
        batch,
        calculate_entropy,
        log_prob_micro_batch_size_override=None,
        log_prob_use_dynamic_bsz_override=None,
        log_prob_max_token_len_override=None,
    ):
        captured["calculate_entropy"] = calculate_entropy
        captured["log_prob_micro_batch_size_override"] = log_prob_micro_batch_size_override
        captured["log_prob_use_dynamic_bsz_override"] = log_prob_use_dynamic_bsz_override
        captured["log_prob_max_token_len_override"] = log_prob_max_token_len_override
        total_rows = batch.batch["old_log_probs"].shape[0]
        values = torch.arange(total_rows * 3, dtype=torch.float32).reshape(total_rows, 3)
        return DataProto.from_dict(tensors={"old_log_probs": values}), 0.0

    trainer = SimpleNamespace(
        m2_replay_log_prob_micro_batch_size_per_gpu=40,
        m2_replay_log_prob_use_dynamic_bsz=True,
        m2_replay_log_prob_max_token_len_per_gpu=12288,
        _compute_old_log_prob=_fake_compute_old_log_prob,
    )

    outputs = RayPPOTrainer._m2_compute_new_log_probs_batch(
        trainer,
        [_make_group_data(rows=2), _make_group_data(rows=1)],
    )

    assert [tuple(t.shape) for t in outputs] == [(2, 3), (1, 3)]
    assert captured == {
        "calculate_entropy": False,
        "log_prob_micro_batch_size_override": 40,
        "log_prob_use_dynamic_bsz_override": True,
        "log_prob_max_token_len_override": 12288,
    }


def test_m2_compute_group_m2_batch_forwards_selection_only_logprob_overrides_to_worker():
    captured = {}

    class _FakeActorRolloutWG:
        def compute_log_prob_m2(self, batch):
            captured["log_prob_micro_batch_size_override"] = batch.meta_info.get("log_prob_micro_batch_size_override")
            captured["log_prob_use_dynamic_bsz_override"] = batch.meta_info.get("log_prob_use_dynamic_bsz_override")
            captured["log_prob_max_token_len_override"] = batch.meta_info.get("log_prob_max_token_len_override")
            group_count = len(batch.meta_info["m2_group_lengths"])
            return DataProto.from_dict(tensors={"m2": torch.arange(1, group_count + 1, dtype=torch.float32)})

    trainer = SimpleNamespace(
        actor_rollout_wg=_FakeActorRolloutWG(),
        m2_replay_log_prob_micro_batch_size_per_gpu=56,
        m2_replay_log_prob_use_dynamic_bsz=True,
        m2_replay_log_prob_max_token_len_per_gpu=20480,
    )

    values = RayPPOTrainer._m2_compute_group_m2_batch(
        trainer,
        [_make_group_data(rows=2), _make_group_data(rows=1)],
    )

    assert values == [1.0, 2.0]
    assert captured == {
        "log_prob_micro_batch_size_override": 56,
        "log_prob_use_dynamic_bsz_override": True,
        "log_prob_max_token_len_override": 20480,
    }
