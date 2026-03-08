from verl.trainer.ppo.ray_trainer import (
    compute_m2_replay_start_gate_state,
    normalize_m2_replay_start_mode,
)


def test_normalize_m2_replay_start_mode_supports_quarter_and_half():
    assert normalize_m2_replay_start_mode("quarter") == "quarter"
    assert normalize_m2_replay_start_mode("half") == "half"
    assert normalize_m2_replay_start_mode("buffer_full") == "buffer_full"
    assert normalize_m2_replay_start_mode("two_turnovers") == "two_turnovers"
    assert normalize_m2_replay_start_mode(None, legacy_one_turnover_gate=False) == "immediate"
    assert normalize_m2_replay_start_mode(None, legacy_one_turnover_gate=True) == "two_turnovers"


def test_normalize_m2_replay_start_mode_falls_back_to_immediate():
    assert normalize_m2_replay_start_mode("unknown-mode") == "immediate"
    assert normalize_m2_replay_start_mode("") == "immediate"


def test_compute_m2_replay_start_gate_state_for_quarter_threshold():
    state = compute_m2_replay_start_gate_state(
        start_mode="quarter",
        buffer_size_pre_select=255,
        buffer_capacity=1024,
        buffer_inserted=255,
    )
    assert state.quarter_threshold == 256
    assert state.half_threshold == 512
    assert state.quarter_ready is False
    assert state.selection_ready is False

    state = compute_m2_replay_start_gate_state(
        start_mode="quarter",
        buffer_size_pre_select=256,
        buffer_capacity=1024,
        buffer_inserted=256,
    )
    assert state.quarter_ready is True
    assert state.selection_ready is True


def test_compute_m2_replay_start_gate_state_for_half_and_two_turnovers():
    half_state = compute_m2_replay_start_gate_state(
        start_mode="half",
        buffer_size_pre_select=511,
        buffer_capacity=1024,
        buffer_inserted=511,
    )
    assert half_state.half_ready is False
    assert half_state.selection_ready is False

    half_state = compute_m2_replay_start_gate_state(
        start_mode="half",
        buffer_size_pre_select=512,
        buffer_capacity=1024,
        buffer_inserted=512,
    )
    assert half_state.half_ready is True
    assert half_state.selection_ready is True

    turnover_state = compute_m2_replay_start_gate_state(
        start_mode="two_turnovers",
        buffer_size_pre_select=1024,
        buffer_capacity=1024,
        buffer_inserted=1500,
    )
    assert turnover_state.buffer_full_ready is True
    assert turnover_state.two_turnovers_ready is False
    assert turnover_state.selection_ready is False

    turnover_state = compute_m2_replay_start_gate_state(
        start_mode="two_turnovers",
        buffer_size_pre_select=1024,
        buffer_capacity=1024,
        buffer_inserted=2048,
    )
    assert turnover_state.two_turnovers_ready is True
    assert turnover_state.selection_ready is True
