from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _read_script(rel_path: str) -> str:
    return (ROOT / rel_path).read_text()


def test_rev7_uses_explicit_two_turnovers_start_mode():
    script = _read_script("RL_side_3/grpo-qwen3-1.7b-s8-m2-replay-rev7.sh")
    assert "start_mode=two_turnovers" in script
    assert "+algorithm.m2_replay.schedule.start_mode=${start_mode}" in script
    assert "+algorithm.m2_replay.schedule.one_turnover_gate=${one_turnover_gate}" not in script


def test_rev6_uses_explicit_two_turnovers_start_mode():
    script = _read_script("RL_side_3/grpo-qwen3-1.7b-base-s8-m2-replay-rev6.sh")
    assert "start_mode=two_turnovers" in script
    assert "+algorithm.m2_replay.schedule.start_mode=${start_mode}" in script
    assert "+algorithm.m2_replay.schedule.one_turnover_gate=${one_turnover_gate}" not in script


def test_rev8_uses_buffer_full_and_tau_0015():
    script = _read_script("RL_side_3/grpo-qwen3-1.7b-s8-m2-replay-rev8.sh")
    assert "experiment_name=grpo-qwen3-1.7b-s8-m2-replay-rev8-tau0015-b1024-ndeg-advmag-bufferfull" in script
    assert "start_mode=buffer_full" in script
    assert "+algorithm.m2_replay.tau=0.0015" in script
    assert "+algorithm.m2_replay.schedule.start_mode=${start_mode}" in script
