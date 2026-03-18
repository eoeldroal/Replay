from verl.experimental.agent_loop.tool_agent_loop import AgentData, _normalize_tool_execute_result
from verl.tools.schemas import ToolResponse


def _make_agent_data():
    return AgentData(
        messages=[],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="req-1",
        tools_kwargs={},
        interaction=None,
        interaction_kwargs={},
        sample_id="sample-1",
        reference_page=[],
        agent_index=0,
    )


def test_normalize_tool_execute_result_accepts_tool_response():
    agent_data = _make_agent_data()
    response = ToolResponse(text="ok")

    normalized = _normalize_tool_execute_result(response, agent_data=agent_data)

    assert normalized.text == "ok"
    assert agent_data.turn_scores == []


def test_normalize_tool_execute_result_accepts_tuple_and_records_reward():
    agent_data = _make_agent_data()
    response = ToolResponse(text="ok")

    normalized = _normalize_tool_execute_result((response, 0.25, {"status": "success"}), agent_data=agent_data)

    assert normalized.text == "ok"
    assert agent_data.turn_scores == [0.25]
