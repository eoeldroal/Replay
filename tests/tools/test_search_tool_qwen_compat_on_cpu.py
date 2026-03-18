import asyncio

from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.tools.search_tool import SearchTool, init_search_execution_pool


def _build_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Searches the web for relevant information based on the given query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A single search query string.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    )


class _FakeExecuteMethod:
    def __init__(self):
        self.calls = []

    async def remote(self, fn, instance_id, query_list, retrieval_service_url, topk, timeout):
        self.calls.append(
            {
                "fn": fn,
                "instance_id": instance_id,
                "query_list": query_list,
                "retrieval_service_url": retrieval_service_url,
                "topk": topk,
                "timeout": timeout,
            }
        )
        return (
            '{"result": "ok"}',
            {
                "query_count": len(query_list),
                "status": "success",
                "total_results": 3,
                "api_request_error": None,
            },
        )


class _FakeExecutionPool:
    def __init__(self):
        self.execute = _FakeExecuteMethod()


class _FakeRemoteActor:
    def __init__(self):
        self.options_kwargs = None
        self.remote_kwargs = None

    def options(self, **kwargs):
        self.options_kwargs = kwargs
        return self

    def remote(self, **kwargs):
        self.remote_kwargs = kwargs
        return {"actor": "ok"}


def test_search_tool_accepts_qwen_query(monkeypatch):
    fake_pool = _FakeExecutionPool()
    calls = []
    monkeypatch.setattr(
        "verl.tools.search_tool.init_search_execution_pool",
        lambda num_workers, enable_global_rate_limit, rate_limit, mode: calls.append(
            (num_workers, enable_global_rate_limit, rate_limit, mode)
        )
        or fake_pool,
    )

    tool = SearchTool(
        config={
            "type": "native",
            "retrieval_service_url": "http://127.0.0.1:8000/retrieve",
            "num_workers": 4,
            "rate_limit": 4,
            "timeout": 30,
        },
        tool_schema=_build_schema(),
    )
    assert tool.execution_pool is None

    async def _run():
        instance_id, _ = await tool.create()
        return await tool.execute(instance_id, {"query": "What is Python?"})

    response, reward, metrics = asyncio.run(_run())

    assert response.text == '{"result": "ok"}'
    assert reward == 0.0
    assert metrics["status"] == "success"
    assert len(calls) == 1
    assert fake_pool.execute.calls[0]["query_list"] == ["What is Python?"]


def test_init_search_execution_pool_uses_named_actor_reuse(monkeypatch):
    fake_actor = _FakeRemoteActor()

    monkeypatch.setattr("verl.tools.search_tool.ray.remote", lambda cls: fake_actor)

    pool = init_search_execution_pool(num_workers=7, enable_global_rate_limit=True, rate_limit=11)

    assert pool == {"actor": "ok"}
    assert fake_actor.options_kwargs["name"] == "search-execution-worker-threadmode-7-11-1"
    assert fake_actor.options_kwargs["get_if_exists"] is True
    assert fake_actor.options_kwargs["max_concurrency"] == 7
    assert fake_actor.remote_kwargs == {
        "enable_global_rate_limit": True,
        "rate_limit": 11,
    }


def test_search_tool_direct_call_mode_skips_ray_pool(monkeypatch):
    fake_calls = []

    def _fake_search(*, retrieval_service_url, query_list, topk, concurrent_semaphore, timeout):
        fake_calls.append(
            {
                "retrieval_service_url": retrieval_service_url,
                "query_list": query_list,
                "topk": topk,
                "concurrent_semaphore": concurrent_semaphore,
                "timeout": timeout,
            }
        )
        return '{"result": "ok"}', {
            "query_count": len(query_list),
            "status": "success",
            "total_results": 3,
            "api_request_error": None,
        }

    monkeypatch.setattr(
        "verl.tools.search_tool.init_search_execution_pool",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Ray pool should not be initialized")),
    )
    monkeypatch.setattr("verl.tools.search_tool.perform_single_search_batch", _fake_search)

    tool = SearchTool(
        config={
            "type": "native",
            "retrieval_service_url": "http://127.0.0.1:8000/retrieve",
            "use_ray_pool": False,
            "num_workers": 4,
            "rate_limit": 4,
            "timeout": 30,
        },
        tool_schema=_build_schema(),
    )
    assert tool.execution_pool is None

    async def _run():
        instance_id, _ = await tool.create()
        return await tool.execute(instance_id, {"query": "What is Python?"})

    response, reward, metrics = asyncio.run(_run())

    assert response.text == '{"result": "ok"}'
    assert reward == 0.0
    assert metrics["status"] == "success"
    assert fake_calls[0]["query_list"] == ["What is Python?"]
