"""Unit tests for SearchTool — create_kwargs flow and position-based resolution.

These tests verify the actual SearchTool class behavior:
1. create() stores row_index and document_images from create_kwargs
2. execute() sends row_index (not split("_")[-1]) to the server
3. execute() resolves position indices to relative doc_id paths
4. image_paths stores relative paths matching reference_documents format
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Helper to load SearchTool without full verl import chain ──

def _load_search_tool_module():
    """Import search_tool.py directly to avoid ray dependency."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "search_tool",
        "verl/tools/LNS/search_tool.py",
        submodule_search_locations=[],
    )
    # Stub verl.tools.base_tool and verl.tools.schemas
    import sys
    import types

    # Minimal stubs
    if "verl" not in sys.modules:
        verl_stub = types.ModuleType("verl")
        sys.modules["verl"] = verl_stub
    if "verl.tools" not in sys.modules:
        tools_stub = types.ModuleType("verl.tools")
        sys.modules["verl.tools"] = tools_stub
    if "verl.tools.base_tool" not in sys.modules:
        base_tool_stub = types.ModuleType("verl.tools.base_tool")

        class BaseTool:
            def __init__(self, config, tool_schema):
                self.config = config
                self.tool_schema = tool_schema
                self.name = "search"

            async def create(self, instance_id=None, **kwargs):
                from uuid import uuid4
                if instance_id is None:
                    return str(uuid4()), None
                return instance_id, None

        base_tool_stub.BaseTool = BaseTool
        sys.modules["verl.tools.base_tool"] = base_tool_stub
    if "verl.tools.schemas" not in sys.modules:
        schemas_stub = types.ModuleType("verl.tools.schemas")

        class ToolResponse:
            def __init__(self, text="", image=None):
                self.text = text
                self.image = image or []

        schemas_stub.ToolResponse = ToolResponse
        sys.modules["verl.tools.schemas"] = schemas_stub
    if "verl.utils" not in sys.modules:
        sys.modules["verl.utils"] = types.ModuleType("verl.utils")
    if "verl.utils.rollout_trace" not in sys.modules:
        rt_stub = types.ModuleType("verl.utils.rollout_trace")
        rt_stub.rollout_trace_op = lambda f: f  # no-op decorator
        sys.modules["verl.utils.rollout_trace"] = rt_stub

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def search_module():
    return _load_search_tool_module()


@pytest.fixture
def tool_config():
    return {
        "retrieval_service_url": "http://localhost:5002/search",
        "timeout": 10,
        "k": 5,
        "rate_limit": 10,
        "local_image_root": "/tmp/test_corpus/img",
    }


@pytest.fixture
def sample_create_kwargs():
    """Simulates what comes from parquet extra_info → tools_kwargs → create_kwargs."""
    return {
        "row_index": 42,
        "document_images": [
            "slidevqa/accel-deck_95/slide_1_1024.jpg",
            "slidevqa/accel-deck_95/slide_2_1024.jpg",
            "slidevqa/accel-deck_95/slide_3_1024.jpg",
            "slidevqa/accel-deck_95/slide_4_1024.jpg",  # GTI at position 3
            "slidevqa/accel-deck_95/slide_5_1024.jpg",
            "slidevqa/accel-deck_95/slide_6_1024.jpg",
            "slidevqa/accel-deck_95/slide_7_1024.jpg",
            "slidevqa/accel-deck_95/slide_8_1024.jpg",
            "slidevqa/accel-deck_95/slide_9_1024.jpg",
            "slidevqa/accel-deck_95/slide_10_1024.jpg",
            "slidevqa/accel-deck_95/slide_11_1024.jpg",
            "slidevqa/accel-deck_95/slide_12_1024.jpg",
            "slidevqa/accel-deck_95/slide_13_1024.jpg",
            "slidevqa/accel-deck_95/slide_14_1024.jpg",
            "slidevqa/accel-deck_95/slide_15_1024.jpg",
            "slidevqa/accel-deck_95/slide_16_1024.jpg",
            "slidevqa/accel-deck_95/slide_17_1024.jpg",
            "slidevqa/accel-deck_95/slide_18_1024.jpg",
            "slidevqa/accel-deck_95/slide_19_1024.jpg",
            "slidevqa/accel-deck_95/slide_20_1024.jpg",
        ],
        "ground_truth": "42",
        "question": "What is the total revenue?",
        "data_source": "VDR_lns",
    }


def _make_agent_data(sample_id="slidevqa-0"):
    """Create a minimal AgentData-like object."""
    ad = SimpleNamespace()
    ad.sample_id = sample_id
    ad.extra_fields = {}
    return ad


# ════════════════════════════════════════════════════════════════════════
# 1. create() — stores create_kwargs per instance
# ════════════════════════════════════════════════════════════════════════


class TestSearchToolCreate:
    """SearchTool.create() must store row_index and document_images
    from create_kwargs for use during execute()."""

    def test_create_stores_row_index(self, search_module, tool_config, sample_create_kwargs):
        tool = search_module.SearchTool(tool_config, MagicMock())
        instance_id, _ = asyncio.get_event_loop().run_until_complete(
            tool.create(create_kwargs=sample_create_kwargs)
        )
        assert instance_id is not None
        assert tool._instance_data[instance_id]["row_index"] == 42

    def test_create_stores_document_images(self, search_module, tool_config, sample_create_kwargs):
        tool = search_module.SearchTool(tool_config, MagicMock())
        instance_id, _ = asyncio.get_event_loop().run_until_complete(
            tool.create(create_kwargs=sample_create_kwargs)
        )
        assert len(tool._instance_data[instance_id]["document_images"]) == 20

    def test_create_without_create_kwargs_still_works(self, search_module, tool_config):
        """Backward compatibility: create() without create_kwargs should not crash."""
        tool = search_module.SearchTool(tool_config, MagicMock())
        instance_id, _ = asyncio.get_event_loop().run_until_complete(
            tool.create()
        )
        assert instance_id is not None


# ════════════════════════════════════════════════════════════════════════
# 2. execute() — sends row_index, not split("_")[-1]
# ════════════════════════════════════════════════════════════════════════


class TestSearchToolPayload:
    """execute() must send row_index to the server, not split('_')[-1]."""

    def test_payload_contains_row_index(self, search_module, tool_config, sample_create_kwargs):
        """The HTTP payload must include row_index from create_kwargs."""
        tool = search_module.SearchTool(tool_config, MagicMock())
        instance_id, _ = asyncio.get_event_loop().run_until_complete(
            tool.create(create_kwargs=sample_create_kwargs)
        )

        captured_payload = {}

        async def mock_search(*args, **kwargs):
            # Server returns position 3 (the GTI slide)
            return [{"results": [{"position": 3, "score": 0.95}]}], None

        agent_data = _make_agent_data()

        with patch.object(search_module, "_run_search_call_with_limit", side_effect=mock_search):
            with patch.object(search_module, "call_search_api") as mock_api:
                # We need to capture what asyncio.to_thread calls
                original_to_thread = asyncio.to_thread

                async def capture_to_thread(func, *args, **kw):
                    # args: limiter, url, payload, timeout
                    if len(args) >= 3:
                        captured_payload["payload"] = args[2]
                    return [{"results": [{"position": 3, "score": 0.95}]}], None

                with patch("asyncio.to_thread", side_effect=capture_to_thread):
                    asyncio.get_event_loop().run_until_complete(
                        tool.execute(instance_id, {"query": "revenue"}, agent_data=agent_data)
                    )

        assert "payload" in captured_payload
        payload = captured_payload["payload"]
        assert isinstance(payload, list)
        assert payload[0].get("row_index") == 42
        # Must NOT contain the old collision-prone "id" field
        assert "id" not in payload[0] or payload[0].get("id") != "0"

    def test_payload_does_not_use_split_suffix(self, search_module, tool_config, sample_create_kwargs):
        """Verify the old split('_')[-1] pattern is not used."""
        tool = search_module.SearchTool(tool_config, MagicMock())
        instance_id, _ = asyncio.get_event_loop().run_until_complete(
            tool.create(create_kwargs=sample_create_kwargs)
        )

        agent_data = _make_agent_data(sample_id="infovqa-train_1")

        captured_payload = {}

        async def capture_to_thread(func, *args, **kw):
            if len(args) >= 3:
                captured_payload["payload"] = args[2]
            return [{"results": [{"position": 0, "score": 0.9}]}], None

        with patch("asyncio.to_thread", side_effect=capture_to_thread):
            asyncio.get_event_loop().run_until_complete(
                tool.execute(instance_id, {"query": "test"}, agent_data=agent_data)
            )

        payload = captured_payload["payload"]
        # "infovqa-train_1".split("_")[-1] == "1" — this must NOT be used
        assert payload[0].get("row_index") == 42  # from create_kwargs, not sample_id


# ════════════════════════════════════════════════════════════════════════
# 3. execute() — resolves position to relative doc_id path
# ════════════════════════════════════════════════════════════════════════


class TestSearchToolPositionResolution:
    """Server returns position indices. SearchTool resolves them using document_images."""

    def test_position_resolved_to_relative_path(self, search_module, tool_config, sample_create_kwargs):
        """Server returns position=3 → image_paths should contain slide_4's relative path."""
        tool = search_module.SearchTool(tool_config, MagicMock())
        instance_id, _ = asyncio.get_event_loop().run_until_complete(
            tool.create(create_kwargs=sample_create_kwargs)
        )

        agent_data = _make_agent_data()

        async def mock_to_thread(func, *args, **kw):
            return [{"results": [{"position": 3, "score": 0.95}]}], None

        # Mock os.path.exists and Image.open since test images don't exist on disk
        with patch("asyncio.to_thread", side_effect=mock_to_thread), \
             patch.object(search_module.os.path, "exists", return_value=True), \
             patch.object(search_module.Image, "open", return_value=MagicMock(convert=MagicMock(return_value="fake_img"))):
            asyncio.get_event_loop().run_until_complete(
                tool.execute(instance_id, {"query": "revenue"}, agent_data=agent_data)
            )

        image_paths = agent_data.extra_fields.get("image_paths", [])
        assert len(image_paths) == 1
        assert image_paths[0] == "slidevqa/accel-deck_95/slide_4_1024.jpg"

    def test_stored_path_is_relative_not_absolute(self, search_module, tool_config, sample_create_kwargs):
        """image_paths must contain relative paths, not /home/work/... absolute paths."""
        tool = search_module.SearchTool(tool_config, MagicMock())
        instance_id, _ = asyncio.get_event_loop().run_until_complete(
            tool.create(create_kwargs=sample_create_kwargs)
        )

        agent_data = _make_agent_data()

        async def mock_to_thread(func, *args, **kw):
            return [{"results": [{"position": 0, "score": 0.9}]}], None

        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            asyncio.get_event_loop().run_until_complete(
                tool.execute(instance_id, {"query": "test"}, agent_data=agent_data)
            )

        for path in agent_data.extra_fields.get("image_paths", []):
            assert not path.startswith("/"), f"Path should be relative, got: {path}"

    def test_invalid_position_skipped(self, search_module, tool_config, sample_create_kwargs):
        """Position outside 0-19 range should be skipped gracefully."""
        tool = search_module.SearchTool(tool_config, MagicMock())
        instance_id, _ = asyncio.get_event_loop().run_until_complete(
            tool.create(create_kwargs=sample_create_kwargs)
        )

        agent_data = _make_agent_data()

        async def mock_to_thread(func, *args, **kw):
            return [{"results": [{"position": 99, "score": 0.5}]}], None

        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            asyncio.get_event_loop().run_until_complete(
                tool.execute(instance_id, {"query": "test"}, agent_data=agent_data)
            )

        image_paths = agent_data.extra_fields.get("image_paths", [])
        assert len(image_paths) == 0

    def test_duplicate_position_skipped(self, search_module, tool_config, sample_create_kwargs):
        """If the same doc_id was already retrieved, skip it."""
        tool = search_module.SearchTool(tool_config, MagicMock())
        instance_id, _ = asyncio.get_event_loop().run_until_complete(
            tool.create(create_kwargs=sample_create_kwargs)
        )

        agent_data = _make_agent_data()
        # Pre-populate: slide_4 already retrieved
        agent_data.extra_fields["image_paths"] = [
            "slidevqa/accel-deck_95/slide_4_1024.jpg"
        ]

        async def mock_to_thread(func, *args, **kw):
            # Server returns position 3 (slide_4) again
            return [{"results": [{"position": 3, "score": 0.95}]}], None

        with patch("asyncio.to_thread", side_effect=mock_to_thread):
            asyncio.get_event_loop().run_until_complete(
                tool.execute(instance_id, {"query": "revenue"}, agent_data=agent_data)
            )

        # Should still be 1 (no duplicate added)
        assert len(agent_data.extra_fields["image_paths"]) == 1
