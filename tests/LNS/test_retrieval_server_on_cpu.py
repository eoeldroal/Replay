"""CPU-only tests for the ColQwen2 retrieval server.

Tests cover:
1. EmbeddingStore: load deck_index, resolve row_index → 20 embeddings
2. MaxSimScorer: ColBERT-style late-interaction scoring correctness
3. Request/response format: payload parsing, position/score sorting
4. Edge cases: invalid row_index, empty query
5. Integration tests (marked, require H100 with GPU + running server)

Run:
    conda run -n verl python -m pytest tests/LNS/test_retrieval_server_on_cpu.py -v -k "not integration"
"""

import os
import sys

import pytest
import torch

# ── Paths ──────────────────────────────────────────────────────────────

VERL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SERVER_DIR = os.path.join(VERL_ROOT, "RL_side_1_LNS")
sys.path.insert(0, VERL_ROOT)

DECK_INDEX_PATH = os.path.join(
    VERL_ROOT, "data", "Visual_Document_Rag", "VDR_final", "deck_index.pt"
)
EMBEDDING_BASE = os.path.join(
    VERL_ROOT, "data", "Visual_Document_Rag", "VDR_processed_filtered_2",
    "embeddings", "colqwen2",
)


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def deck_index():
    """Load the real deck_index.pt. Skip if not present."""
    if not os.path.exists(DECK_INDEX_PATH):
        pytest.skip(f"deck_index.pt not found: {DECK_INDEX_PATH}")
    return torch.load(DECK_INDEX_PATH, map_location="cpu", weights_only=False)


@pytest.fixture
def embedding_store():
    """Create an EmbeddingStore instance. Skip if data not available."""
    if not os.path.exists(DECK_INDEX_PATH):
        pytest.skip("deck_index.pt not found")
    if not os.path.isdir(EMBEDDING_BASE):
        pytest.skip("Embedding base dir not found")

    from RL_side_1_LNS.retrieval_server import EmbeddingStore
    store = EmbeddingStore(DECK_INDEX_PATH, EMBEDDING_BASE)
    return store


@pytest.fixture
def maxsim_scorer():
    """Create a MaxSimScorer instance."""
    from RL_side_1_LNS.retrieval_server import MaxSimScorer
    return MaxSimScorer()


# ══════════════════════════════════════════════════════════════════════
# 1. EmbeddingStore tests
# ══════════════════════════════════════════════════════════════════════

class TestEmbeddingStore:
    """Tests for EmbeddingStore: loading deck_index and retrieving embeddings."""

    def test_deck_index_structure(self, deck_index):
        """deck_index.pt has expected keys."""
        assert "deck_locations" in deck_index
        assert "datasets" in deck_index
        assert "shard_paths" in deck_index
        assert len(deck_index["deck_locations"]) == 80_292

    def test_deck_index_row_has_20_locations(self, deck_index):
        """Each row has exactly 20 location tuples."""
        for row_idx in [0, 100, 50000, 80291]:
            locs = deck_index["deck_locations"][row_idx]
            assert len(locs) == 20, f"Row {row_idx} has {len(locs)} locations"

    def test_deck_index_location_tuple_format(self, deck_index):
        """Each location is (dataset_idx, shard_idx, local_idx) with valid ranges."""
        ds_count = len(deck_index["datasets"])
        loc = deck_index["deck_locations"][0][0]
        assert len(loc) == 3
        ds_idx, shard_idx, local_idx = loc
        assert 0 <= ds_idx < ds_count
        assert shard_idx >= 0
        assert local_idx >= 0

    def test_store_num_rows(self, embedding_store):
        """EmbeddingStore reports correct number of rows."""
        assert embedding_store.num_rows == 80_292

    def test_store_get_deck_embeddings_returns_20(self, embedding_store):
        """get_deck_embeddings(row_index) returns exactly 20 tensors."""
        embs = embedding_store.get_deck_embeddings(0)
        assert len(embs) == 20

    def test_store_embedding_shape(self, embedding_store):
        """Each embedding is a 2D tensor [T, 128]."""
        embs = embedding_store.get_deck_embeddings(0)
        for i, emb in enumerate(embs):
            assert emb.ndim == 2, f"Embedding {i} has {emb.ndim} dims, expected 2"
            assert emb.shape[1] == 128, f"Embedding {i} has dim {emb.shape[1]}, expected 128"

    def test_store_different_rows_give_different_embeddings(self, embedding_store):
        """Different row_index values give different embedding sets."""
        embs_0 = embedding_store.get_deck_embeddings(0)
        embs_1 = embedding_store.get_deck_embeddings(1)
        # At least one embedding should differ (different deck compositions)
        any_different = any(
            not torch.equal(embs_0[i], embs_1[i]) for i in range(20)
        )
        assert any_different, "Row 0 and Row 1 have identical embeddings"

    def test_store_invalid_row_index_raises(self, embedding_store):
        """Invalid row_index raises an error."""
        with pytest.raises((IndexError, ValueError, KeyError)):
            embedding_store.get_deck_embeddings(-1)
        with pytest.raises((IndexError, ValueError, KeyError)):
            embedding_store.get_deck_embeddings(80_292)

    def test_store_last_row(self, embedding_store):
        """Last valid row_index (80291) works."""
        embs = embedding_store.get_deck_embeddings(80_291)
        assert len(embs) == 20


# ══════════════════════════════════════════════════════════════════════
# 2. MaxSimScorer tests
# ══════════════════════════════════════════════════════════════════════

class TestMaxSimScorer:
    """Tests for ColBERT-style MaxSim scoring."""

    def test_identity_scores_highest(self, maxsim_scorer):
        """A query identical to a doc should score highest against that doc."""
        # Simulate: query [3, 128], 3 docs [P, 128]
        torch.manual_seed(42)
        doc0 = torch.randn(10, 128)
        doc1 = torch.randn(10, 128)
        doc2 = torch.randn(10, 128)

        # Query = first doc (should score highest for doc0)
        query = doc0[:3]  # Take first 3 "tokens" as query
        results = maxsim_scorer.score(query, [doc0, doc1, doc2])

        # Results sorted descending by score
        assert results[0][0] == 0, f"Expected position 0 first, got {results[0][0]}"
        assert results[0][1] > results[1][1], "Identity doc should have highest score"

    def test_score_returns_all_positions(self, maxsim_scorer):
        """score() returns position and score for all docs."""
        query = torch.randn(5, 128)
        docs = [torch.randn(10, 128) for _ in range(20)]
        results = maxsim_scorer.score(query, docs)
        assert len(results) == 20
        positions = [r[0] for r in results]
        assert sorted(positions) == list(range(20)), "All positions 0-19 should be present"

    def test_score_descending_order(self, maxsim_scorer):
        """Results are sorted in descending score order."""
        query = torch.randn(5, 128)
        docs = [torch.randn(10, 128) for _ in range(20)]
        results = maxsim_scorer.score(query, docs)
        scores = [r[1] for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"Score[{i}]={scores[i]} < Score[{i+1}]={scores[i+1]}"

    def test_maxsim_manual_calculation(self, maxsim_scorer):
        """Verify MaxSim matches manual calculation."""
        # Simple case: query [2, 128], doc [3, 128]
        torch.manual_seed(0)
        query = torch.randn(2, 128)
        doc = torch.randn(3, 128)

        # Manual MaxSim: for each query token, max similarity across doc tokens, then sum
        sim = query @ doc.T  # [2, 3]
        expected_score = sim.max(dim=1).values.sum().item()

        results = maxsim_scorer.score(query, [doc])
        actual_score = results[0][1]

        assert abs(actual_score - expected_score) < 1e-4, (
            f"MaxSim mismatch: expected {expected_score}, got {actual_score}"
        )

    def test_score_with_variable_length_docs(self, maxsim_scorer):
        """MaxSim works with variable-length document embeddings."""
        query = torch.randn(5, 128)
        docs = [
            torch.randn(100, 128),  # short doc
            torch.randn(700, 128),  # typical image patch count
            torch.randn(780, 128),  # long doc
        ]
        results = maxsim_scorer.score(query, docs)
        assert len(results) == 3
        # All positions present
        assert sorted(r[0] for r in results) == [0, 1, 2]

    def test_score_single_doc(self, maxsim_scorer):
        """Scoring with a single doc works."""
        query = torch.randn(5, 128)
        doc = torch.randn(100, 128)
        results = maxsim_scorer.score(query, [doc])
        assert len(results) == 1
        assert results[0][0] == 0


# ══════════════════════════════════════════════════════════════════════
# 3. Request/Response format tests
# ══════════════════════════════════════════════════════════════════════

class TestRequestResponseFormat:
    """Tests for payload parsing and response format."""

    def test_valid_payload_parsing(self):
        """Server should parse the SearchTool payload format."""
        from RL_side_1_LNS.retrieval_server import parse_search_payload

        payload = [{"query": "What is the website?", "request_idx": 0, "row_index": 42}]
        items = parse_search_payload(payload)
        assert len(items) == 1
        assert items[0]["query"] == "What is the website?"
        assert items[0]["row_index"] == 42

    def test_response_format(self):
        """Response matches SearchTool expected format."""
        from RL_side_1_LNS.retrieval_server import format_search_response

        # Simulate scored results: list of (position, score) tuples per request
        scored_results = [
            [(3, 0.95), (7, 0.82), (0, 0.75), (1, 0.60), (15, 0.55),
             (2, 0.50), (4, 0.45), (5, 0.40), (6, 0.35), (8, 0.30),
             (9, 0.28), (10, 0.25), (11, 0.22), (12, 0.20), (13, 0.18),
             (14, 0.15), (16, 0.12), (17, 0.10), (18, 0.08), (19, 0.05)],
        ]
        response = format_search_response(scored_results)

        assert isinstance(response, list)
        assert len(response) == 1
        result = response[0]
        assert "results" in result
        results = result["results"]
        assert len(results) == 20
        # First result should be position 3 with score 0.95
        assert results[0]["position"] == 3
        assert abs(results[0]["score"] - 0.95) < 1e-6

    def test_response_positions_are_ints(self):
        """Positions in response must be integers."""
        from RL_side_1_LNS.retrieval_server import format_search_response

        scored_results = [[(i, float(20 - i)) for i in range(20)]]
        response = format_search_response(scored_results)
        for item in response[0]["results"]:
            assert isinstance(item["position"], int)

    def test_response_scores_are_floats(self):
        """Scores in response must be floats."""
        from RL_side_1_LNS.retrieval_server import format_search_response

        scored_results = [[(i, float(20 - i)) for i in range(20)]]
        response = format_search_response(scored_results)
        for item in response[0]["results"]:
            assert isinstance(item["score"], float)

    def test_empty_payload_returns_empty(self):
        """Empty payload returns empty response."""
        from RL_side_1_LNS.retrieval_server import parse_search_payload

        items = parse_search_payload([])
        assert items == []

    def test_batch_payload(self):
        """Multiple items in a single payload are parsed correctly."""
        from RL_side_1_LNS.retrieval_server import parse_search_payload

        payload = [
            {"query": "q1", "request_idx": 0, "row_index": 10},
            {"query": "q2", "request_idx": 1, "row_index": 20},
        ]
        items = parse_search_payload(payload)
        assert len(items) == 2
        assert items[0]["row_index"] == 10
        assert items[1]["row_index"] == 20


# ══════════════════════════════════════════════════════════════════════
# 4. Edge case tests
# ══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases and error handling."""

    def test_maxsim_empty_docs_raises(self, maxsim_scorer):
        """Scoring with no docs should raise or return empty."""
        query = torch.randn(5, 128)
        with pytest.raises((ValueError, IndexError)):
            maxsim_scorer.score(query, [])

    def test_parse_payload_missing_query(self):
        """Payload missing 'query' is handled."""
        from RL_side_1_LNS.retrieval_server import parse_search_payload

        payload = [{"request_idx": 0, "row_index": 42}]
        items = parse_search_payload(payload)
        # Should either skip the item or set query to None/empty
        if items:
            assert items[0].get("query") is None or items[0].get("query") == ""

    def test_parse_payload_missing_row_index(self):
        """Payload missing 'row_index' is handled."""
        from RL_side_1_LNS.retrieval_server import parse_search_payload

        payload = [{"query": "test", "request_idx": 0}]
        items = parse_search_payload(payload)
        if items:
            assert items[0].get("row_index") is None


# ══════════════════════════════════════════════════════════════════════
# 5. Integration tests (require running server on H100)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring a running server. Skip in CI."""

    SERVER_URL = "http://localhost:5002"

    @pytest.fixture(autouse=True)
    def _skip_without_server(self):
        """Skip if server is not running."""
        try:
            import requests
            resp = requests.get(f"{self.SERVER_URL}/health", timeout=2)
            if resp.status_code != 200:
                pytest.skip("Server not healthy")
        except Exception:
            pytest.skip("Server not reachable")

    def test_health_endpoint(self):
        """Server /health returns 200."""
        import requests
        resp = requests.get(f"{self.SERVER_URL}/health", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "ok"

    def test_e2e_search(self):
        """End-to-end search returns positions and scores."""
        import requests
        payload = [{"query": "What is the website?", "request_idx": 0, "row_index": 0}]
        resp = requests.post(f"{self.SERVER_URL}/search", json=payload, timeout=30)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list) and len(data) == 1
        result = data[0]
        assert "results" in result
        results = result["results"]
        assert len(results) == 20
        # All positions 0-19 present
        positions = [r["position"] for r in results]
        assert sorted(positions) == list(range(20))
        # Scores are in descending order
        scores = [r["score"] for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_e2e_last_row(self):
        """Search with last valid row_index works."""
        import requests
        payload = [{"query": "test", "request_idx": 0, "row_index": 80291}]
        resp = requests.post(f"{self.SERVER_URL}/search", json=payload, timeout=30)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data[0]["results"]) == 20
