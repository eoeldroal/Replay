"""Tests for train.parquet RL column correctness.

The RL columns (extra_info, tools_kwargs) must contain all information
needed by SearchTool and the retrieval server:
1. row_index — unique integer identifier for server deck lookup
2. document_images — the 20-image deck for position-based resolution
3. reference_documents — GTI paths for NDCG reward computation

These fields flow through the pipeline:
  parquet extra_info → rl_dataset.py promotes tools_kwargs → tool_agent_loop.py
  → SearchTool receives create_kwargs → sends row_index to server
  → resolves position response using document_images
"""

import pytest


# ════════════════════════════════════════════════════════════════════════
# 1. extra_info structure
# ════════════════════════════════════════════════════════════════════════


class TestExtraInfoStructure:
    """extra_info must contain all fields needed by the RL pipeline."""

    def test_has_row_index(self, parquet_path):
        """extra_info must contain 'index' as the row_index for server lookup."""
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        for idx in [0, 100, len(df) - 1]:
            ei = df.iloc[idx]["extra_info"]
            assert "index" in ei, f"Row {idx}: extra_info missing 'index'"
            assert ei["index"] == idx, (
                f"Row {idx}: extra_info['index']={ei['index']}, expected {idx}"
            )

    def test_has_reference_documents(self, parquet_path):
        """extra_info must contain 'reference_documents' for NDCG reward.

        Note: parquet round-trip converts Python lists inside dicts to numpy
        arrays. This is safe because format_ndcg_reward._to_list() explicitly
        handles numpy arrays via .tolist() (line 129-131).
        """
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        for idx in [0, 100, len(df) - 1]:
            ei = df.iloc[idx]["extra_info"]
            assert "reference_documents" in ei
            refs = ei["reference_documents"]
            if hasattr(refs, "tolist"):
                refs = refs.tolist()
            assert isinstance(refs, list) and len(refs) > 0

    def test_reference_documents_match_gti(self, parquet_path):
        """extra_info['reference_documents'] must equal the gti column."""
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        for idx in range(min(50, len(df))):
            ei = df.iloc[idx]["extra_info"]
            gti = df.iloc[idx]["gti"]
            if hasattr(gti, "tolist"):
                gti = gti.tolist()
            if not isinstance(gti, list):
                gti = [gti]
            assert ei["reference_documents"] == gti, (
                f"Row {idx}: reference_documents={ei['reference_documents']}, gti={gti}"
            )

    def test_has_document_images_in_extra_info(self, parquet_path):
        """extra_info must contain 'document_images' — the 20-image deck.

        This is needed by SearchTool to resolve position-based server responses
        to actual image paths without querying the parquet again.

        Note: parquet round-trip converts lists to numpy arrays. Safe because
        search_tool.py only uses indexing (deck[position]) which works on both.
        """
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        for idx in [0, 100, len(df) - 1]:
            ei = df.iloc[idx]["extra_info"]
            assert "document_images" in ei, (
                f"Row {idx}: extra_info missing 'document_images'. "
                f"Keys present: {list(ei.keys())}"
            )
            deck = ei["document_images"]
            if hasattr(deck, "tolist"):
                deck = deck.tolist()
            assert isinstance(deck, list) and len(deck) == 20, (
                f"Row {idx}: document_images should be list of 20, got {type(deck).__name__} len={len(deck)}"
            )

    def test_document_images_matches_column(self, parquet_path):
        """extra_info['document_images'] must match the document_images column."""
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        for idx in range(min(50, len(df))):
            ei = df.iloc[idx]["extra_info"]
            if "document_images" not in ei:
                pytest.skip("document_images not yet in extra_info")
            ei_val = ei["document_images"]
            col_val = df.iloc[idx]["document_images"]
            # parquet round-trip may return numpy arrays; normalize both
            if hasattr(ei_val, "tolist"):
                ei_val = ei_val.tolist()
            if hasattr(col_val, "tolist"):
                col_val = col_val.tolist()
            assert ei_val == col_val, f"Row {idx}: mismatch"


# ════════════════════════════════════════════════════════════════════════
# 2. tools_kwargs → create_kwargs structure
# ════════════════════════════════════════════════════════════════════════


class TestCreateKwargsStructure:
    """tools_kwargs['search']['create_kwargs'] must provide SearchTool
    with all information needed for server communication and path resolution."""

    def _get_create_kwargs(self, df, idx):
        ei = df.iloc[idx]["extra_info"]
        return ei["tools_kwargs"]["search"]["create_kwargs"]

    def test_has_row_index_in_create_kwargs(self, parquet_path):
        """create_kwargs must contain 'row_index' for server deck identification.

        SearchTool sends this to the server instead of split('_')[-1].
        """
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        for idx in [0, 100, len(df) - 1]:
            ck = self._get_create_kwargs(df, idx)
            assert "row_index" in ck, (
                f"Row {idx}: create_kwargs missing 'row_index'. "
                f"Keys present: {list(ck.keys())}"
            )
            assert ck["row_index"] == idx

    def test_has_document_images_in_create_kwargs(self, parquet_path):
        """create_kwargs must contain 'document_images' for position resolution.

        When server returns position indices, SearchTool uses
        create_kwargs['document_images'][position] to get the image path.
        """
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        for idx in [0, 100, len(df) - 1]:
            ck = self._get_create_kwargs(df, idx)
            assert "document_images" in ck, (
                f"Row {idx}: create_kwargs missing 'document_images'. "
                f"Keys present: {list(ck.keys())}"
            )
            assert len(ck["document_images"]) == 20

    def test_preserves_existing_create_kwargs(self, parquet_path):
        """Existing fields (ground_truth, question, data_source) must still exist."""
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        ck = self._get_create_kwargs(df, 0)
        assert "ground_truth" in ck
        assert "question" in ck
        assert "data_source" in ck

    def test_data_source_is_vdr_lns(self, parquet_path):
        """All samples should have data_source='VDR_lns'."""
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        for idx in [0, 1000, len(df) - 1]:
            ck = self._get_create_kwargs(df, idx)
            assert ck["data_source"] == "VDR_lns"


# ════════════════════════════════════════════════════════════════════════
# 3. Consistency between extra_info and columns
# ════════════════════════════════════════════════════════════════════════


class TestExtraInfoConsistency:
    """extra_info fields must be consistent with the parquet columns."""

    def test_index_matches_row_position(self, parquet_path):
        """extra_info['index'] must equal the DataFrame row index."""
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        # Check a spread of rows
        for idx in range(0, len(df), len(df) // 20):
            ei = df.iloc[idx]["extra_info"]
            assert ei["index"] == idx

    def test_question_matches_query_column(self, parquet_path):
        """extra_info['question'] must match the query column."""
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        for idx in range(min(50, len(df))):
            ei = df.iloc[idx]["extra_info"]
            assert ei["question"] == df.iloc[idx]["query"]
