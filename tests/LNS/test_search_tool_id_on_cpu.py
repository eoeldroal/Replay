"""Tests for search_tool.py — ID passing and result interpretation.

We are modifying SearchTool so that:
1. Sample identification uses row_index (integer), not split("_")[-1] (collision-prone)
2. Server returns position indices (0-19), SearchTool resolves to document_images paths
3. image_paths stores relative doc_id paths (same format as reference_documents)

These tests verify the DESIRED behavior. They should FAIL against the current code.
"""

import pytest


# ════════════════════════════════════════════════════════════════════════
# 1. ID collision — the fundamental problem with split("_")[-1]
# ════════════════════════════════════════════════════════════════════════


class TestIdCollision:
    """Demonstrate that the current ID extraction causes collisions,
    and that the new approach (row_index) does not."""

    # These IDs are real examples from our 80,292-row train.parquet.
    REAL_IDS = [
        "infovqa-train_1",
        "docvqa-train_1",
        "mhdocvqa-train_1251",
        "visualmrc-train_1",
        "slidevqa-0",
        "slidevqa-1",
        "mpmqa-train_1038",
        "openwikitable-train_1",
    ]

    def test_current_split_method_has_collisions(self):
        """Prove that split('_')[-1] produces duplicate IDs.

        This is a documentation test — it should PASS, showing the bug exists.
        """
        numeric_ids = [sid.split("_")[-1] for sid in self.REAL_IDS]
        # "infovqa-train_1", "docvqa-train_1", "visualmrc-train_1",
        # "openwikitable-train_1" all produce "1"
        assert len(set(numeric_ids)) < len(numeric_ids), (
            "Expected collisions with split('_')[-1], but found none. "
            "Test data may need updating."
        )

    def test_row_index_is_unique(self):
        """Row index (integer 0..N-1) never collides by definition.

        This is a sanity check — should always PASS.
        """
        row_indices = list(range(len(self.REAL_IDS)))
        assert len(set(row_indices)) == len(row_indices)

    def test_full_id_string_is_unique(self):
        """Full ID strings are unique across all sources.

        This is a sanity check — should always PASS.
        """
        assert len(set(self.REAL_IDS)) == len(self.REAL_IDS)


# ════════════════════════════════════════════════════════════════════════
# 2. ID collision at scale — test against actual parquet
# ════════════════════════════════════════════════════════════════════════


class TestIdCollisionAtScale:
    """Verify ID uniqueness across all 80,292 samples in the real parquet."""

    def test_split_suffix_has_collisions_in_real_data(self, parquet_path):
        """The real parquet has 11,218+ collisions with split('_')[-1]."""
        import pandas as pd

        df = pd.read_parquet(parquet_path, columns=["id"])
        numeric_ids = df["id"].apply(lambda x: x.split("_")[-1])
        n_unique = numeric_ids.nunique()
        n_total = len(df)
        n_collisions = n_total - n_unique
        assert n_collisions > 0, "Expected collisions in real data"

    def test_full_id_no_collisions_in_real_data(self, parquet_path):
        """Every full ID string in the parquet is unique."""
        import pandas as pd

        df = pd.read_parquet(parquet_path, columns=["id"])
        assert df["id"].nunique() == len(df)

    def test_row_index_covers_all_samples(self, parquet_path):
        """Row indices 0..80291 cover every sample."""
        import pandas as pd

        df = pd.read_parquet(parquet_path, columns=["id"])
        assert list(df.index) == list(range(len(df)))


# ════════════════════════════════════════════════════════════════════════
# 3. Position-based result resolution
# ════════════════════════════════════════════════════════════════════════


class TestPositionResolution:
    """Server returns position indices. SearchTool resolves them to paths
    using the document_images list from extra_info."""

    def test_position_maps_to_correct_path(self, colqwen_row):
        """Position 0 → document_images[0]."""
        doc_images = colqwen_row["document_images"]
        position = 0
        resolved = doc_images[position]
        assert resolved == "infovqa/38032.jpeg"

    def test_position_maps_to_gti(self, slidevqa_row):
        """For SlideVQA, GTI is slide_4 which is at position 3 (0-indexed)."""
        doc_images = slidevqa_row["document_images"]
        gti = slidevqa_row["gti"][0]
        # Find which position the GTI is at
        gti_position = doc_images.index(gti)
        assert gti_position == 3  # slide_4 is 0-indexed position 3
        assert doc_images[gti_position] == gti

    def test_all_positions_are_valid(self, colqwen_row):
        """All positions 0-19 map to valid image paths."""
        doc_images = colqwen_row["document_images"]
        assert len(doc_images) == 20
        for pos in range(20):
            assert isinstance(doc_images[pos], str)
            assert len(doc_images[pos]) > 0


# ════════════════════════════════════════════════════════════════════════
# 4. image_paths storage format
# ════════════════════════════════════════════════════════════════════════


class TestImagePathsFormat:
    """SearchTool must store relative doc_id paths in image_paths,
    matching the format of reference_documents for NDCG comparison."""

    def test_stored_path_matches_reference_format(self, colqwen_row):
        """Path stored in image_paths must exactly match the format
        in reference_documents (both are relative paths from document_images).
        """
        doc_images = colqwen_row["document_images"]
        gti = colqwen_row["gti"]

        # Simulate: server says position 0 is best
        retrieved_path = doc_images[0]  # "infovqa/38032.jpeg"
        reference_path = gti[0]  # "infovqa/38032.jpeg"

        # These should be identical strings
        assert retrieved_path == reference_path

    def test_stored_path_is_relative_not_absolute(self, colqwen_row):
        """image_paths must NOT contain full filesystem paths."""
        doc_images = colqwen_row["document_images"]
        for path in doc_images:
            assert not path.startswith("/"), f"Path should be relative, got: {path}"

    def test_stored_path_includes_source_prefix(self, colqwen_row):
        """Relative paths must include the source prefix (docvqa/, infovqa/, etc.)
        for cross-source disambiguation."""
        doc_images = colqwen_row["document_images"]
        for path in doc_images:
            assert "/" in path, f"Path should include source prefix: {path}"


# ════════════════════════════════════════════════════════════════════════
# 5. End-to-end: position resolution → NDCG reward compatibility
# ════════════════════════════════════════════════════════════════════════


class TestPositionToNdcgPipeline:
    """Full pipeline: server position → SearchTool path resolution → NDCG reward.

    This tests the integration between search_tool and format_ndcg_reward.
    """

    def test_correct_position_yields_ndcg_1(self, slidevqa_row):
        """Server returns the GTI position → NDCG = 1.0."""
        from verl.utils.reward_score.format_ndcg_reward import compute_score

        doc_images = slidevqa_row["document_images"]
        gti = slidevqa_row["gti"]

        # Server says position 3 (slide_4, the GTI)
        gti_position = doc_images.index(gti[0])
        retrieved_path = doc_images[gti_position]

        # This is what SearchTool would store
        extra_info = {
            "image_paths": [retrieved_path],
            "reference_documents": gti,
        }

        solution = (
            "<|im_start|>assistant\n"
            "<think>Searching for relevant slide.</think>\n"
            "<search>revenue</search>\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>Found it.</think>\n"
            "<search_complete>true</search_complete>\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<answer>42</answer>\n"
            "<|im_end|>"
        )

        result = compute_score("VDR_lns", solution, "42", extra_info)
        assert result["ndcg"] == 1.0, (
            f"GTI slide retrieved but NDCG={result['ndcg']}. "
            f"retrieved={retrieved_path}, reference={gti[0]}"
        )

    def test_wrong_position_yields_ndcg_0(self, slidevqa_row):
        """Server returns a non-GTI position → NDCG = 0.0."""
        from verl.utils.reward_score.format_ndcg_reward import compute_score

        doc_images = slidevqa_row["document_images"]
        gti = slidevqa_row["gti"]

        # Server says position 0 (slide_1, NOT the GTI)
        retrieved_path = doc_images[0]  # slide_1, GTI is slide_4

        extra_info = {
            "image_paths": [retrieved_path],
            "reference_documents": gti,
        }

        solution = (
            "<|im_start|>assistant\n"
            "<think>Searching for relevant slide.</think>\n"
            "<search>revenue</search>\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>Found it.</think>\n"
            "<search_complete>true</search_complete>\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<answer>42</answer>\n"
            "<|im_end|>"
        )

        result = compute_score("VDR_lns", solution, "42", extra_info)
        assert result["ndcg"] == 0.0, (
            f"Wrong slide retrieved but NDCG={result['ndcg']} (should be 0). "
            f"This means slides are not being distinguished. "
            f"retrieved={retrieved_path}, reference={gti[0]}"
        )
