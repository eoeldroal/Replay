"""Tests for format_ndcg_reward.py — NDCG reward computation.

We are modifying _normalize_doc_id and related functions so that:
1. SlideVQA slides in the same deck are DISTINGUISHABLE (not all → "1024")
2. All image extensions (.jpg, .png, .jpeg) are handled consistently
3. Retrieved paths and reference paths in the same format match correctly
4. Cross-source paths (docvqa, vdr, slidevqa, infovqa) never false-match

These tests describe the DESIRED behavior after our changes.
They should FAIL against the current (buggy) code.
"""

import pytest

from verl.utils.reward_score.format_ndcg_reward import (
    _basename_no_ext,
    _extract_reference,
    _extract_retrieved,
    _ndcg,
    _normalize_doc_id,
    compute_score,
)


# ════════════════════════════════════════════════════════════════════════
# 1. _normalize_doc_id — the core comparison function
# ════════════════════════════════════════════════════════════════════════


class TestNormalizeDocId:
    """_normalize_doc_id must produce values that uniquely identify images."""

    # ── SlideVQA: the critical bug fix ──

    def test_slidevqa_different_slides_produce_different_values(self):
        """slide_1 and slide_4 in the same deck MUST normalize differently.

        Current bug: both normalize to "1024" because _normalize_doc_id
        extracts the last numeric segment after '_'.
        """
        slide_1 = _normalize_doc_id("slidevqa/deck_name/slide_1_1024.jpg")
        slide_4 = _normalize_doc_id("slidevqa/deck_name/slide_4_1024.jpg")
        assert slide_1 != slide_4

    def test_slidevqa_all_20_slides_are_unique(self):
        """All 20 slides in a SlideVQA deck must normalize to unique values."""
        deck = "slidevqa/accel-deck_95"
        normalized = set()
        for i in range(1, 21):
            val = _normalize_doc_id(f"{deck}/slide_{i}_1024.jpg")
            normalized.add(val)
        assert len(normalized) == 20, f"Expected 20 unique values, got {len(normalized)}: {normalized}"

    # ── Extension stripping consistency ──

    def test_strips_jpg_extension(self):
        result = _normalize_doc_id("slidevqa/deck/slide_4_1024.jpg")
        assert not result.endswith(".jpg")

    def test_strips_png_extension(self):
        result = _normalize_doc_id("docvqa/hzym0020_14.png")
        assert not result.endswith(".png")

    def test_strips_jpeg_extension(self):
        result = _normalize_doc_id("infovqa/38032.jpeg")
        assert not result.endswith(".jpeg")

    # ── Same image, same normalized value ──

    def test_same_path_normalizes_identically(self):
        """Retrieved and reference with identical path must match."""
        path = "docvqa/hzym0020_14.png"
        assert _normalize_doc_id(path) == _normalize_doc_id(path)

    # ── Different images, different normalized values ──

    def test_different_docvqa_images_differ(self):
        a = _normalize_doc_id("docvqa/hzym0020_14.png")
        b = _normalize_doc_id("docvqa/jybx0223_94.png")
        assert a != b

    def test_cross_source_images_differ(self):
        """Images from different sources must not collide."""
        docvqa = _normalize_doc_id("docvqa/hzym0020_14.png")
        infovqa = _normalize_doc_id("infovqa/38032.jpeg")
        vdr = _normalize_doc_id("vdr/4f0d89780c8fb747ca03398424d76635f5f270da.png")
        assert len({docvqa, infovqa, vdr}) == 3

    # ── VDR hash-based IDs (no underscore) ──

    def test_vdr_hash_id_preserved(self):
        """VDR images with hash names must normalize to a unique identifier."""
        a = _normalize_doc_id("vdr/4f0d89780c8fb747ca03398424d76635f5f270da.png")
        b = _normalize_doc_id("vdr/bd5bc38821f72b8dd88416988e37835277943bd9.png")
        assert a != b

    # ── Path structure preservation ──

    def test_preserves_directory_prefix(self):
        """Normalization must keep the source directory to prevent cross-source collision.

        Without directory prefix: "hzym0020_14" (from docvqa) could match
        an identically-named file from another source.
        """
        result = _normalize_doc_id("docvqa/hzym0020_14.png")
        assert "docvqa" in result


# ════════════════════════════════════════════════════════════════════════
# 2. _basename_no_ext — extraction from full paths
# ════════════════════════════════════════════════════════════════════════


class TestBasenameNoExt:
    """_basename_no_ext extracts comparable identifiers from image paths."""

    def test_handles_png_extension(self):
        """Must strip .png, not just .jpg.

        Current bug: only strips .jpg, leaving .png intact.
        """
        result = _basename_no_ext("/full/path/to/corpus/img/docvqa/hzym0020_14.png")
        assert not result.endswith(".png")

    def test_handles_jpeg_extension(self):
        result = _basename_no_ext("/full/path/to/infovqa/38032.jpeg")
        assert not result.endswith(".jpeg")

    def test_handles_jpg_extension(self):
        result = _basename_no_ext("/full/path/to/slidevqa/deck/slide_4_1024.jpg")
        assert not result.endswith(".jpg")

    def test_relative_path_preserves_structure(self):
        """When given a relative path (not full local path), directory structure
        should be preserved for cross-source disambiguation.

        In the new system, SearchTool stores relative paths like
        'docvqa/hzym0020_14.png', not full local paths.
        """
        result = _basename_no_ext("docvqa/hzym0020_14.png")
        # After _basename_no_ext and then _normalize_doc_id, the result
        # must be distinguishable from other sources.
        # At minimum, the source prefix should survive or the full
        # path should be returned without extension.
        assert "docvqa" in result or "hzym0020_14" in result


# ════════════════════════════════════════════════════════════════════════
# 3. NDCG computation — end-to-end correctness
# ════════════════════════════════════════════════════════════════════════


class TestNdcgComputation:
    """_ndcg must correctly score retrieval quality."""

    def test_perfect_retrieval(self):
        """All golden docs retrieved in order → NDCG = 1.0."""
        assert _ndcg(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_no_match(self):
        """No golden docs retrieved → NDCG = 0.0."""
        assert _ndcg(["x", "y", "z"], ["a", "b", "c"]) == 0.0

    def test_partial_match(self):
        """One golden doc retrieved → 0 < NDCG < 1."""
        score = _ndcg(["a", "x", "y"], ["a", "b"])
        assert 0.0 < score < 1.0

    def test_single_golden_retrieved_first(self):
        """Single golden doc at position 1 → NDCG = 1.0."""
        assert _ndcg(["a", "x", "y"], ["a"]) == 1.0

    def test_empty_golden_list(self):
        """No golden docs → NDCG = 0.0 (no ideal ranking possible)."""
        assert _ndcg(["a", "b"], []) == 0.0


# ════════════════════════════════════════════════════════════════════════
# 4. compute_score — full pipeline with real path formats
# ════════════════════════════════════════════════════════════════════════


def _make_format_valid_solution(search_query="test query"):
    """Create a solution string that passes format_checker."""
    return (
        "<|im_start|>assistant\n"
        f"<think>I need to search for relevant documents.</think>\n"
        f"<search>{search_query}</search>\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>Now I have the results, let me finish.</think>\n"
        "<search_complete>true</search_complete>\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<answer>The answer is 42.</answer>\n"
        "<|im_end|>"
    )


VALID_SOLUTION = _make_format_valid_solution()


class TestComputeScoreSlideVQA:
    """compute_score with SlideVQA-style paths.

    This is the critical test: the current code normalizes ALL slides in a
    deck to the same value ("1024"), so NDCG is always 1.0 regardless of
    which slide was retrieved. After our fix, retrieving the WRONG slide
    must produce NDCG < 1.0.
    """

    def test_correct_slide_retrieved_scores_high(self):
        """Retrieving the GTI slide → NDCG = 1.0, score = 1.0."""
        extra_info = {
            "image_paths": ["slidevqa/accel-deck_95/slide_4_1024.jpg"],
            "reference_documents": ["slidevqa/accel-deck_95/slide_4_1024.jpg"],
        }
        result = compute_score(
            data_source="VDR_lns",
            solution_str=VALID_SOLUTION,
            ground_truth="The answer is 42.",
            extra_info=extra_info,
        )
        assert result["ndcg"] == 1.0
        assert result["score"] == pytest.approx(0.1 + 0.9 * 1.0)

    def test_wrong_slide_retrieved_scores_zero_ndcg(self):
        """Retrieving a NON-GTI slide → NDCG = 0.0.

        This test FAILS with the current code because all slides normalize
        to "1024", making any slide match the reference.
        """
        extra_info = {
            "image_paths": ["slidevqa/accel-deck_95/slide_1_1024.jpg"],
            "reference_documents": ["slidevqa/accel-deck_95/slide_4_1024.jpg"],
        }
        result = compute_score(
            data_source="VDR_lns",
            solution_str=VALID_SOLUTION,
            ground_truth="The answer is 42.",
            extra_info=extra_info,
        )
        assert result["ndcg"] == 0.0, (
            f"Expected NDCG=0.0 for wrong slide, got {result['ndcg']}. "
            f"Retrieved: slide_1, Reference: slide_4. "
            f"If NDCG=1.0, slides are not being distinguished."
        )
        assert result["score"] == pytest.approx(0.1 + 0.9 * 0.0)


class TestComputeScoreColQwen:
    """compute_score with ColQwen-source paths (docvqa, infovqa, etc.)."""

    def test_correct_image_retrieved(self):
        extra_info = {
            "image_paths": ["docvqa/hzym0020_14.png"],
            "reference_documents": ["docvqa/hzym0020_14.png"],
        }
        result = compute_score(
            data_source="VDR_lns",
            solution_str=VALID_SOLUTION,
            ground_truth="answer",
            extra_info=extra_info,
        )
        assert result["ndcg"] == 1.0

    def test_wrong_image_retrieved(self):
        extra_info = {
            "image_paths": ["docvqa/jybx0223_94.png"],
            "reference_documents": ["docvqa/hzym0020_14.png"],
        }
        result = compute_score(
            data_source="VDR_lns",
            solution_str=VALID_SOLUTION,
            ground_truth="answer",
            extra_info=extra_info,
        )
        assert result["ndcg"] == 0.0

    def test_cross_source_no_false_match(self):
        """An infovqa image must NOT match a docvqa image even if basenames
        happen to share a numeric suffix after '_'.
        """
        extra_info = {
            "image_paths": ["infovqa/some_14.jpeg"],
            "reference_documents": ["docvqa/other_14.png"],
        }
        result = compute_score(
            data_source="VDR_lns",
            solution_str=VALID_SOLUTION,
            ground_truth="answer",
            extra_info=extra_info,
        )
        assert result["ndcg"] == 0.0, (
            "Cross-source images with the same numeric suffix should NOT match. "
            "If they match, _normalize_doc_id is stripping too aggressively."
        )


class TestComputeScoreVDR:
    """compute_score with VDR hash-based image paths."""

    def test_correct_vdr_image(self):
        extra_info = {
            "image_paths": ["vdr/4f0d89780c8fb747ca03398424d76635f5f270da.png"],
            "reference_documents": ["vdr/4f0d89780c8fb747ca03398424d76635f5f270da.png"],
        }
        result = compute_score(
            data_source="VDR_lns",
            solution_str=VALID_SOLUTION,
            ground_truth="answer",
            extra_info=extra_info,
        )
        assert result["ndcg"] == 1.0

    def test_wrong_vdr_image(self):
        extra_info = {
            "image_paths": ["vdr/bd5bc38821f72b8dd88416988e37835277943bd9.png"],
            "reference_documents": ["vdr/4f0d89780c8fb747ca03398424d76635f5f270da.png"],
        }
        result = compute_score(
            data_source="VDR_lns",
            solution_str=VALID_SOLUTION,
            ground_truth="answer",
            extra_info=extra_info,
        )
        assert result["ndcg"] == 0.0


class TestComputeScoreMultipleRetrieved:
    """compute_score with multiple retrieved images (multi-turn search)."""

    def test_two_retrieved_one_correct(self):
        """Model searched twice: first miss, then hit. NDCG should reflect ordering."""
        extra_info = {
            "image_paths": [
                "docvqa/wrong_image.png",
                "docvqa/hzym0020_14.png",
            ],
            "reference_documents": ["docvqa/hzym0020_14.png"],
        }
        result = compute_score(
            data_source="VDR_lns",
            solution_str=VALID_SOLUTION,
            ground_truth="answer",
            extra_info=extra_info,
        )
        # GTI at position 2 of 2 → NDCG < 1.0 but > 0.0
        assert 0.0 < result["ndcg"] < 1.0

    def test_two_retrieved_first_correct(self):
        """First search hit the GTI → NDCG = 1.0."""
        extra_info = {
            "image_paths": [
                "docvqa/hzym0020_14.png",
                "docvqa/wrong_image.png",
            ],
            "reference_documents": ["docvqa/hzym0020_14.png"],
        }
        result = compute_score(
            data_source="VDR_lns",
            solution_str=VALID_SOLUTION,
            ground_truth="answer",
            extra_info=extra_info,
        )
        assert result["ndcg"] == 1.0


class TestComputeScoreFormatFailure:
    """compute_score when format check fails → score = 0.0 regardless of NDCG."""

    def test_bad_format_gets_zero(self):
        bad_solution = "This has no think or search tags at all."
        extra_info = {
            "image_paths": ["docvqa/hzym0020_14.png"],
            "reference_documents": ["docvqa/hzym0020_14.png"],
        }
        result = compute_score(
            data_source="VDR_lns",
            solution_str=bad_solution,
            ground_truth="answer",
            extra_info=extra_info,
        )
        assert result["score"] == 0.0
        assert result["format_score"] == 0.0


# ════════════════════════════════════════════════════════════════════════
# 5. _extract_retrieved and _extract_reference — path extraction
# ════════════════════════════════════════════════════════════════════════


class TestExtractRetrieved:
    """_extract_retrieved reads image_paths from extra_info."""

    def test_reads_image_paths_key(self):
        extra = {"image_paths": ["docvqa/abc.png", "infovqa/xyz.jpeg"]}
        result = _extract_retrieved(extra)
        assert result == ["docvqa/abc.png", "infovqa/xyz.jpeg"]

    def test_empty_extra_info(self):
        assert _extract_retrieved({}) == []


class TestExtractReference:
    """_extract_reference reads reference_documents from extra_info."""

    def test_reads_reference_documents(self):
        extra = {"reference_documents": ["docvqa/hzym0020_14.png"]}
        result = _extract_reference(extra)
        assert result == ["docvqa/hzym0020_14.png"]

    def test_multiple_references(self):
        extra = {"reference_documents": ["infovqa/a.jpeg", "infovqa/b.jpeg"]}
        result = _extract_reference(extra)
        assert len(result) == 2
