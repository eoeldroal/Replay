"""Tests for deck index build — maps row_index → embedding shard locations.

The deck index maps each of the 80,292 parquet rows to the exact
(dataset_idx, shard_idx, local_idx) for all 20 document images.

Key challenge: VDR_ibm embedding doc_ids lack file extensions,
while the parquet uses extensions (e.g. 'vdr/HASH.png' vs 'vdr/HASH').
"""

import os

import pytest

EMBEDDING_BASE = (
    "/home/work/DDAI_revised/verl/data/Visual_Document_Rag/"
    "VDR_processed_filtered_2/embeddings/colqwen2"
)
VDR_FINAL_DIR = (
    "/home/work/DDAI_revised/verl/data/Visual_Document_Rag/VDR_final"
)
DECK_INDEX_PATH = os.path.join(VDR_FINAL_DIR, "deck_index.pt")


# ════════════════════════════════════════════════════════════════════════
# 1. strip_ext() normalization
# ════════════════════════════════════════════════════════════════════════


class TestStripExt:
    """strip_ext() must remove image extensions, leave others unchanged."""

    @pytest.fixture(autouse=True)
    def _import(self):
        import importlib
        import sys

        sys.path.insert(0, VDR_FINAL_DIR)
        mod = importlib.import_module("build_deck_index")
        self.strip_ext = mod.strip_ext

    def test_removes_png(self):
        assert self.strip_ext("vdr/abc123.png") == "vdr/abc123"

    def test_removes_jpg(self):
        assert self.strip_ext("docvqa/test.jpg") == "docvqa/test"

    def test_removes_jpeg(self):
        assert self.strip_ext("infovqa/38032.jpeg") == "infovqa/38032"

    def test_no_extension_unchanged(self):
        assert self.strip_ext("vdr/abc123") == "vdr/abc123"

    def test_non_image_extension_unchanged(self):
        assert self.strip_ext("data/file.txt") == "data/file.txt"

    def test_case_insensitive(self):
        assert self.strip_ext("vdr/abc.PNG") == "vdr/abc"
        assert self.strip_ext("vdr/abc.JPG") == "vdr/abc"
        assert self.strip_ext("vdr/abc.JPEG") == "vdr/abc"

    def test_preserves_path_structure(self):
        assert (
            self.strip_ext("slidevqa/deck_95/slide_1_1024.jpg")
            == "slidevqa/deck_95/slide_1_1024"
        )


# ════════════════════════════════════════════════════════════════════════
# 2. Embedding lookup table construction
# ════════════════════════════════════════════════════════════════════════


class TestLookupTableConstruction:
    """Verify that the lookup table maps normalized doc_ids to shard locations."""

    @pytest.fixture
    def sample_shard_data(self):
        """Minimal fake shard data for unit testing."""
        return {
            "doc_ids": [
                "vdr/abc123",
                "vdr/def456",
                "docvqa/test.png",
            ],
            "dataset": "VDR_ibm",
        }

    def test_build_lookup_from_single_shard(self, sample_shard_data):
        import importlib
        import sys

        sys.path.insert(0, VDR_FINAL_DIR)
        mod = importlib.import_module("build_deck_index")

        lookup = {}
        dataset_idx = 1
        shard_idx = 0
        for local_idx, doc_id in enumerate(sample_shard_data["doc_ids"]):
            norm = mod.strip_ext(doc_id)
            lookup[norm] = (dataset_idx, shard_idx, local_idx)

        assert lookup["vdr/abc123"] == (1, 0, 0)
        assert lookup["vdr/def456"] == (1, 0, 1)
        assert lookup["docvqa/test"] == (1, 0, 2)


# ════════════════════════════════════════════════════════════════════════
# 3. VDR_ibm normalization matching
# ════════════════════════════════════════════════════════════════════════


class TestVDRNormalizationMatching:
    """VDR_ibm parquet doc_ids have .png, embedding doc_ids don't."""

    def test_vdr_parquet_to_embedding_match(self):
        import importlib
        import sys

        sys.path.insert(0, VDR_FINAL_DIR)
        mod = importlib.import_module("build_deck_index")

        # Simulate: parquet has 'vdr/HASH.png', embedding has 'vdr/HASH'
        parquet_doc_id = "vdr/4f0d89780c8fb747ca03398424d76635f5f270da.png"
        embedding_doc_id = "vdr/4f0d89780c8fb747ca03398424d76635f5f270da"

        assert mod.strip_ext(parquet_doc_id) == mod.strip_ext(embedding_doc_id)

    def test_non_vdr_no_mismatch(self):
        """OpenDocVQA/SlideVQA: parquet and embedding doc_ids already match."""
        import importlib
        import sys

        sys.path.insert(0, VDR_FINAL_DIR)
        mod = importlib.import_module("build_deck_index")

        # OpenDocVQA embeds include extension
        emb_id = "visualmrc/okfn.org__about.png"
        par_id = "visualmrc/okfn.org__about.png"
        assert mod.strip_ext(emb_id) == mod.strip_ext(par_id)

        # SlideVQA embeds include extension
        emb_id2 = "slidevqa/deck_95/slide_1_1024.jpg"
        par_id2 = "slidevqa/deck_95/slide_1_1024.jpg"
        assert mod.strip_ext(emb_id2) == mod.strip_ext(par_id2)


# ════════════════════════════════════════════════════════════════════════
# 4. Full deck_index.pt validation (integration tests)
# ════════════════════════════════════════════════════════════════════════


@pytest.fixture
def deck_index():
    """Load the built deck_index.pt. Skip if not yet generated."""
    import torch

    if not os.path.exists(DECK_INDEX_PATH):
        pytest.skip(f"deck_index.pt not found: {DECK_INDEX_PATH}")
    return torch.load(DECK_INDEX_PATH, map_location="cpu", weights_only=False)


class TestDeckIndexStructure:
    """deck_index.pt must have the correct structure and fields."""

    def test_has_required_keys(self, deck_index):
        required = {
            "deck_locations",
            "datasets",
            "shard_paths",
            "doc_ids_original",
            "doc_ids_normalized",
        }
        assert required.issubset(set(deck_index.keys())), (
            f"Missing keys: {required - set(deck_index.keys())}"
        )

    def test_datasets_list(self, deck_index):
        assert deck_index["datasets"] == ["OpenDocVQA", "VDR_ibm", "SlideVQA"]

    def test_shard_paths_keys(self, deck_index):
        assert set(deck_index["shard_paths"].keys()) == {
            "OpenDocVQA", "VDR_ibm", "SlideVQA"
        }

    def test_shard_paths_counts(self, deck_index):
        assert len(deck_index["shard_paths"]["OpenDocVQA"]) == 473
        assert len(deck_index["shard_paths"]["VDR_ibm"]) == 200
        assert len(deck_index["shard_paths"]["SlideVQA"]) == 204


class TestDeckIndexCoverage:
    """Every row must be fully mapped — no missing images."""

    def test_deck_locations_length(self, deck_index):
        assert len(deck_index["deck_locations"]) == 80292

    def test_each_row_has_20_locations(self, deck_index):
        bad = []
        for i, locs in enumerate(deck_index["deck_locations"]):
            if len(locs) != 20:
                bad.append((i, len(locs)))
                if len(bad) >= 5:
                    break
        assert len(bad) == 0, f"Rows with != 20 locations: {bad}"

    def test_location_tuple_format(self, deck_index):
        """Each location must be (dataset_idx, shard_idx, local_idx)."""
        for loc in deck_index["deck_locations"][0]:
            assert len(loc) == 3
            ds_idx, shard_idx, local_idx = loc
            assert 0 <= ds_idx < 3
            assert isinstance(shard_idx, int) and shard_idx >= 0
            assert isinstance(local_idx, int) and local_idx >= 0

    def test_dataset_idx_in_range(self, deck_index):
        """All dataset_idx values must be valid."""
        num_datasets = len(deck_index["datasets"])
        for row_locs in deck_index["deck_locations"][:1000]:
            for ds_idx, _, _ in row_locs:
                assert 0 <= ds_idx < num_datasets

    def test_shard_idx_in_range(self, deck_index):
        """All shard_idx must be valid for the corresponding dataset."""
        ds_names = deck_index["datasets"]
        shard_paths = deck_index["shard_paths"]
        for row_locs in deck_index["deck_locations"][:1000]:
            for ds_idx, shard_idx, _ in row_locs:
                ds_name = ds_names[ds_idx]
                assert shard_idx < len(shard_paths[ds_name]), (
                    f"shard_idx {shard_idx} out of range for {ds_name} "
                    f"(max {len(shard_paths[ds_name]) - 1})"
                )


class TestDocIdsPreservation:
    """Both original and normalized doc_ids must be preserved."""

    def test_doc_ids_original_length(self, deck_index):
        assert len(deck_index["doc_ids_original"]) == 80292

    def test_doc_ids_normalized_length(self, deck_index):
        assert len(deck_index["doc_ids_normalized"]) == 80292

    def test_each_row_has_20_doc_ids(self, deck_index):
        for i in range(min(100, len(deck_index["doc_ids_original"]))):
            assert len(deck_index["doc_ids_original"][i]) == 20
            assert len(deck_index["doc_ids_normalized"][i]) == 20

    def test_vdr_original_has_extension(self, deck_index):
        """VDR doc_ids in original should retain .png extension."""
        import pandas as pd

        parquet_path = os.path.join(VDR_FINAL_DIR, "train.parquet")
        if not os.path.exists(parquet_path):
            pytest.skip("Parquet not found")
        df = pd.read_parquet(parquet_path, columns=["document_images"])

        # Find a VDR row
        for idx, row in df.iterrows():
            imgs = row["document_images"]
            if hasattr(imgs, "tolist"):
                imgs = imgs.tolist()
            if any(img.startswith("vdr/") and img.endswith(".png") for img in imgs):
                orig = deck_index["doc_ids_original"][idx]
                vdr_originals = [d for d in orig if d.startswith("vdr/")]
                for d in vdr_originals:
                    assert d.endswith(".png"), f"VDR original missing .png: {d}"
                break

    def test_vdr_normalized_no_extension(self, deck_index):
        """VDR doc_ids in normalized should NOT have .png extension."""
        import pandas as pd

        parquet_path = os.path.join(VDR_FINAL_DIR, "train.parquet")
        if not os.path.exists(parquet_path):
            pytest.skip("Parquet not found")
        df = pd.read_parquet(parquet_path, columns=["document_images"])

        for idx, row in df.iterrows():
            imgs = row["document_images"]
            if hasattr(imgs, "tolist"):
                imgs = imgs.tolist()
            if any(img.startswith("vdr/") and img.endswith(".png") for img in imgs):
                norm = deck_index["doc_ids_normalized"][idx]
                vdr_norms = [d for d in norm if d.startswith("vdr/")]
                for d in vdr_norms:
                    assert not d.endswith(".png"), f"VDR normalized has .png: {d}"
                break

    def test_original_matches_parquet(self, deck_index):
        """doc_ids_original must match the parquet document_images exactly."""
        import pandas as pd

        parquet_path = os.path.join(VDR_FINAL_DIR, "train.parquet")
        if not os.path.exists(parquet_path):
            pytest.skip("Parquet not found")
        df = pd.read_parquet(parquet_path, columns=["document_images"])

        # Spot-check first 50 rows
        for idx in range(min(50, len(df))):
            parquet_imgs = df.iloc[idx]["document_images"]
            if hasattr(parquet_imgs, "tolist"):
                parquet_imgs = parquet_imgs.tolist()
            assert deck_index["doc_ids_original"][idx] == parquet_imgs, (
                f"Row {idx} mismatch"
            )


# ════════════════════════════════════════════════════════════════════════
# 5. Cross-validation: deck_index vs actual embedding shards
# ════════════════════════════════════════════════════════════════════════


class TestCrossValidation:
    """Spot-check that deck_index locations actually point to correct shards."""

    def test_spot_check_locations(self, deck_index):
        """Load a few shards and verify doc_id at the pointed location."""
        import torch

        if not os.path.exists(EMBEDDING_BASE):
            pytest.skip("Embedding shards not available")

        import importlib
        import sys

        sys.path.insert(0, VDR_FINAL_DIR)
        mod = importlib.import_module("build_deck_index")

        ds_names = deck_index["datasets"]
        shard_paths = deck_index["shard_paths"]

        # Check rows 0, 1000, 50000, 80291
        check_rows = [0, 1000, 50000, 80291]
        for row_idx in check_rows:
            if row_idx >= len(deck_index["deck_locations"]):
                continue
            locs = deck_index["deck_locations"][row_idx]
            norms = deck_index["doc_ids_normalized"][row_idx]

            # Check first and last image
            for img_pos in [0, 19]:
                ds_idx, shard_idx, local_idx = locs[img_pos]
                ds_name = ds_names[ds_idx]
                shard_file = os.path.join(
                    EMBEDDING_BASE, ds_name, shard_paths[ds_name][shard_idx]
                )
                shard = torch.load(shard_file, map_location="cpu", weights_only=False)
                actual_doc_id = shard["doc_ids"][local_idx]
                expected_norm = norms[img_pos]
                assert mod.strip_ext(actual_doc_id) == expected_norm, (
                    f"Row {row_idx}, pos {img_pos}: "
                    f"expected {expected_norm}, got {mod.strip_ext(actual_doc_id)}"
                )
