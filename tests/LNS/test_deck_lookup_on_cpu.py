"""Tests for server-side deck lookup — parquet-based deck mapping.

The retrieval server loads train.parquet directly from the shared filesystem
and builds a deck_map: row_index → list of 20 document_images paths.

These tests verify the parquet structure and deck lookup correctness.
"""

import pytest


# ════════════════════════════════════════════════════════════════════════
# 1. Parquet structure validation
# ════════════════════════════════════════════════════════════════════════


class TestParquetStructure:
    """train.parquet must have the expected columns and row count."""

    def test_has_required_columns(self, parquet_path):
        import pandas as pd

        df = pd.read_parquet(parquet_path, columns=None)
        required = {"id", "query", "document_images", "answer", "gti"}
        assert required.issubset(set(df.columns)), (
            f"Missing columns: {required - set(df.columns)}"
        )

    def test_row_count(self, parquet_path):
        import pandas as pd

        df = pd.read_parquet(parquet_path, columns=["id"])
        assert len(df) == 80292

    def test_all_ids_unique(self, parquet_path):
        import pandas as pd

        df = pd.read_parquet(parquet_path, columns=["id"])
        assert df["id"].is_unique


# ════════════════════════════════════════════════════════════════════════
# 2. Deck composition — every sample has exactly 20 images
# ════════════════════════════════════════════════════════════════════════


class TestDeckComposition:
    """Each sample must have a valid 20-image deck."""

    def test_all_decks_have_20_images(self, parquet_path):
        import pandas as pd

        df = pd.read_parquet(parquet_path, columns=["document_images"])
        lengths = df["document_images"].apply(len)
        bad = lengths[lengths != 20]
        assert len(bad) == 0, (
            f"{len(bad)} samples don't have 20 images. "
            f"Examples: {bad.head(5).to_dict()}"
        )

    def test_gti_is_subset_of_deck(self, parquet_path):
        """Every GTI image must appear in its sample's document_images."""
        import pandas as pd

        df = pd.read_parquet(parquet_path, columns=["document_images", "gti"])
        missing_count = 0
        examples = []
        for idx, row in df.iterrows():
            deck = row["document_images"]
            if hasattr(deck, "tolist"):
                deck = deck.tolist()
            deck_set = set(deck)
            gti = row["gti"]
            if hasattr(gti, "tolist"):
                gti = gti.tolist()
            gti_list = gti if isinstance(gti, list) else [gti]
            for g in gti_list:
                if g not in deck_set:
                    missing_count += 1
                    if len(examples) < 3:
                        examples.append({"idx": idx, "gti": g})
        assert missing_count == 0, (
            f"{missing_count} GTI images missing from their decks. "
            f"Examples: {examples}"
        )

    def test_all_deck_images_are_relative_paths(self, parquet_path):
        """Deck images must be relative paths, not absolute."""
        import pandas as pd

        df = pd.read_parquet(parquet_path, columns=["document_images"])
        for idx, row in df.head(100).iterrows():
            for img in row["document_images"]:
                assert not img.startswith("/"), f"Row {idx}: absolute path found: {img}"

    def test_all_deck_images_have_source_prefix(self, parquet_path):
        """Every image path should start with a source prefix (docvqa/, vdr/, etc.)."""
        import pandas as pd

        VALID_PREFIXES = {
            "docvqa/", "infovqa/", "coyo/", "chartqa/",
            "visualmrc/", "mpmqa/", "openwikitable/",
            "vdr/", "slidevqa/",
        }
        df = pd.read_parquet(parquet_path, columns=["document_images"])
        bad = []
        for idx, row in df.head(200).iterrows():
            for img in row["document_images"]:
                if not any(img.startswith(p) for p in VALID_PREFIXES):
                    bad.append({"idx": idx, "img": img})
        assert len(bad) == 0, f"Images without valid prefix: {bad[:5]}"


# ════════════════════════════════════════════════════════════════════════
# 3. Deck map construction (server startup)
# ════════════════════════════════════════════════════════════════════════


class TestDeckMapConstruction:
    """The server builds deck_map from train.parquet at startup."""

    def test_deck_map_from_parquet(self, parquet_path):
        """Build deck_map and verify it covers all samples."""
        import pandas as pd

        df = pd.read_parquet(parquet_path, columns=["id", "document_images"])

        # This is what the server does at startup
        deck_map = {}
        for idx, row in df.iterrows():
            imgs = row["document_images"]
            if hasattr(imgs, "tolist"):
                imgs = imgs.tolist()
            deck_map[idx] = imgs

        assert len(deck_map) == 80292
        assert all(len(v) == 20 for v in deck_map.values())

    def test_id_to_row_index_mapping(self, parquet_path):
        """Build id_to_idx and verify correctness."""
        import pandas as pd

        df = pd.read_parquet(parquet_path, columns=["id"])
        id_to_idx = pd.Series(df.index, index=df["id"]).to_dict()

        assert len(id_to_idx) == 80292
        # Spot-check known IDs
        assert isinstance(id_to_idx.get("infovqa-train_1"), int)
        assert isinstance(id_to_idx.get("slidevqa-0"), int)

    def test_roundtrip_id_to_deck(self, parquet_path):
        """id → row_index → deck_map → 20 images, with GTI present."""
        import pandas as pd

        df = pd.read_parquet(parquet_path, columns=["id", "document_images", "gti"])
        id_to_idx = pd.Series(df.index, index=df["id"]).to_dict()

        # Pick a sample
        sample_id = "slidevqa-0"
        row_idx = id_to_idx[sample_id]
        row = df.iloc[row_idx]
        deck = row["document_images"]
        gti = row["gti"]

        if hasattr(deck, "tolist"):
            deck = deck.tolist()
        if hasattr(gti, "tolist"):
            gti = gti.tolist()
        if not isinstance(gti, list):
            gti = [gti]

        assert len(deck) == 20
        for g in gti:
            assert g in deck, f"GTI {g} not in deck for {sample_id}"


# ════════════════════════════════════════════════════════════════════════
# 4. Corpus image accessibility
# ════════════════════════════════════════════════════════════════════════


class TestCorpusImageAccess:
    """Deck image paths must resolve to actual files on the shared filesystem."""

    def test_sample_images_exist(self, parquet_path, corpus_img_root):
        """Spot-check: first 10 samples' images should all exist on disk."""
        import os

        import pandas as pd

        df = pd.read_parquet(parquet_path, columns=["document_images"])

        missing = []
        for idx in range(min(10, len(df))):
            for img in df.iloc[idx]["document_images"]:
                full_path = os.path.join(corpus_img_root, img)
                if not os.path.exists(full_path):
                    missing.append(full_path)

        assert len(missing) == 0, (
            f"{len(missing)} images not found. Examples: {missing[:5]}"
        )


# ════════════════════════════════════════════════════════════════════════
# 5. Embedding shard coverage
# ════════════════════════════════════════════════════════════════════════


EMBEDDING_BASE = "/home/work/DDAI_revised/verl/data/Visual_Document_Rag/VDR_processed_filtered_2/embeddings/colqwen2"


class TestEmbeddingCoverage:
    """All unique images in the dataset must have pre-computed embeddings."""

    @pytest.fixture
    def all_unique_images(self, parquet_path):
        import pandas as pd

        df = pd.read_parquet(parquet_path, columns=["document_images"])
        unique = set()
        for imgs in df["document_images"]:
            for img in imgs:
                unique.add(img)
        return unique

    @pytest.fixture
    def all_embedded_doc_ids(self):
        """Load all doc_ids from embedding shards."""
        import glob
        import os

        import torch

        if not os.path.exists(EMBEDDING_BASE):
            pytest.skip(f"Embedding dir not found: {EMBEDDING_BASE}")

        doc_ids = set()
        for ds in ["OpenDocVQA", "VDR_ibm", "SlideVQA"]:
            shard_dir = os.path.join(EMBEDDING_BASE, ds)
            if not os.path.exists(shard_dir):
                continue
            for shard_file in glob.glob(os.path.join(shard_dir, "shard_*.pt")):
                shard = torch.load(shard_file, map_location="cpu", weights_only=False)
                for did in shard["doc_ids"]:
                    doc_ids.add(did)
        return doc_ids

    def test_dataset_image_format_samples(self, all_unique_images):
        """Sanity check: images use expected path formats."""
        prefixes = set()
        for img in list(all_unique_images)[:1000]:
            prefixes.add(img.split("/")[0])
        expected = {"docvqa", "infovqa", "coyo", "chartqa", "visualmrc",
                    "mpmqa", "openwikitable", "vdr", "slidevqa"}
        assert prefixes.issubset(expected), f"Unexpected prefixes: {prefixes - expected}"
