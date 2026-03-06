"""Shared fixtures for LNS tests."""

import os

import pytest

# Path to the actual VDR_final train.parquet (shared filesystem)
VDR_FINAL_PARQUET = "/home/work/DDAI_revised/verl/data/Visual_Document_Rag/VDR_final/train.parquet"
CORPUS_IMG_ROOT = "/home/work/DDAI_revised/verl/data/Visual_Document_Rag/VDR_final/corpus/img"


@pytest.fixture
def parquet_path():
    """Path to VDR_final train.parquet. Skips if not available."""
    if not os.path.exists(VDR_FINAL_PARQUET):
        pytest.skip(f"Parquet not found: {VDR_FINAL_PARQUET}")
    return VDR_FINAL_PARQUET


@pytest.fixture
def corpus_img_root():
    """Path to corpus/img root. Skips if not available."""
    if not os.path.exists(CORPUS_IMG_ROOT):
        pytest.skip(f"Corpus not found: {CORPUS_IMG_ROOT}")
    return CORPUS_IMG_ROOT


# ── Representative sample data from train.parquet ──
# These are real examples used across tests to avoid hard-coding everywhere.

SAMPLE_COLQWEN_ROW = {
    "id": "infovqa-train_1",
    "query": "What is the website mentioned?",
    "document_images": [
        "infovqa/38032.jpeg",
        "docvqa/jybx0223_94.png",
        "docvqa/jybx0223_58.png",
        "coyo/sample1.jpg",
        "chartqa/chart1.png",
        "visualmrc/page1.png",
        "mpmqa/doc1.png",
        "openwikitable/table1.png",
        "infovqa/38033.jpeg",
        "infovqa/38034.jpeg",
        "docvqa/abc_1.png",
        "docvqa/abc_2.png",
        "docvqa/abc_3.png",
        "docvqa/abc_4.png",
        "docvqa/abc_5.png",
        "docvqa/abc_6.png",
        "docvqa/abc_7.png",
        "docvqa/abc_8.png",
        "docvqa/abc_9.png",
        "docvqa/abc_10.png",
    ],
    "gti": ["infovqa/38032.jpeg"],
}

SAMPLE_SLIDEVQA_ROW = {
    "id": "slidevqa-0",
    "query": "What is the total revenue?",
    "document_images": [
        "slidevqa/accel-deck_95/slide_1_1024.jpg",
        "slidevqa/accel-deck_95/slide_2_1024.jpg",
        "slidevqa/accel-deck_95/slide_3_1024.jpg",
        "slidevqa/accel-deck_95/slide_4_1024.jpg",
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
    "gti": ["slidevqa/accel-deck_95/slide_4_1024.jpg"],
}

SAMPLE_VDR_ROW = {
    "id": "vdr-train_100",
    "query": "What does the table show?",
    "document_images": [
        "vdr/4f0d89780c8fb747ca03398424d76635f5f270da.png",
        "vdr/bd5bc38821f72b8dd88416988e37835277943bd9.png",
        "vdr/cd3759171531dcac639ac9495789a56c5ccd7913.png",
        "docvqa/hzym0020_14.png",
        "infovqa/12345.jpeg",
        "vdr/aaa111.png",
        "vdr/bbb222.png",
        "vdr/ccc333.png",
        "vdr/ddd444.png",
        "vdr/eee555.png",
        "vdr/fff666.png",
        "vdr/ggg777.png",
        "vdr/hhh888.png",
        "vdr/iii999.png",
        "vdr/jjj000.png",
        "vdr/kkk111.png",
        "vdr/lll222.png",
        "vdr/mmm333.png",
        "vdr/nnn444.png",
        "vdr/ooo555.png",
    ],
    "gti": ["vdr/4f0d89780c8fb747ca03398424d76635f5f270da.png"],
}


@pytest.fixture
def colqwen_row():
    return SAMPLE_COLQWEN_ROW


@pytest.fixture
def slidevqa_row():
    return SAMPLE_SLIDEVQA_ROW


@pytest.fixture
def vdr_row():
    return SAMPLE_VDR_ROW
