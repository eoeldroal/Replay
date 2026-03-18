# Copyright 2026

import numpy as np
import pytest
import torch

from verl import DataProto


def _make_grouped_data(group_lengths: list[int]) -> DataProto:
    total_rows = sum(group_lengths)
    values = torch.arange(total_rows, dtype=torch.float32).reshape(total_rows, 1)
    group_ids: list[str] = []
    for idx, group_length in enumerate(group_lengths):
        group_ids.extend([f"g{idx}"] * group_length)
    return DataProto.from_dict(
        tensors={"old_log_probs": values},
        non_tensors={"group_id": np.array(group_ids, dtype=object)},
        meta_info={"m2_group_lengths": list(group_lengths), "marker": "m2-fastpath"},
    )


def test_chunk_preserves_full_m2_groups_and_shards_group_lengths():
    data = _make_grouped_data([8, 8, 5, 7, 8])

    chunks = data.chunk(4)

    assert [chunk.meta_info["m2_group_lengths"] for chunk in chunks] == [[8, 8], [5], [7], [8]]
    assert [len(chunk) for chunk in chunks] == [16, 5, 7, 8]

    for chunk in chunks:
        assert sum(chunk.meta_info["m2_group_lengths"]) == len(chunk)
        observed_lengths = [
            int((chunk.non_tensor_batch["group_id"] == group_id).sum())
            for group_id in dict.fromkeys(chunk.non_tensor_batch["group_id"].tolist())
        ]
        assert observed_lengths == chunk.meta_info["m2_group_lengths"]


def test_chunk_with_m2_group_lengths_allows_more_chunks_than_groups_and_keeps_meta_isolated():
    data = _make_grouped_data([3, 2])

    chunks = data.chunk(4)

    assert [chunk.meta_info["m2_group_lengths"] for chunk in chunks] == [[3], [2], [], []]
    assert [len(chunk) for chunk in chunks] == [3, 2, 0, 0]

    chunks[0].meta_info.pop("m2_group_lengths")
    assert chunks[1].meta_info["m2_group_lengths"] == [2]
    assert chunks[2].meta_info["m2_group_lengths"] == []


@pytest.mark.parametrize(
    ("group_lengths", "chunks"),
    [
        ([8], 1),
        ([8], 4),
        ([2, 3, 5], 2),
        ([9, 1, 8, 2, 7], 3),
        ([4, 4, 4, 4], 6),
        ([1, 2, 3, 4, 5, 6], 4),
    ],
)
def test_chunk_with_m2_group_lengths_never_splits_group_boundaries(group_lengths: list[int], chunks: int):
    data = _make_grouped_data(group_lengths)

    chunked = data.chunk(chunks)

    flattened_group_lengths = [length for chunk in chunked for length in chunk.meta_info["m2_group_lengths"]]
    assert flattened_group_lengths == group_lengths
    assert sum(len(chunk) for chunk in chunked) == len(data)

    observed_group_order: list[str] = []
    for chunk in chunked:
        chunk_group_ids = chunk.non_tensor_batch["group_id"].tolist()
        if not chunk_group_ids:
            assert chunk.meta_info["m2_group_lengths"] == []
            continue
        contiguous_ids = list(dict.fromkeys(chunk_group_ids))
        observed_group_order.extend(contiguous_ids)
        for group_id in contiguous_ids:
            positions = [idx for idx, value in enumerate(chunk_group_ids) if value == group_id]
            assert positions == list(range(positions[0], positions[-1] + 1))

    assert observed_group_order == [f"g{idx}" for idx in range(len(group_lengths))]


def test_chunk_with_m2_group_lengths_rejects_nonpositive_lengths():
    data = DataProto.from_dict(
        tensors={"old_log_probs": torch.arange(5, dtype=torch.float32).reshape(5, 1)},
        meta_info={"m2_group_lengths": [3, 0, 2]},
    )

    with pytest.raises(AssertionError, match="positive lengths"):
        data.chunk(2)


def test_chunk_with_m2_group_lengths_rejects_sum_mismatch():
    data = DataProto.from_dict(
        tensors={"old_log_probs": torch.arange(5, dtype=torch.float32).reshape(5, 1)},
        meta_info={"m2_group_lengths": [2, 2]},
    )

    with pytest.raises(AssertionError, match="must sum to the DataProto batch size"):
        data.chunk(2)


def test_chunk_and_concat_round_trip_restores_m2_group_lengths():
    data = _make_grouped_data([8, 8, 5, 7, 8])

    reconstructed = DataProto.concat(data.chunk(4))

    assert torch.equal(reconstructed.batch["old_log_probs"], data.batch["old_log_probs"])
    assert np.array_equal(reconstructed.non_tensor_batch["group_id"], data.non_tensor_batch["group_id"])
    assert reconstructed.meta_info["m2_group_lengths"] == [8, 8, 5, 7, 8]
    assert reconstructed.meta_info["marker"] == "m2-fastpath"


def test_chunk_without_m2_group_lengths_copies_meta_info_per_chunk():
    data = DataProto.from_dict(
        tensors={"old_log_probs": torch.arange(6, dtype=torch.float32).reshape(6, 1)},
        meta_info={"flag": True, "note": "general-path"},
    )

    chunks = data.chunk(3)

    chunks[0].meta_info.pop("flag")
    assert chunks[1].meta_info["flag"] is True
    assert chunks[2].meta_info["note"] == "general-path"
