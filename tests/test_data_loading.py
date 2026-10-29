"""Checks for sliding windows, stride behavior, and batching."""

import pytest
import torch

from llm_from_scratch.data_loading import GPTDatasetV1, create_dataloader_v1

TOKEN_IDS = " ".join(str(index) for index in range(21))


class IntegerTokenizer:
    """Use numbered tokens to make dataset boundaries independent of BPE."""

    def encode(self, text):
        return [int(token) for token in text.split()]


@pytest.mark.parametrize(("max_length", "stride"), [(4, 1), (4, 4), (2, 2), (8, 2)])
def test_sliding_windows_across_context_settings(max_length, stride):
    ids = IntegerTokenizer().encode(TOKEN_IDS)
    dataset = GPTDatasetV1(TOKEN_IDS, IntegerTokenizer(), max_length, stride)
    starts = list(range(0, len(ids) - max_length, stride))
    assert len(dataset) == len(starts)
    for index in (0, 1, len(dataset) - 1):
        inputs, targets = dataset[index]
        start = starts[index]
        assert inputs.dtype == targets.dtype == torch.long
        assert inputs.shape == targets.shape == (max_length,)
        assert inputs.tolist() == ids[start : start + max_length]
        assert targets.tolist() == ids[start + 1 : start + max_length + 1]
        assert torch.equal(inputs[1:], targets[:-1])


@pytest.mark.parametrize(
    ("text", "length", "stride", "count"),
    [
        ("", 4, 1, 0),
        ("0 1 2", 4, 1, 0),
        ("0 1 2 3", 4, 1, 0),
        ("0 1 2 3 4", 4, 1, 1),
        ("0 1 2 3 4 5 6 7 8", 4, 4, 2),
        ("0 1 2 3 4 5 6 7", 4, 4, 1),
        ("0 1 2 3 4 5 6 7 8", 2, 5, 2),
    ],
)
def test_incomplete_windows_are_discarded(text, length, stride, count):
    dataset = GPTDatasetV1(text, IntegerTokenizer(), length, stride)
    assert len(dataset) == count
    if count:
        assert dataset[-1][1][-1].item() == (count - 1) * stride + length


@pytest.mark.parametrize(("length", "stride"), [(0, 1), (-1, 1), (4, 0), (4, -1)])
def test_invalid_window_settings(length, stride):
    with pytest.raises(ValueError, match="must be positive"):
        GPTDatasetV1("0 1 2 3", IntegerTokenizer(), length, stride)


def test_dataloader_uses_supplied_tokenizer():
    loader = create_dataloader_v1(
        TOKEN_IDS,
        batch_size=2,
        max_length=3,
        stride=3,
        shuffle=False,
        tokenizer=IntegerTokenizer(),
    )
    inputs, targets = next(iter(loader))
    assert inputs.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert targets.tolist() == [[1, 2, 3], [4, 5, 6]]


@pytest.mark.parametrize(("drop_last", "sizes"), [(True, [2, 2]), (False, [2, 2, 1])])
def test_drop_last_controls_partial_batches(drop_last, sizes):
    loader = create_dataloader_v1(
        "0 1 2 3 4 5",
        batch_size=2,
        max_length=1,
        stride=1,
        shuffle=False,
        drop_last=drop_last,
        tokenizer=IntegerTokenizer(),
    )
    assert [inputs.shape[0] for inputs, _ in loader] == sizes


def test_empty_dataset_can_be_loaded_without_shuffling():
    loader = create_dataloader_v1(
        "", max_length=4, shuffle=False, tokenizer=IntegerTokenizer()
    )
    assert list(loader) == []


def test_shuffle_is_reproducible_with_torch_seed():
    def collect(shuffle):
        loader = create_dataloader_v1(
            TOKEN_IDS,
            batch_size=4,
            max_length=2,
            stride=2,
            shuffle=shuffle,
            tokenizer=IntegerTokenizer(),
        )
        return torch.cat([inputs for inputs, _ in loader])

    torch.manual_seed(123)
    first = collect(True)
    torch.manual_seed(123)
    assert torch.equal(first, collect(True))
    assert not torch.equal(first, collect(False))
