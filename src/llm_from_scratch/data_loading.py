"""Sliding-window data for next-token prediction."""

from typing import Protocol

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


class GPTDatasetV1(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Fixed-length windows paired with one-token-shifted targets."""

    def __init__(
        self, txt: str, tokenizer: Tokenizer, max_length: int, stride: int
    ) -> None:
        if max_length <= 0 or stride <= 0:
            raise ValueError("max_length and stride must be positive")

        token_ids = tokenizer.encode(txt)
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []
        for start in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(
                torch.tensor(token_ids[start : start + max_length], dtype=torch.long)
            )
            self.target_ids.append(
                torch.tensor(
                    token_ids[start + 1 : start + max_length + 1], dtype=torch.long
                )
            )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    tokenizer: Tokenizer | None = None,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Batch tokenized text into next-token prediction examples.

    If tokenizer is omitted, use GPT-2's byte pair encoding. Incomplete windows
    are discarded by the dataset. drop_last independently controls whether to
    discard the final incomplete batch of full windows.
    """
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
