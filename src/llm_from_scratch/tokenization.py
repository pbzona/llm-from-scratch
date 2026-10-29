"""Word-level tokenization and vocabulary construction."""

import re
from collections.abc import Iterable

END_OF_TEXT = "<|endoftext|>"
UNKNOWN_TOKEN = "<|unk|>"
_TOKEN_PATTERN = re.compile(r"""([,.:;?_!"()']|--|\s)""")
_PUNCTUATION_SPACING = re.compile(r"""\s+([,.:;?!"()'])""")


def tokenize(text: str) -> list[str]:
    """Split words, punctuation, and double dashes; discard whitespace."""
    return [part.strip() for part in _TOKEN_PATTERN.split(text) if part.strip()]


def build_vocabulary(
    text: str, *, include_special_tokens: bool = False
) -> dict[str, int]:
    """Assign sorted tokens consecutive IDs, optionally adding special tokens."""
    tokens = sorted(set(tokenize(text)))
    if include_special_tokens:
        for token in (END_OF_TEXT, UNKNOWN_TOKEN):
            if token not in tokens:
                tokens.append(token)
    return {token: index for index, token in enumerate(tokens)}


class SimpleTokenizerV1:
    """Word-level tokenizer; unknown words raise KeyError.

    Decoding normalizes whitespace and only removes spaces before punctuation,
    as in the book. It cannot recover the original layout or apostrophe spacing.
    """

    def __init__(self, vocab: dict[str, int]) -> None:
        self.str_to_int = dict(vocab)
        self.int_to_str = {index: token for token, index in vocab.items()}

    def encode(self, text: str) -> list[int]:
        return [self.str_to_int[token] for token in tokenize(text)]

    def decode(self, ids: Iterable[int]) -> str:
        text = " ".join(self.int_to_str[index] for index in ids)
        return _PUNCTUATION_SPACING.sub(r"\1", text)


class SimpleTokenizerV2(SimpleTokenizerV1):
    """Replace out-of-vocabulary words with <|unk|>."""

    def __init__(self, vocab: dict[str, int]) -> None:
        if UNKNOWN_TOKEN not in vocab or END_OF_TEXT not in vocab:
            raise ValueError("V2 requires <|unk|> and <|endoftext|> in the vocabulary")
        super().__init__(vocab)

    def encode(self, text: str) -> list[int]:
        unknown_id = self.str_to_int[UNKNOWN_TOKEN]
        return [self.str_to_int.get(token, unknown_id) for token in tokenize(text)]
