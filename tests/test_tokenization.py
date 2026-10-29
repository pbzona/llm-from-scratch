"""Checks for word-level tokenization and vocabulary mapping."""

import pytest

from llm_from_scratch.tokenization import (
    END_OF_TEXT,
    UNKNOWN_TOKEN,
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    build_vocabulary,
    tokenize,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Hello, world. Is this-- a test?",
            ["Hello", ",", "world", ".", "Is", "this", "--", "a", "test", "?"],
        ),
        ("a:b;c_d!", ["a", ":", "b", ";", "c", "_", "d", "!"]),
        ("(It's)\n\t fine", ["(", "It", "'", "s", ")", "fine"]),
        (" \n\t", []),
        ("", []),
    ],
)
def test_tokenization(text, expected):
    assert tokenize(text) == expected


def test_vocabulary_is_sorted_and_order_independent():
    assert build_vocabulary("z a z b") == {"a": 0, "b": 1, "z": 2}
    assert build_vocabulary("b a z") == build_vocabulary("z a z b")
    assert build_vocabulary("") == {}


def test_v1_round_trip_preserves_normalized_text():
    text = "Hello, world. Is this a test?"
    tokenizer = SimpleTokenizerV1(build_vocabulary(text))
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert tokenizer.encode(tokenizer.decode(ids)) == ids


def test_v1_rejects_unknown_words_and_ids():
    tokenizer = SimpleTokenizerV1(build_vocabulary("Hello world"))
    with pytest.raises(KeyError, match="unknown"):
        tokenizer.encode("Hello unknown")
    with pytest.raises(KeyError):
        tokenizer.decode([2])


def test_simple_tokenizer_normalizes_whitespace_and_punctuation():
    text = "Hello, world: a test; really?"
    tokenizer = SimpleTokenizerV1(build_vocabulary(text))
    assert tokenizer.decode(tokenizer.encode("Hello,\nworld: a\ttest; really?")) == text
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_v2_replaces_unknown_words():
    vocab = build_vocabulary("Hello world", include_special_tokens=True)
    tokenizer = SimpleTokenizerV2(vocab)
    assert tokenizer.encode("Hello unknown") == [vocab["Hello"], vocab[UNKNOWN_TOKEN]]
    assert tokenizer.decode(tokenizer.encode("Hello unknown")) == "Hello <|unk|>"


def test_special_tokens_are_not_duplicated():
    vocab = build_vocabulary(
        f"{END_OF_TEXT} {UNKNOWN_TOKEN}", include_special_tokens=True
    )
    assert len(vocab) == 2
    assert sorted(vocab.values()) == [0, 1]


@pytest.mark.parametrize("vocab", [{}, {UNKNOWN_TOKEN: 0}, {END_OF_TEXT: 0}])
def test_v2_requires_special_tokens(vocab):
    with pytest.raises(ValueError, match="V2 requires"):
        SimpleTokenizerV2(vocab)
