"""Book-output checks and example isolation for Chapter 2."""

import pytest
import tiktoken

from llm_from_scratch.data_loading import create_dataloader_v1
from llm_from_scratch.examples import chapter02 as examples
from llm_from_scratch.tokenization import (
    END_OF_TEXT,
    UNKNOWN_TOKEN,
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    build_vocabulary,
    tokenize,
)

EXAMPLES = (
    (examples.example_tokenization, "2.2 Tokenizing"),
    (examples.example_token_ids, "2.3 Converting"),
    (examples.example_special_tokens, "2.4 Adding"),
    (examples.example_byte_pair_encoding, "2.5 Byte pair"),
    (examples.exercise_byte_pair_encoding_unknown_words, "Exercise 2.1"),
    (examples.example_sliding_window, "2.6 Data sampling"),
    (examples.exercise_context_sizes_and_strides, "Exercise 2.2"),
    (examples.example_batched_inputs, "2.6 Batched"),
    (examples.example_token_embeddings, "2.7 Creating"),
    (examples.example_positional_embeddings, "2.8 Encoding"),
)


@pytest.fixture(scope="module")
def raw_text():
    return examples.load_verdict()


@pytest.fixture(scope="module")
def vocab(raw_text):
    return build_vocabulary(raw_text)


@pytest.fixture(scope="module")
def gpt2():
    return tiktoken.get_encoding("gpt2")


def test_story_and_token_counts_match_book(raw_text, vocab):
    assert len(raw_text) == 20479
    assert raw_text.startswith("I HAD always thought Jack Gisburn")
    tokens = tokenize(raw_text)
    assert len(tokens) == 4690
    assert tokens[:10] == [
        "I",
        "HAD",
        "always",
        "thought",
        "Jack",
        "Gisburn",
        "rather",
        "a",
        "cheap",
        "genius",
    ]
    assert len(vocab) == 1130
    assert list(vocab.items())[:3] == [("!", 0), ('"', 1), ("'", 2)]
    assert list(vocab.items())[-1] == ("yourself", 1129)
    assert list(vocab.values()) == list(range(len(vocab)))


def test_v1_ids_match_book(vocab):
    tokenizer = SimpleTokenizerV1(vocab)
    text = '"It\'s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
    ids = tokenizer.encode(text)
    assert ids == [
        1,
        56,
        2,
        850,
        988,
        602,
        533,
        746,
        5,
        1126,
        596,
        5,
        1,
        67,
        7,
        38,
        851,
        1108,
        754,
        793,
        7,
    ]
    assert (
        tokenizer.decode(ids)
        == '" It\' s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
    )


def test_v2_special_tokens_and_unknown_words_match_book(raw_text):
    vocab = build_vocabulary(raw_text, include_special_tokens=True)
    assert len(vocab) == 1132
    assert vocab[END_OF_TEXT] == 1130
    assert vocab[UNKNOWN_TOKEN] == 1131
    tokenizer = SimpleTokenizerV2(vocab)
    text = (
        f"Hello, do you like tea? {END_OF_TEXT} In the sunlit terraces of the palace."
    )
    ids = tokenizer.encode(text)
    assert ids == [
        1131,
        5,
        355,
        1126,
        628,
        975,
        10,
        1130,
        55,
        988,
        956,
        984,
        722,
        988,
        1131,
        7,
    ]
    assert (
        tokenizer.decode(ids)
        == f"{UNKNOWN_TOKEN}, do you like tea? {END_OF_TEXT} In the sunlit terraces of the {UNKNOWN_TOKEN}."
    )
    assert tokenizer.encode(tokenizer.decode(ids)) == ids


def test_byte_pair_encoding_round_trip_and_special_token(gpt2):
    text = f"Hello, do you like tea? {END_OF_TEXT} In the sunlit terraces of someunknownPlace."
    ids = gpt2.encode(text, allowed_special={END_OF_TEXT})
    assert gpt2.n_vocab == 50257
    assert ids == [
        15496,
        11,
        466,
        345,
        588,
        8887,
        30,
        220,
        50256,
        554,
        262,
        4252,
        18250,
        8812,
        2114,
        286,
        617,
        34680,
        27271,
        13,
    ]
    assert gpt2.decode(ids) == text


def test_byte_pair_encoding_unknown_subwords(gpt2):
    text = "Akwirw ier"
    ids = gpt2.encode(text)
    assert ids == [33901, 86, 343, 86, 220, 959]
    assert [gpt2.decode([token_id]) for token_id in ids] == [
        "Ak",
        "w",
        "ir",
        "w",
        " ",
        "ier",
    ]
    assert gpt2.decode(ids) == text


def test_encoded_story_and_context_match_book(raw_text, gpt2):
    ids = gpt2.encode(raw_text)
    assert len(ids) == 5145
    assert ids[50:54] == [290, 4920, 2241, 287]
    assert ids[51:55] == [4920, 2241, 287, 257]


def test_first_two_batches_match_book(raw_text):
    loader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )
    batches = iter(loader)
    inputs, targets = next(batches)
    assert inputs.tolist() == [[40, 367, 2885, 1464]]
    assert targets.tolist() == [[367, 2885, 1464, 1807]]
    inputs, targets = next(batches)
    assert inputs.tolist() == [[367, 2885, 1464, 1807]]
    assert targets.tolist() == [[2885, 1464, 1807, 3619]]


def test_eight_sample_batch_matches_book(raw_text):
    loader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
    )
    inputs, targets = next(iter(loader))
    assert inputs.shape == targets.shape == (8, 4)
    assert inputs.tolist() == [
        [40, 367, 2885, 1464],
        [1807, 3619, 402, 271],
        [10899, 2138, 257, 7026],
        [15632, 438, 2016, 257],
        [922, 5891, 1576, 438],
        [568, 340, 373, 645],
        [1049, 5975, 284, 502],
        [284, 3285, 326, 11],
    ]
    assert targets[-1].tolist() == [3285, 326, 11, 287]


@pytest.mark.parametrize(("example", "heading"), EXAMPLES)
def test_examples_run_independently_without_other_example_output(
    example, heading, capsys
):
    example()
    captured = capsys.readouterr()
    assert heading in captured.out
    assert captured.err == ""
    for _, other_heading in EXAMPLES:
        if other_heading != heading:
            assert other_heading not in captured.out


def test_main_runs_only_tokenization_by_default(capsys):
    examples.main()
    output = capsys.readouterr().out
    assert "2.2 Tokenizing" in output
    for _, heading in EXAMPLES[1:]:
        assert heading not in output


@pytest.mark.parametrize(
    "example",
    [
        examples.example_tokenization,
        examples.example_token_ids,
        examples.example_special_tokens,
        examples.example_token_embeddings,
    ],
)
def test_non_bpe_examples_do_not_initialize_gpt2(example, monkeypatch, capsys):
    def unexpected_encoding(*args, **kwargs):
        pytest.fail("This example should not initialize the GPT-2 tokenizer")

    monkeypatch.setattr(tiktoken, "get_encoding", unexpected_encoding)
    example()
    assert capsys.readouterr().out


def test_walkthrough_reaches_final_embeddings(capsys):
    examples.run_all_examples()
    output = capsys.readouterr().out
    positions = [output.index(heading) for _, heading in EXAMPLES]
    assert positions == sorted(positions)
    assert "Expected V1 failure" in output
    assert "Input embeddings shape: torch.Size([8, 4, 256])" in output
