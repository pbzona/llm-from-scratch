"""Chapter 2 examples: tokenization, batching, and embeddings."""

import re
from importlib.resources import files

import tiktoken
import torch

from llm_from_scratch.data_loading import create_dataloader_v1
from llm_from_scratch.tokenization import (
    END_OF_TEXT,
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    build_vocabulary,
    tokenize,
)


def load_verdict() -> str:
    """Read Edith Wharton's public-domain short story "The Verdict"."""
    return (
        files("llm_from_scratch.examples")
        .joinpath("data/chapter02/the-verdict.txt")
        .read_text(encoding="utf-8")
    )


# Book 2.2: Tokenizing text (listing 2.1).
def example_tokenization() -> None:
    """Load the story and split text into words and punctuation."""
    print("\n2.2 Tokenizing text (listing 2.1)")
    raw_text = load_verdict()
    print("Total number of characters:", len(raw_text))
    print(raw_text[:99])

    text = "Hello, world. This, is a test."
    print("Whitespace split:", re.split(r"(\s)", text))
    punctuation_split = re.split(r"([,.]|\s)", text)
    print("Punctuation split:", punctuation_split)
    print("Without whitespace:", [part for part in punctuation_split if part.strip()])
    print("Extended punctuation:", tokenize("Hello, world. Is this-- a test?"))
    preprocessed = tokenize(raw_text)
    print("Number of word/punctuation tokens:", len(preprocessed))
    print("First 30 tokens:", preprocessed[:30])


# Book 2.3: Converting tokens into token IDs (listings 2.2 and 2.3).
def example_token_ids() -> None:
    """Build a vocabulary and try the first simple tokenizer."""
    print("\n2.3 Converting tokens into token IDs (listings 2.2 and 2.3)")
    vocab = build_vocabulary(load_verdict())
    print("Vocabulary size:", len(vocab))
    print("First 51 vocabulary entries:", list(vocab.items())[:51])
    tokenizer_v1 = SimpleTokenizerV1(vocab)
    text = (
        '"It\'s the last he painted, you know," '
        "Mrs. Gisburn said with pardonable pride."
    )
    ids = tokenizer_v1.encode(text)
    print("Token IDs:", ids)
    print("Decoded:", tokenizer_v1.decode(ids))
    try:
        tokenizer_v1.encode("Hello, do you like tea?")
    except KeyError as error:
        print(f"Expected V1 failure on an unknown word: {error}")


# Book 2.4: Adding special context tokens (listing 2.4).
def example_special_tokens() -> None:
    """Mark document boundaries and replace unknown words."""
    print("\n2.4 Adding special context tokens (listing 2.4)")
    vocab = build_vocabulary(load_verdict(), include_special_tokens=True)
    print("Vocabulary size with special tokens:", len(vocab))
    print("Last five entries:", list(vocab.items())[-5:])
    tokenizer_v2 = SimpleTokenizerV2(vocab)
    text = (
        f"Hello, do you like tea? {END_OF_TEXT} In the sunlit terraces of the palace."
    )
    ids = tokenizer_v2.encode(text)
    print("Input:", text)
    print("Token IDs:", ids)
    print("Decoded:", tokenizer_v2.decode(ids))


# Book 2.5: Byte pair encoding.
def example_byte_pair_encoding() -> None:
    """Encode and decode unfamiliar words with GPT-2's tokenizer."""
    print("\n2.5 Byte pair encoding")
    tokenizer = tiktoken.get_encoding("gpt2")
    text = (
        f"Hello, do you like tea? {END_OF_TEXT} "
        "In the sunlit terraces of someunknownPlace."
    )
    ids = tokenizer.encode(text, allowed_special={END_OF_TEXT})
    print("GPT-2 vocabulary size:", tokenizer.n_vocab)
    print("BPE token IDs:", ids)
    print("Decoded:", tokenizer.decode(ids))


# Book exercise 2.1: Byte pair encoding of unknown words.
def exercise_byte_pair_encoding_unknown_words() -> None:
    """Inspect the individual BPE tokens of an unfamiliar word."""
    print("\nExercise 2.1: unknown-word subword tokens")
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Akwirw ier"
    ids = tokenizer.encode(text)
    print("Token IDs:", ids)
    print("Individual tokens:", [tokenizer.decode([token_id]) for token_id in ids])
    print("Decoded:", tokenizer.decode(ids))


# Book 2.6: Data sampling with a sliding window (listings 2.5 and 2.6).
def example_sliding_window() -> None:
    """Create next-token targets and compare consecutive windows."""
    print("\n2.6 Data sampling with a sliding window (listings 2.5 and 2.6)")
    raw_text = load_verdict()
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_text = tokenizer.encode(raw_text)
    print("Number of BPE tokens:", len(encoded_text))
    sample = encoded_text[50:]
    context_size = 4
    print("x:", sample[:context_size])
    print("y:", sample[1 : context_size + 1])
    for size in range(1, context_size + 1):
        context, target = sample[:size], sample[size]
        print(f"{context} -> {target}")
        print(f"{tokenizer.decode(context)!r} -> {tokenizer.decode([target])!r}")

    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )
    data_iter = iter(dataloader)
    print("First batch (stride=1):", next(data_iter))
    print("Second batch (stride=1):", next(data_iter))


# Book exercise 2.2: Data loaders with different strides and context sizes.
def exercise_context_sizes_and_strides() -> None:
    """Compare context lengths 2 and 8 with a stride of 2."""
    print("\nExercise 2.2: different strides and context sizes")
    raw_text = load_verdict()
    for max_length, stride in ((2, 2), (8, 2)):
        dataloader = create_dataloader_v1(
            raw_text,
            batch_size=1,
            max_length=max_length,
            stride=stride,
            shuffle=False,
        )
        data_iter = iter(dataloader)
        print(f"max_length={max_length}, stride={stride}")
        print("First batch:", next(data_iter))
        print("Second batch:", next(data_iter))


# Book 2.6: Collect eight nonoverlapping input-target windows in a batch.
def example_batched_inputs() -> None:
    """Show a full batch of input and target tensors."""
    print("\n2.6 Batched inputs and targets")
    dataloader = create_dataloader_v1(
        load_verdict(), batch_size=8, max_length=4, stride=4, shuffle=False
    )
    inputs, targets = next(iter(dataloader))
    print("Batched inputs:\n", inputs)
    print("Batched targets:\n", targets)


# Book 2.7: Creating token embeddings.
def example_token_embeddings() -> None:
    """Inspect a small embedding table and look up token vectors."""
    print("\n2.7 Creating token embeddings")
    torch.manual_seed(123)
    input_ids = torch.tensor([2, 3, 5, 1])
    embedding_layer = torch.nn.Embedding(6, 3)
    print("Embedding weights:\n", embedding_layer.weight)
    print("Embedding for token 3:\n", embedding_layer(torch.tensor([3])))
    print("Embeddings for [2, 3, 5, 1]:\n", embedding_layer(input_ids))


# Book 2.8: Encoding word positions.
def example_positional_embeddings() -> None:
    """Combine token embeddings with absolute position embeddings."""
    print("\n2.8 Encoding word positions")
    torch.manual_seed(123)
    dataloader = create_dataloader_v1(
        load_verdict(), batch_size=8, max_length=4, stride=4, shuffle=False
    )
    inputs, _ = next(iter(dataloader))
    tokenizer = tiktoken.get_encoding("gpt2")
    output_dim = 256
    context_length = inputs.shape[1]
    token_embedding_layer = torch.nn.Embedding(tokenizer.n_vocab, output_dim)
    token_embeddings = token_embedding_layer(inputs)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    input_embeddings = token_embeddings + pos_embeddings
    print("Inputs shape:", inputs.shape)
    print("Token embeddings shape:", token_embeddings.shape)
    print("Position embeddings shape:", pos_embeddings.shape)
    print("Input embeddings shape:", input_embeddings.shape)


def run_all_examples() -> None:
    """Run the full walkthrough in reading order."""
    example_tokenization()
    example_token_ids()
    example_special_tokens()
    example_byte_pair_encoding()
    exercise_byte_pair_encoding_unknown_words()
    example_sliding_window()
    exercise_context_sizes_and_strides()
    example_batched_inputs()
    example_token_embeddings()
    example_positional_embeddings()


def main() -> None:
    """Choose which examples to run by uncommenting their calls below."""
    example_tokenization()
    # example_token_ids()
    # example_special_tokens()
    # example_byte_pair_encoding()
    # exercise_byte_pair_encoding_unknown_words()
    # example_sliding_window()
    # exercise_context_sizes_and_strides()
    # example_batched_inputs()
    # example_token_embeddings()
    # example_positional_embeddings()


if __name__ == "__main__":
    main()
