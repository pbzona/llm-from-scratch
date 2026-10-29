# LLM from scratch

Python implementations of text-processing components for training a language
model. Reusable code lives at the package root, and book demonstrations live
under `examples/`, grouped by chapter.

The package includes word-level tokenization, vocabulary construction, special
tokens, byte pair encoding examples, sliding-window datasets, and token and
position embeddings. Everything runs on the CPU; no model weights, API keys, or
training are needed.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Run an example

Each demonstration is a no-argument function in
`src/llm_from_scratch/examples/chapter02.py`. Call one directly:

```sh
uv run python -c "from llm_from_scratch.examples.chapter02 import example_tokenization; example_tokenization()"
```

Available functions:

- Tokenization: `example_tokenization`, `example_token_ids`,
  `example_special_tokens`, `example_byte_pair_encoding`,
  `exercise_byte_pair_encoding_unknown_words`
- Data loading: `example_sliding_window`,
  `exercise_context_sizes_and_strides`, `example_batched_inputs`
- Embeddings: `example_token_embeddings`, `example_positional_embeddings`

To run the whole chapter walkthrough:

```sh
uv run python -c "from llm_from_scratch.examples.chapter02 import run_all_examples; run_all_examples()"
```

The installed command runs `main()` from the chapter module. Edit `main()` to
choose which calls it includes, then run:

```sh
uv run llm-from-scratch
```

The chapter module can also be run directly with the same result:

```sh
uv run python -m llm_from_scratch.examples.chapter02
```

The byte pair encoding and GPT-2 data-loading examples use `tiktoken`. It
downloads the encoding tables on first use, which requires network access.
Later runs use its local cache.

## Use the components

```python
from llm_from_scratch.data_loading import create_dataloader_v1
from llm_from_scratch.tokenization import SimpleTokenizerV1, build_vocabulary

text = "Hello, world. This, is a test."
tokenizer = SimpleTokenizerV1(build_vocabulary(text))
ids = tokenizer.encode(text)
print(tokenizer.decode(ids))

loader = create_dataloader_v1(
    text,
    batch_size=1,
    max_length=4,
    stride=4,
    shuffle=False,
    tokenizer=tokenizer,
)
inputs, targets = next(iter(loader))
```

If no tokenizer is supplied, `create_dataloader_v1` uses GPT-2's tokenizer.

## Layout

- `src/llm_from_scratch/tokenization.py`: text splitting, vocabularies, and
  simple tokenizers.
- `src/llm_from_scratch/data_loading.py`: sliding-window dataset and data
  loader.
- `src/llm_from_scratch/examples/chapter02.py`: chapter demonstrations and
  entry point.
- `src/llm_from_scratch/examples/data/chapter02/`: chapter sample text.

## Test

```sh
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```
