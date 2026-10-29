# Example conventions

## Structure

- Put shared implementation in top-level modules named for the concept they
  implement. Use names such as `tokenization.py`, `data_loading.py`, and
  `exercise_context_sizes_and_strides()`.
- Keep shared modules chapter-agnostic. They should not reference book sections,
  import example modules, or load chapter-specific data.
- Put runnable demonstrations in `src/llm_from_scratch/examples/chapterNN.py`,
  one module per book chapter.
- Put chapter-specific sample data under
  `src/llm_from_scratch/examples/data/chapterNN/`.
- Put book section, listing, and exercise references in comments above the
  example functions. Update those comments when the example changes.
- Make example functions callable without arguments. Each function prepares its
  own data, tokenizers, and layers so it can run alone or in any order.
- Keep reusable logic separate from the example functions that print
  demonstrations. See `src/llm_from_scratch/examples/chapter02.py` for the
  pattern.
- Initialize only the resources the selected example needs. Simple examples
  should work without downloading GPT-2's encoding tables.
- Seed random examples within their own functions for reproducible results.

## Execution and output

- Keep `main()` a short, editable list of example calls. Enable one example by
  default and leave the other calls commented out for easy switching.
- Preserve the user's selected calls when making unrelated changes.
- Keep the full walkthrough available through an explicit
  `run_all_examples()` function, with calls in reading order.
- Print only the selected example's heading and results. Prepare prerequisites
  directly or through quiet helpers, without running other demonstrations.
- Keep imports quiet and guard script execution with `if __name__ == "__main__":`.

## Verification

- Put shared-component tests in `tests/test_<concept>.py` and chapter-specific
  example tests in `tests/examples/test_chapterNN.py`.
- When adding or refactoring examples, test each function independently and
  assert that its output excludes other examples' headings.
- Check that `main()` runs only the selected examples and that
  `run_all_examples()` still completes in reading order.
- Run focused tests and lint/format checks for the changed Python files.
  See `README.md` for the project's commands and single-example usage.
