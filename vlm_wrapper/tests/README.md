# `vlm_wrapper.tests`

Pytest package for schema parsers, validators, and **live** VLM calls (gated with `-m live` where applicable).

## Running tests

From the repo root, with the `vlm_benchmarks` conda env (or equivalent) activated:

```bash
pytest vlm_wrapper/tests -q
```

Live API tests (require keys and may bill):

```bash
pytest vlm_wrapper/tests/test_gpt4o_parsers.py -m live -q
```

Shared fixtures live in `conftest.py`. Parent documentation: [`../README.md`](../README.md).
