# MetaInsight Python scaffold (uv)

This repository now contains a companion Python project scaffold to support a full
Python implementation of MetaInsight on top of the existing R package code.

## Layout

- `pyproject.toml` - uv + PEP 621 project metadata and dependency definitions
- `src/metainsight/` - Python package sources
- `src/metainsight/core.py` - initial data-loading and configuration API
- `src/metainsight/cli.py` - lightweight command-line interface
- `src/metainsight/__main__.py` - `python -m metainsight`
- `tests/python/` - pytest suite for the scaffold
- `uv.lock` - resolved dependency versions

## Setup (local)

```bash
uv sync --group dev
```

Run tests:

```bash
uv run pytest
```

Run CLI help:

```bash
uv run metainsight --help
```

Run with data file:

```bash
uv run metainsight path/to/study_data.csv --outcome "Continuous"
```

## Next steps

- Move domain models and calculation logic from R into `src/metainsight` modules.
- Add model-fit backends (frequentist, Bayesian) and plotting/export layers.
- Expand tests using parity fixtures against legacy R outputs.
