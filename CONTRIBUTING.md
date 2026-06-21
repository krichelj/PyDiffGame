# Contribution Guidelines

This repo is part of a research conducted at Ben Gurion University,
and is thus open source. We would love to receive your input!

Please ensure your pull request adheres to the following guidelines:

- Search previous suggestions before making a new one, as yours may be a duplicate.
- Make an individual pull request for each suggestion.
- New categories or improvements to the existing categorization are welcome.
- Check your spelling and grammar.
- Make sure your text editor is set to remove trailing whitespace.
- The pull request and commit should have a useful title.

## Development setup

PyDiffGame requires **Python >= 3.11** and uses [uv](https://docs.astral.sh/uv/)
for environment and dependency management. Install uv, sync the locked
development environment, and enable the pre-commit hooks so the code-quality
tools run automatically on every commit:

```bash
# install uv: https://docs.astral.sh/uv/getting-started/installation/
uv sync --extra dev
uv run pre-commit install
```

`uv sync` creates the virtual environment and installs the exact, locked
dependencies. Run the tooling through `uv run`:

```bash
uv run ruff format src/PyDiffGame tests   # auto-format (black-compatible)
uv run ruff check src/PyDiffGame tests    # lint
uv run mypy src/PyDiffGame                # type-check
uv run pytest                             # test suite
```

Continuous integration runs the formatter check, the linter, the type checker
and the full suite (all via uv) across Python 3.11–3.14, so please make sure
they pass locally.

## Releasing

Publishing a new version to PyPI and creating the matching GitHub Release is a
single automated step:

1. Bump `version` in `pyproject.toml` (e.g. `2.0.0` -> `2.1.0`) and merge it to
   `master`.
2. Run the publish workflow: **Actions -> Upload Python Package -> Run workflow**
   (on `master`).

That run builds the distributions with `uv build`, uploads them to PyPI via
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC — no
tokens or secrets), and then creates a `v<version>` GitHub Release with
auto-generated notes and the built wheels/sdist attached.

The workflow is idempotent: re-running it for a version already on PyPI skips
the upload (`skip-existing`) and just refreshes the release assets, so it is
safe to re-run.

Thank you for your contribution!

Joshua Shay Kricheli
