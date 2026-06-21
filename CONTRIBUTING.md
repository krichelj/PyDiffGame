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

PyDiffGame requires **Python >= 3.11**. Install the package from source with the
development extras, then enable the pre-commit hooks so the code-quality tools
run automatically on every commit:

```bash
pip install -e ".[dev]"
pre-commit install
```

The tooling can also be run by hand:

```bash
ruff format src/PyDiffGame tests     # auto-format (black-compatible)
ruff check src/PyDiffGame tests      # lint
mypy src/PyDiffGame                  # type-check
pytest                               # test suite
```

Continuous integration runs the formatter check, the linter, the type checker
and the full suite across Python 3.11–3.14, so please make sure they pass
locally.

Thank you for your contribution!

Joshua Shay Kricheli
