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

The project is on **continuous deployment for source changes**: every commit
to `master` that touches source files automatically cuts a new release.

A commit triggers a release when it changes any of:

- `src/PyDiffGame/**` — the package itself
- `pyproject.toml` — packaging metadata, dependencies, classifiers

Docs (`*.md`, `docs/**`), tests (`tests/**`), tooling (`tools/**`, `.github/**`,
`.pre-commit-config.yaml`), images and the lock file do **not** trigger a
release on their own. A mixed commit (e.g. `src/foo.py` + `README.md`) does
trigger one — the path filter matches if any changed file matches.

When a release-triggering commit lands on `master`, the publish workflow:

1. **Increments the version** with `tools/bump_version.py`, which rolls each
   component over at 9 (`2.0.9 -> 2.1.0`, `2.9.9 -> 3.0.0`), updates both
   `pyproject.toml` and `src/PyDiffGame/__init__.py`, and commits the bump back
   to `master` as `chore: bump version to X.Y.Z [skip ci]`.
2. Builds the distributions with `uv build`.
3. Uploads them to PyPI via
   [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC — no
   tokens or secrets).
4. Creates a `v<version>` GitHub Release with auto-generated notes and the built
   wheels/sdist attached.

The workflow can also be triggered manually from **Actions -> Upload Python
Package -> Run workflow** if you ever need to re-run it.

You normally never edit the version by hand. To bump it locally (e.g. to test),
run `uv run python tools/bump_version.py` (`--dry-run` to preview, `--current`
to print the current version). The PyPI upload is idempotent (`skip-existing`),
so re-running the workflow is safe.

### What stops an infinite loop?

The publish workflow itself commits the version bump back to `master`. Three
independent guards stop that commit from re-triggering the workflow:

1. The bump commit message contains `[skip ci]`, which GitHub Actions natively
   honors by **not creating a workflow run at all** for that push.
2. The `bump-version` job has an explicit `if:` that skips when the actor is
   `github-actions[bot]` or the head-commit message contains `[skip ci]`.
3. `bump_version.py` is the single source of truth for the version, so a
   tampered bump commit still wouldn't double-bump.

Thank you for your contribution!

Joshua Shay Kricheli
