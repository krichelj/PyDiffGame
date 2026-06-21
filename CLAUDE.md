# PyDiffGame — repository guide for Claude

## Python tooling (uv-first)
- Use **uv** for everything; pip is only a documented fallback.
  - `uv sync --extra dev`, `uv run pytest`, `uv run ruff check`, `uv run ruff format`,
    `uv run mypy src/PyDiffGame`, `uv build`.
- Keep the quality gates green before committing: ruff format, ruff check, mypy on
  `src/PyDiffGame`, and the pytest suite — all via `uv run`.

## Versioning (carry-at-9)
- The version is `X.Y.Z` with single-digit components that roll over at 9:
  `2.0.9 -> 2.1.0`, `2.9.9 -> 3.0.0` (the major keeps growing).
- Increment **only** via `tools/bump_version.py`, which updates the version in both
  `pyproject.toml` and `src/PyDiffGame/__init__.py`. Never hand-edit version strings.
- The version is bumped **automatically** by the publish workflow on each release, so
  do not bump it in ordinary PRs — the release run does it.

## Releasing
- `Actions -> Upload Python Package -> Run workflow` (on `master`). The workflow
  auto-increments the version, commits it to `master`, builds with `uv build`,
  publishes to PyPI via Trusted Publishing (OIDC, no tokens), and creates the matching
  `v<version>` GitHub Release with notes and the built dists attached. It is idempotent
  (`skip-existing`).

## Docs
- `README.md` is the single canonical readme and is also the PyPI long-description
  (`pyproject.toml: readme = "README.md"`), so its image/file links must be **absolute**
  (`raw.githubusercontent.com/.../master/...` for images, `github.com/.../blob/master/...`
  for files) so they render on PyPI.
- Keep `docs/README.md` identical to `README.md`.
- README figures are generated from the live solver:
  `uv run python tools/generate_readme_figures.py`.
