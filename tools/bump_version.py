#!/usr/bin/env python3
"""Increment the project version using single-digit, carry-at-9 components.

The version is ``X.Y.Z`` where each component rolls over at 9: incrementing the
patch past 9 carries into the minor, and the minor past 9 carries into the major
(e.g. ``2.0.9 -> 2.1.0`` and ``2.9.9 -> 3.0.0``). The major keeps growing.

It updates the single source of truth in both ``pyproject.toml`` and
``src/PyDiffGame/__init__.py`` and prints the new version to stdout.

Usage::

    python tools/bump_version.py            # bump the files, print the new version
    python tools/bump_version.py --dry-run  # print the next version without writing
    python tools/bump_version.py --current  # print the current version
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "src" / "PyDiffGame" / "__init__.py"

_PYPROJECT_RE = re.compile(r'^(version\s*=\s*)"([^"]+)"', re.MULTILINE)
_INIT_RE = re.compile(r'^(__version__\s*=\s*)"([^"]+)"', re.MULTILINE)


def increment_version(version: str) -> str:
    """Return the next version, rolling each component over at 9 (carry up)."""
    parts = version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"expected an 'X.Y.Z' numeric version, got {version!r}")
    major, minor, patch = (int(p) for p in parts)
    patch += 1
    if patch > 9:
        patch, minor = 0, minor + 1
        if minor > 9:
            minor, major = 0, major + 1
    return f"{major}.{minor}.{patch}"


def read_current_version() -> str:
    match = _PYPROJECT_RE.search(PYPROJECT.read_text())
    if match is None:
        raise SystemExit("could not find 'version = \"...\"' in pyproject.toml")
    return match.group(2)


def write_version(new_version: str) -> None:
    """Write ``new_version`` into both pyproject.toml and __init__.py."""
    PYPROJECT.write_text(_PYPROJECT_RE.sub(rf'\1"{new_version}"', PYPROJECT.read_text(), count=1))
    INIT.write_text(_INIT_RE.sub(rf'\1"{new_version}"', INIT.read_text(), count=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="print the next version without writing")
    group.add_argument("--current", action="store_true", help="print the current version and exit")
    args = parser.parse_args()

    current = read_current_version()
    if args.current:
        print(current)
        return

    new_version = increment_version(current)
    if not args.dry_run:
        write_version(new_version)
    print(new_version)


if __name__ == "__main__":
    main()
