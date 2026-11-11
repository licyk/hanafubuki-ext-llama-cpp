"""Builds the extension's installable archive: ``dist/<id>-<version>.zip``.

The archive holds the package's files at its root (what Hanafubuki reads)
and nothing of the repository's development: no tests, tools, caches or Git.

Usage: ``python scripts/package.py [--out DIR]``.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent
# What the package is: these files, and every file under these directories.
FILES = ("hanafubuki-extension.toml", "LICENSE", "README.md", "scripts/install.py")
DIRECTORIES = ("backend", "hanafubuki_llama", "images", "locales", "schemas", "types", "ui")
SKIPPED_PARTS = ("__pycache__",)


def package_files(root: Path = ROOT) -> list[str]:
    """The package's files, relative and ``/``-separated, in order.

    Args:
        root: The repository.

    Returns:
        The files.
    """
    found = set(FILES)
    for directory in DIRECTORIES:
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root)
            skipped = any(part in SKIPPED_PARTS for part in relative.parts)
            if path.is_file() and not skipped and path.suffix not in (".pyc", ".pyo"):
                found.add(relative.as_posix())
    return sorted(found)


def build(out: Path, root: Path = ROOT) -> Path:
    """Writes the archive.

    Args:
        out: The directory it goes in.
        root: The repository.

    Returns:
        The archive's path.
    """
    manifest = tomllib.loads((root / "hanafubuki-extension.toml").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    archive = out / f"{manifest['id']}-{manifest['version']}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as target:
        for name in package_files(root):
            target.write(root / name, name)
    return archive


def main() -> int:
    """Builds the archive and prints where it is.

    Returns:
        The exit code.
    """
    parser = argparse.ArgumentParser(description="Build the extension's archive.")
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    archive = build(parser.parse_args().out)
    print(archive)
    return 0


if __name__ == "__main__":
    sys.exit(main())
