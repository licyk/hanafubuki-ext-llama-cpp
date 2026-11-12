"""Builds the extension's installable archive: ``dist/<id>-<version>.zip``.

The archive holds the package's files at its root (what Hanafubuki reads)
and nothing of the repository's development: no tests, tools, caches or Git.
``latest.json`` is written beside it: the update source the manifest's
``[update] url`` names, one entry for every system, whose ``download_url``
is the archive as the release ``v<version>`` publishes it.

Usage: ``python scripts/package.py [--out DIR]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent
# What the package is: these files, and every file under these directories.
FILES = ("hanafubuki-extension.toml", "LICENSE", "README.md", "scripts/install.py")
DIRECTORIES = ("backend", "hanafubuki_llama", "images", "locales", "schemas", "types", "ui")
SKIPPED_PARTS = ("__pycache__",)
# The newest release's asset, as the manifest's `[update] url` names it.
LATEST_ASSET = "/releases/latest/download/latest.json"


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
    """Writes the archive and its ``latest.json``.

    Args:
        out: The directory they go in.
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
    entries = latest(archive, manifest["version"], manifest["update"]["url"])
    (out / "latest.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return archive


def latest(archive: Path, version: str, update_url: str) -> list[dict[str, str]]:
    """``latest.json`` for an archive: one entry for any system.

    Args:
        archive: The archive.
        version: Its version, published as the release ``v<version>``.
        update_url: The manifest's ``[update] url``, a GitHub release's ``latest.json``.

    Returns:
        The entries.

    Raises:
        ValueError: The update source is not a GitHub release's.
    """
    if not update_url.endswith(LATEST_ASSET):
        raise ValueError(f"[update] url is not a GitHub release's latest.json: {update_url}")
    releases = update_url.removesuffix(LATEST_ASSET)
    return [
        {
            "version": version,
            "download_url": f"{releases}/releases/download/v{version}/{archive.name}",
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "platform": "any",
            "architecture": "any",
        }
    ]


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
