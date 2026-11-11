"""Installs or updates llama.cpp in an instance, and downloads its Hugging Face model.

Run by the instance's install plan as a ``run`` step on the application's
managed Python (``{host.python}``), which puts ``sd_webui_all_in_one`` on the
import path. It writes only inside ``--core`` and ``--models``:

1. Resolve the release on GitHub (``latest``: the newest one that already
   holds this platform's build of the variant).
2. Download the build (and a CUDA build's runtime) into ``core/.staging``
   with ``sd_webui_all_in_one``'s downloader, trying GitHub mirrors as the
   ``--github-mirror`` mode says, and check each file's SHA-256.
3. Extract it with ``sd_webui_all_in_one``'s ``extract_archive``, find the
   directory holding ``llama-server``, and swap it in for the previous build.
4. Record the build in ``core/.hanafubuki-llama.json``.
5. With ``--hf-repo``, download the model file (every part of a split one)
   into ``--models``.

Proxy variables in the environment (the instance's) apply to every request.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import posixpath
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# Write no bytecode beside the extension's package (read-only) or the
# application's verified launcher code, which the imports below read.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sd_webui_all_in_one.archive_manager import extract_archive  # noqa: E402
from sd_webui_all_in_one.downloader.downloader import download_file  # noqa: E402
from sd_webui_all_in_one.mirror_manager import (  # noqa: E402
    GITHUB_MIRROR_LIST,
    HUGGINGFACE_MIRROR_LIST,
)

from hanafubuki_llama.record import RECORD, STAGING, read_record, write_record  # noqa: E402
from hanafubuki_llama.releases import REPOSITORY, digest, host_platform, select  # noqa: E402

GITHUB_API = "https://api.github.com"
HUGGINGFACE = "https://huggingface.co"
_SPLIT = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$")


class InstallError(Exception):
    """What stops the install, said to the user."""


def main(argv: list[str] | None = None) -> int:
    """Runs the install.

    Args:
        argv: The arguments; the process's when ``None``.

    Returns:
        The exit code.
    """
    args = _parse(argv)
    core = Path(args.core)
    models = Path(args.models)
    core.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    try:
        install_build(args, core)
        if args.hf_repo:
            download_model(args, models)
    except InstallError as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


def install_build(args: argparse.Namespace, core: Path) -> None:
    """Puts the requested build in ``core`` unless it is there already.

    Args:
        args: The parsed arguments.
        core: The instance's ``core`` directory.

    Raises:
        InstallError: The release, a download or the archive failed.
    """
    os_name, arch = host_platform()
    selection = _resolve(args, os_name, arch)
    server = core / ("llama-server.exe" if os_name == "windows" else "llama-server")
    record = read_record(core) or {}
    if (
        not args.force
        and record.get("tag") == selection.tag
        and record.get("variant") == args.variant
        and server.is_file()
    ):
        _say(f"llama.cpp {selection.tag} ({args.variant}) is installed already")
        return
    _say(f"Installing llama.cpp {selection.tag} ({args.variant}, {os_name}-{arch})")
    staging = core / STAGING
    _clear_staging(staging)
    downloads = staging / "downloads"
    archive = _download_asset(selection.asset, downloads, args.github_mirror)
    runtime = (
        _download_asset(selection.cudart, downloads, args.github_mirror)
        if selection.cudart
        else None
    )
    build = _unpack(archive, staging / "build")
    build = _server_dir(build, server.name)
    if runtime is not None:
        for entry in _single_top(_unpack(runtime, staging / "runtime")).iterdir():
            _remove(build / entry.name)
            entry.rename(build / entry.name)
    _swap(core, build, staging / "previous")
    write_record(
        core,
        {
            "tag": selection.tag,
            "variant": args.variant,
            "os": os_name,
            "arch": arch,
            "asset": selection.asset["name"],
            "sha256": digest(selection.asset),
            "runtime": selection.cudart["name"] if selection.cudart else None,
            "installedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    shutil.rmtree(staging, ignore_errors=True)
    _say(f"Installed llama.cpp {selection.tag} into {core}")


def download_model(args: argparse.Namespace, models: Path) -> None:
    """Downloads a Hugging Face model file (each part of a split one) into ``models``.

    A file already there with the size the repository lists is kept.

    Args:
        args: The parsed arguments.
        models: The instance's ``models`` directory.

    Raises:
        InstallError: The repository or the file cannot be read or downloaded.
    """
    endpoints = _hf_endpoints(args)
    files = _hf_files(args.hf_repo, posixpath.dirname(args.hf_file), endpoints)
    for path in _model_parts(args.hf_file):
        entry = files.get(path)
        if entry is None:
            raise InstallError(f"{args.hf_repo} has no file {path}")
        target = models / posixpath.basename(path)
        lfs = entry.get("lfs")
        if not isinstance(lfs, dict):
            lfs = {}
        size = lfs.get("size", entry.get("size"))
        if target.is_file() and target.stat().st_size == size:
            _say(f"{target.name} is downloaded already")
            continue
        quoted = urllib.parse.quote(path)
        candidates = [f"{endpoint}/{args.hf_repo}/resolve/main/{quoted}" for endpoint in endpoints]
        _download(candidates, models, target.name, lfs.get("oid"))


def _resolve(args: argparse.Namespace, os_name: str, arch: str):
    api = args.github_api.rstrip("/")
    url = (
        f"{api}/repos/{REPOSITORY}/releases?per_page=20"
        if args.version == "latest"
        else f"{api}/repos/{REPOSITORY}/releases/tags/{urllib.parse.quote(args.version)}"
    )
    try:
        found = _fetch_json(url, "GITHUB_TOKEN")
    except (OSError, ValueError) as error:
        raise InstallError(
            f"cannot read llama.cpp's releases ({url}): {error}. "
            "Check the network or proxy; set GITHUB_TOKEN in the instance's environment "
            "if GitHub's rate limit is reached."
        ) from error
    releases = found if isinstance(found, list) else [found]
    try:
        return select(releases, os_name, arch, args.variant, args.version)
    except LookupError as error:
        raise InstallError(str(error)) from error


def _download_asset(asset: dict[str, Any], directory: Path, mirror: str) -> Path:
    url = str(asset["browser_download_url"])
    return _download(_github_candidates(url, mirror), directory, str(asset["name"]), digest(asset))


def _download(candidates: list[str], directory: Path, name: str, sha256: str | None) -> Path:
    target = directory / name
    if target.is_file():
        # A finished file (a partial one is the downloader's `.tmp`) is kept
        # only when it is the one asked for.
        if sha256 and _sha256(target) == sha256:
            return target
        target.unlink()
    # Never let the library install `requests` into the shared managed Python.
    tool = "requests" if importlib.util.find_spec("requests") else "urllib"
    errors = []
    for url in candidates:
        _say(f"Downloading {name} from {url}")
        try:
            path = download_file(
                url,
                path=directory,
                save_name=name,
                tool=tool,
                progress=False,
                hash_value=sha256,
                connect_timeout=20,
                progress_callback=_Progress(name),
            )
        except Exception as error:
            errors.append(f"{url}: {error}")
            continue
        # The urllib downloader checks no digest: check every download here.
        if sha256 and _sha256(path) != sha256:
            path.unlink(missing_ok=True)
            errors.append(f"{url}: the SHA-256 does not match {sha256}")
            continue
        return path
    raise InstallError(f"cannot download {name}:\n  " + "\n  ".join(errors))


def _github_candidates(url: str, mode: str) -> list[str]:
    prefix = "https://github.com"
    if mode == "off" or not url.startswith(prefix + "/"):
        return [url]
    mirrors = [mirror.rstrip("/") + url[len(prefix) :] for mirror in GITHUB_MIRROR_LIST]
    return [*mirrors, url] if mode == "always" else [url, *mirrors]


def _hf_endpoints(args: argparse.Namespace) -> list[str]:
    if args.hf_endpoint:
        return [args.hf_endpoint.rstrip("/")]
    mirrors = [mirror.rstrip("/") for mirror in HUGGINGFACE_MIRROR_LIST]
    if args.hf_mirror == "off":
        return [HUGGINGFACE]
    return [*mirrors, HUGGINGFACE] if args.hf_mirror == "always" else [HUGGINGFACE, *mirrors]


def _hf_files(repo: str, directory: str, endpoints: list[str]) -> dict[str, dict[str, Any]]:
    suffix = f"/{urllib.parse.quote(directory)}" if directory else ""
    errors = []
    for endpoint in endpoints:
        url = f"{endpoint}/api/models/{repo}/tree/main{suffix}"
        try:
            entries = _fetch_json(url, "HF_TOKEN")
        except (OSError, ValueError) as error:
            errors.append(f"{url}: {error}")
            continue
        return {
            str(entry["path"]): entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("type") == "file" and "path" in entry
        }
    raise InstallError(f"cannot list {repo}'s files:\n  " + "\n  ".join(errors))


def _model_parts(path: str) -> list[str]:
    match = _SPLIT.match(path)
    if match is None:
        return [path]
    count = int(match["count"])
    return [f"{match['prefix']}-{index:05d}-of-{count:05d}.gguf" for index in range(1, count + 1)]


def _fetch_json(url: str, token_variable: str) -> Any:
    headers = {"User-Agent": "hanafubuki-ext-llama-cpp", "Accept": "application/json"}
    token = os.environ.get(token_variable)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _clear_staging(staging: Path) -> None:
    # Downloads stay, so a retry resumes them; anything else is a run's leftover.
    staging.mkdir(exist_ok=True)
    for entry in staging.iterdir():
        if entry.name != "downloads":
            _remove(entry)


def _unpack(archive: Path, into: Path) -> Path:
    _say(f"Extracting {archive.name}")
    try:
        extract_archive(archive, into, progress=False)
    except (OSError, ValueError, RuntimeError) as error:
        raise InstallError(f"cannot extract {archive.name}: {error}") from error
    return into


def _server_dir(root: Path, program: str) -> Path:
    found = sorted(
        (path for path in root.rglob(program) if path.is_file()), key=lambda p: len(p.parts)
    )
    if not found:
        raise InstallError(f"the archive holds no {program}")
    return found[0].parent


def _single_top(root: Path) -> Path:
    entries = list(root.iterdir())
    if len(entries) == 1 and entries[0].is_dir() and not entries[0].is_symlink():
        return entries[0]
    return root


def _swap(core: Path, build: Path, previous: Path) -> None:
    """Moves the previous build aside and the new one in, restoring the old on failure."""
    previous.mkdir()
    kept = {STAGING, RECORD}
    moved_out = []
    for entry in list(core.iterdir()):
        if entry.name not in kept:
            entry.rename(previous / entry.name)
            moved_out.append(entry.name)
    moved_in = []
    try:
        for entry in list(build.iterdir()):
            entry.rename(core / entry.name)
            moved_in.append(entry.name)
    except OSError as error:
        for name in moved_in:
            _remove(core / name)
        for name in moved_out:
            (previous / name).rename(core / name)
        raise InstallError(f"cannot move the new build into {core}: {error}") from error


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _say(message: str) -> None:
    print(message, flush=True)


class _Progress:
    """Prints a download's progress every tenth of the way."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._shown = -1

    def __call__(self, event: Any) -> None:
        total = getattr(event, "total_size", None)
        done = getattr(event, "completed_size", None)
        if not isinstance(total, int) or not isinstance(done, int) or total <= 0:
            return
        tenth = min(done * 10 // total, 10)
        if tenth > self._shown:
            self._shown = tenth
            _say(f"  {self._name}: {tenth * 10}% of {total / 1048576:.1f} MiB")


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install llama.cpp into a Hanafubuki instance.")
    parser.add_argument("--core", required=True, help="the instance's core directory")
    parser.add_argument("--models", required=True, help="the instance's models directory")
    parser.add_argument("--version", default="latest", help="`latest` or a tag such as b12287")
    parser.add_argument("--variant", default="cpu")
    parser.add_argument("--force", action="store_true", help="reinstall an installed build")
    parser.add_argument("--github-mirror", choices=("auto", "always", "off"), default="auto")
    parser.add_argument("--github-api", default=GITHUB_API)
    parser.add_argument("--hf-repo", help="a Hugging Face repository, owner/name")
    parser.add_argument("--hf-file", help="the .gguf file's path in the repository")
    parser.add_argument("--hf-mirror", choices=("auto", "always", "off"), default="auto")
    parser.add_argument("--hf-endpoint", help="one Hugging Face endpoint instead of the mirrors")
    args = parser.parse_args(argv)
    if args.hf_repo and not args.hf_file:
        parser.error("--hf-repo needs --hf-file")
    return args


if __name__ == "__main__":
    sys.exit(main())
