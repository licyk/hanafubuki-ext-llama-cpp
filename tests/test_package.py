"""The package as Hanafubuki reads it: the manifest, what it contributes, and the archive."""

from __future__ import annotations

import json
import re
import struct
import sys
import zipfile
from pathlib import Path

import pytest
from support import ROOT

if sys.version_info < (3, 11):
    pytest.skip("tomllib needs Python 3.11", allow_module_level=True)
import tomllib

sys.path.insert(0, str(ROOT / "scripts"))
import package

PREFIX = "ext.io.github.licyk.llama-cpp."


def manifest() -> dict:
    return tomllib.loads((ROOT / "hanafubuki-extension.toml").read_text(encoding="utf-8"))


def descriptor() -> dict:
    return tomllib.loads((ROOT / "types" / "server.toml").read_text(encoding="utf-8"))


def flatten(prefix: str, value: object) -> set[str]:
    if isinstance(value, dict):
        return {key for name, inner in value.items() for key in flatten(f"{prefix}{name}.", inner)}
    return {prefix.rstrip(".")}


def message_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        # `{ kind = "key", key }` in JSON forms, `{ key }` in a descriptor.
        text = value.get("kind") == "key" or set(value) <= {"key", "params"}
        found = {value["key"]} if text and isinstance(value.get("key"), str) else set()
        return found.union(*(message_keys(inner) for inner in value.values()))
    if isinstance(value, list):
        return set().union(*(message_keys(inner) for inner in value))
    return set()


def test_the_manifest():
    data = manifest()
    assert data["id"] == "io.github.licyk.llama-cpp"
    assert re.match(r"^\d+\.\d+\.\d+$", data["version"])
    assert data["engines"] == {"hanafubuki": ">=0.3.28, <0.4", "api": 1}
    assert data["backend"] == {"python": "backend/main.py", "sdWebuiAllInOne": True}
    assert data["permissions"]["instances"] == "own-types"
    contributes = data["contributes"]
    for path in (data["backend"]["python"], contributes["settings"]):
        assert (ROOT / path).is_file()
    for entry in contributes["ui"]:
        assert (ROOT / entry["source"]).is_file()
    for locale in ("en", "zh"):
        assert (ROOT / contributes["locales"].replace("{locale}", locale)).is_file()
    assert list(ROOT.glob(contributes["instanceTypes"])) == [ROOT / "types" / "server.toml"]


def test_the_type():
    data = descriptor()
    assert data["name"] == "server"
    assert data["provider"] == "backend"
    assert "steps" not in data.get("install", {})
    assert [component["name"] for component in data["components"]] == ["core", "models"]
    assert data["management"]["operations"] == ["update", "terminal", "check-update"]
    (server,) = data["servers"]
    assert server["readiness"] == {"http": "/health", "timeoutSeconds": 600}
    assert "command" not in server
    names = {parameter["name"] for parameter in data["install"]["parameters"]}
    for signature in data["discovery"]:
        assert set(signature["values"]) <= names
        assert set(signature["components"]) == {"core", "models"}
    variants = next(p for p in data["install"]["parameters"] if p["name"] == "variant")
    from hanafubuki_llama.releases import VARIANTS

    assert [option["value"] for option in variants["options"]] == list(VARIANTS)


def test_the_preview_is_a_16_by_9_png():
    path = ROOT / descriptor()["preview"]
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", header[16:24])
    assert width * 9 == height * 16


def test_every_message_key_is_in_both_locales():
    used = set()
    used |= message_keys(descriptor())
    used |= message_keys(json.loads((ROOT / "schemas" / "settings.json").read_text("utf-8")))
    used |= message_keys(json.loads((ROOT / "ui" / "status.json").read_text("utf-8")))
    assert used
    assert all(key.startswith(PREFIX) for key in used)
    strings = {
        locale: flatten(
            PREFIX, json.loads((ROOT / "locales" / f"{locale}.json").read_text("utf-8"))
        )
        for locale in ("en", "zh")
    }
    assert strings["en"] == strings["zh"]
    assert used <= strings["en"], used - strings["en"]


def test_the_repository_holds_no_links():
    """A development link of the repository refuses any link inside it."""
    links = [path for path in ROOT.rglob("*") if path.is_symlink()]
    assert links == []


def test_the_archive_holds_the_package_only(tmp_path):
    archive = package.build(tmp_path)
    assert archive.name == f"io.github.licyk.llama-cpp-{manifest()['version']}.zip"
    with zipfile.ZipFile(archive) as opened:
        names = set(opened.namelist())
    for required in (
        "hanafubuki-extension.toml",
        "backend/main.py",
        "scripts/install.py",
        "hanafubuki_llama/releases.py",
        "types/server.toml",
        "images/preview.png",
        "ui/status.json",
        "locales/zh.json",
    ):
        assert required in names
    for name in names:
        assert not name.startswith(("tests/", "tools/", ".git", "dist/"))
        assert "__pycache__" not in name
    assert "scripts/package.py" not in names
    assert "pyproject.toml" not in names


def test_package_files_are_the_repository_files_that_ship():
    files = package.package_files()
    assert all((ROOT / name).is_file() for name in files)
    assert files == sorted(files)
    assert Path(files[0]).parts
