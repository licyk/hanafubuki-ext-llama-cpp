"""scripts/install.py against releases served from a local HTTP server.

The script runs as the host runs it, in a process of its own, with the real
sd_webui_all_in_one on its import path; the server stands in for GitHub's API,
its downloads and Hugging Face.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
import tarfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from support import PYTHON, ROOT, SD_WEBUI_ALL_IN_ONE, environment, known

from hanafubuki_llama.record import RECORD, read_record
from hanafubuki_llama.releases import host_platform

OS_NAME, ARCH = host_platform()
PROGRAM = "llama-server.exe" if OS_NAME == "windows" else "llama-server"
PLATFORM = {"linux": "ubuntu", "windows": "win-cpu", "macos": "macos"}[OS_NAME]

pytestmark = pytest.mark.skipif(
    not (SD_WEBUI_ALL_IN_ONE / "sd_webui_all_in_one").is_dir(),
    reason="needs a checkout of sd-webui-all-in-one (SD_WEBUI_ALL_IN_ONE)",
)


class Site:
    """Files and JSON documents served at fixed paths."""

    def __init__(self) -> None:
        self.routes: dict[str, bytes] = {}
        self.requests: list[str] = []
        site = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                site.requests.append(path)
                body = site.routes.get(path)
                if body is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_HEAD(self) -> None:
                body = site.routes.get(self.path.split("?", 1)[0])
                self.send_response(200 if body is not None else 404)
                self.send_header("Content-Length", str(len(body or b"")))
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def release(self, tag: str, assets: dict[str, bytes], digests: bool = True) -> dict:
        listed = []
        for name, data in assets.items():
            path = f"/download/{tag}/{name}"
            self.routes[path] = data
            listed.append(
                {
                    "name": name,
                    "size": len(data),
                    "digest": f"sha256:{hashlib.sha256(data).hexdigest()}" if digests else None,
                    "browser_download_url": self.url + path,
                }
            )
        release = {"tag_name": tag, "draft": False, "prerelease": True, "assets": listed}
        self.routes[f"/repos/ggml-org/llama.cpp/releases/tags/{tag}"] = json.dumps(release).encode()
        return release

    def publish(self, *releases: dict) -> None:
        self.routes["/repos/ggml-org/llama.cpp/releases"] = json.dumps(list(releases)).encode()


@pytest.fixture
def site():
    site = Site()
    yield site
    site.server.shutdown()


def tar_build(top: str | None, files: dict[str, bytes], links: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        prefix = f"{top}/" if top else ""
        if top:
            info = tarfile.TarInfo(top)
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            archive.addfile(info)
        for name, data in files.items():
            info = tarfile.TarInfo(prefix + name)
            info.size = len(data)
            info.mode = 0o755
            archive.addfile(info, io.BytesIO(data))
        for name, target in links.items():
            info = tarfile.TarInfo(prefix + name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            archive.addfile(info)
    return buffer.getvalue()


def zip_build(top: str | None, files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            info = zipfile.ZipInfo((f"{top}/" if top else "") + name)
            info.external_attr = (stat.S_IFREG | 0o755) << 16
            archive.writestr(info, data)
    return buffer.getvalue()


def server_program(tag: str) -> bytes:
    return f"#!/bin/sh\necho 'version: {tag}'\n".encode()


def install(
    site: Site, instance: Path, *args: str, python: tuple[str, ...] = (PYTHON,)
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            *python,
            str(ROOT / "scripts" / "install.py"),
            "--core",
            str(instance / "core"),
            "--models",
            str(instance / "models"),
            "--github-api",
            site.url,
            "--github-mirror",
            "off",
            *args,
        ],
        capture_output=True,
        text=True,
        env=environment(SD_WEBUI_ALL_IN_ONE),
        timeout=120,
        check=False,
    )


def assert_ok(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, result.stdout + result.stderr


def publish_two(site: Site) -> None:
    older = site.release(
        "b100",
        {
            f"llama-b100-bin-{PLATFORM}-{ARCH}.tar.gz": tar_build(
                "llama-b100",
                {PROGRAM: server_program("b100"), "libllama.so.0.1": b"lib", "only-b100": b"x"},
                {"libllama.so": "libllama.so.0.1"},
            )
        },
    )
    newer = site.release(
        "b101",
        {
            f"llama-b101-bin-{PLATFORM}-{ARCH}.zip": zip_build(
                None, {PROGRAM: server_program("b101")}
            )
        },
    )
    site.publish(older, newer)


def test_a_named_release_with_a_top_directory_is_installed_flat(site, tmp_path):
    publish_two(site)
    result = install(site, tmp_path, "--version", "b100")
    assert_ok(result)
    core = tmp_path / "core"
    assert (core / PROGRAM).read_bytes() == server_program("b100")
    assert os.access(core / PROGRAM, os.X_OK)
    if OS_NAME != "windows":
        assert os.readlink(core / "libllama.so") == "libllama.so.0.1"
    record = known(read_record(core))
    assert record["tag"] == "b100"
    assert record["variant"] == "cpu"
    assert record["asset"] == f"llama-b100-bin-{PLATFORM}-{ARCH}.tar.gz"
    assert not (core / ".staging").exists()
    assert (tmp_path / "models").is_dir()


def test_latest_replaces_the_previous_build(site, tmp_path):
    publish_two(site)
    assert_ok(install(site, tmp_path, "--version", "b100"))
    result = install(site, tmp_path, "--version", "latest")
    assert_ok(result)
    core = tmp_path / "core"
    assert (core / PROGRAM).read_bytes() == server_program("b101")
    assert os.access(core / PROGRAM, os.X_OK)
    assert sorted(path.name for path in core.iterdir()) == sorted([PROGRAM, RECORD])
    assert known(read_record(core))["tag"] == "b101"


def test_an_installed_build_is_not_downloaded_again(site, tmp_path):
    publish_two(site)
    assert_ok(install(site, tmp_path))
    site.requests.clear()
    result = install(site, tmp_path)
    assert_ok(result)
    assert "installed already" in result.stdout
    assert not any(path.startswith("/download/") for path in site.requests)
    site.requests.clear()
    assert_ok(install(site, tmp_path, "--force"))
    assert any(path.startswith("/download/") for path in site.requests)


def test_a_download_that_does_not_match_its_digest_changes_nothing(site, tmp_path):
    publish_two(site)
    assert_ok(install(site, tmp_path, "--version", "b100"))
    name = f"llama-b101-bin-{PLATFORM}-{ARCH}.zip"
    site.routes[f"/download/b101/{name}"] = b"not the archive"
    result = install(site, tmp_path, "--version", "latest")
    assert result.returncode == 1
    assert "sha256" in result.stderr.lower()
    core = tmp_path / "core"
    assert (core / PROGRAM).read_bytes() == server_program("b100")
    assert known(read_record(core))["tag"] == "b100"


def test_without_requests_the_urllib_downloader_is_used(site, tmp_path):
    """The managed Python may lack `requests`; the library must not pip-install it."""
    publish_two(site)
    # `-S`: no site-packages, so no `requests`.
    result = install(site, tmp_path, python=(PYTHON, "-S"))
    assert_ok(result)
    assert known(read_record(tmp_path / "core"))["tag"] == "b101"
    # What the library logs before it would pip-install `requests`.
    assert "未安装 requests" not in result.stdout + result.stderr


def test_an_unknown_release_fails_with_why(site, tmp_path):
    publish_two(site)
    result = install(site, tmp_path, "--version", "b999")
    assert result.returncode == 1
    assert "b999" in result.stderr


@pytest.mark.skipif((OS_NAME, ARCH) != ("linux", "x64"), reason="a Linux x64 CUDA build")
def test_a_cuda_build_gets_its_runtime(site, tmp_path):
    release = site.release(
        "b102",
        {
            "llama-b102-bin-ubuntu-cuda-12.8-x64.tar.gz": tar_build(
                "llama-b102", {PROGRAM: server_program("b102"), "libggml-cuda.so": b"c"}, {}
            ),
            "cudart-llama-b102-bin-ubuntu-cuda-12.8-x64.tar.gz": tar_build(
                "cudart", {"libcudart.so.12": b"rt", "libcublas.so.12": b"bl"}, {}
            ),
        },
    )
    site.publish(release)
    assert_ok(install(site, tmp_path, "--variant", "cuda-12"))
    core = tmp_path / "core"
    assert {path.name for path in core.iterdir()} == {
        PROGRAM,
        "libggml-cuda.so",
        "libcudart.so.12",
        "libcublas.so.12",
        RECORD,
    }
    record = known(read_record(core))
    assert record["runtime"] == "cudart-llama-b102-bin-ubuntu-cuda-12.8-x64.tar.gz"


def test_a_hugging_face_model_is_downloaded_once(site, tmp_path):
    publish_two(site)
    model = b"GGUF" + b"\0" * 2048
    site.routes["/api/models/owner/repo/tree/main/Q4"] = json.dumps(
        [
            {
                "type": "file",
                "path": "Q4/tiny.gguf",
                "size": 134,
                "lfs": {"oid": hashlib.sha256(model).hexdigest(), "size": len(model)},
            },
            {"type": "directory", "path": "Q4/sub"},
        ]
    ).encode()
    site.routes["/owner/repo/resolve/main/Q4/tiny.gguf"] = model
    args = ("--hf-repo", "owner/repo", "--hf-file", "Q4/tiny.gguf", "--hf-endpoint", site.url)
    assert_ok(install(site, tmp_path, *args))
    assert (tmp_path / "models" / "tiny.gguf").read_bytes() == model
    site.requests.clear()
    result = install(site, tmp_path, *args)
    assert_ok(result)
    assert "/owner/repo/resolve/main/Q4/tiny.gguf" not in site.requests
    missing = install(
        site,
        tmp_path,
        "--hf-repo",
        "owner/repo",
        "--hf-file",
        "Q4/other.gguf",
        "--hf-endpoint",
        site.url,
    )
    assert missing.returncode == 1
    assert "has no file Q4/other.gguf" in missing.stderr
