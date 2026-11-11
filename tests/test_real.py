"""The real thing: install the newest CPU build from GitHub and start its server.

Downloads about 25 MB. Runs only with ``LLAMA_CPP_REAL_TEST=1``.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
import urllib.request

import pytest
from support import PYTHON, ROOT, SD_WEBUI_ALL_IN_ONE, environment

from hanafubuki_llama.record import read_record
from hanafubuki_llama.releases import host_platform
from hanafubuki_llama.spec import server_argv

pytestmark = pytest.mark.skipif(
    os.environ.get("LLAMA_CPP_REAL_TEST") != "1", reason="set LLAMA_CPP_REAL_TEST=1"
)


def test_install_the_latest_build_and_serve_a_folder_of_models(tmp_path):
    core, models = tmp_path / "core", tmp_path / "models"
    env = {**environment(SD_WEBUI_ALL_IN_ONE)}
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "NO_PROXY"):
        if name in os.environ:
            env[name] = os.environ[name]
        else:
            env.pop(name, None)
    result = subprocess.run(
        [
            PYTHON,
            str(ROOT / "scripts" / "install.py"),
            "--core",
            str(core),
            "--models",
            str(models),
            "--github-mirror",
            "auto",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = read_record(core)
    assert record is not None and record["tag"].startswith("b")

    os_name, _ = host_platform()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    argv = [
        arg.replace("{component.core}", str(core))
        .replace("{component.models}", str(models))
        .replace("{port}", str(port))
        for arg in server_argv({"modelSource": "directory", "contextSize": 2048}, os_name)
    ]
    version = subprocess.run([argv[0], "--version"], capture_output=True, text=True, timeout=60)
    assert record["tag"].removeprefix("b") in version.stdout + version.stderr
    server = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as answer:
                    assert answer.status == 200
                    break
            except OSError:
                assert time.monotonic() < deadline, "the server did not answer /health"
                time.sleep(0.5)
    finally:
        server.terminate()
        server.wait(timeout=10)
