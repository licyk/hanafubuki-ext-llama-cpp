"""backend/main.py over its protocol, spawned and answered as the host does."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest
from hanafubuki_extension.protocol import read_frame, write_frame
from support import PYTHON, ROOT, SDK, environment, known

from hanafubuki_llama.record import write_record
from hanafubuki_llama.releases import RELEASES_URL, host_platform

TYPE = "io.github.licyk.llama-cpp:server"
FIXTURE = Path(__file__).parent / "fixtures" / "releases.json"


class Host:
    """The application's side: calls the backend and answers its calls."""

    def __init__(self, data_dir: Path, answers: dict[str, Any]) -> None:
        self.process = subprocess.Popen(
            [PYTHON, str(ROOT / "backend" / "main.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=ROOT,
            env={**environment(SDK), "HANAFUBUKI_EXTENSION_DATA": str(data_dir)},
        )
        self.stdin = known(self.process.stdin)
        self.stdout = known(self.process.stdout)
        self.stderr = known(self.process.stderr)
        self.answers = answers
        self.calls: list[tuple[str, Any]] = []
        self._next = 1
        self._lock = threading.Lock()

    def call(self, method: str, params: Any = None) -> dict[str, Any]:
        """Sends a request and returns its response, answering the backend meanwhile."""
        with self._lock:
            call_id = self._next
            self._next += 1
            write_frame(
                self.stdin,
                {"jsonrpc": "2.0", "id": call_id, "method": method, "params": params or {}},
            )
            while True:
                message = read_frame(self.stdout)
                assert message is not None, self.stderr.read().decode()
                if "method" not in message:
                    assert message["id"] == call_id
                    return message
                self.calls.append((message["method"], message.get("params")))
                if "id" in message:
                    write_frame(self.stdin, self._answer(message))

    def _answer(self, message: dict[str, Any]) -> dict[str, Any]:
        answer = self.answers.get(message["method"])
        if answer is None:
            return {
                "jsonrpc": "2.0",
                "id": message["id"],
                "error": {"code": -32601, "message": f"no method {message['method']}"},
            }
        result = answer(message.get("params") or {}) if callable(answer) else answer
        return {"jsonrpc": "2.0", "id": message["id"], "result": result}

    def close(self) -> int:
        self.call("shutdown")
        self.stdin.close()
        code = self.process.wait(timeout=10)
        self.stdout.close()
        self.stderr.close()
        return code


@pytest.fixture
def instance(tmp_path):
    root = tmp_path / "instance"
    (root / "core").mkdir(parents=True)
    (root / "models" / "sub").mkdir(parents=True)
    (root / "models" / "a.gguf").write_bytes(b"x" * 3000)
    (root / "models" / "sub" / "b.gguf").write_bytes(b"y")
    (root / "models" / "README.txt").write_text("not a model")
    return root


@pytest.fixture
def host(tmp_path, instance):
    data = tmp_path / "data"
    data.mkdir()

    def fetch(params):
        assert params["url"] == RELEASES_URL
        target = data / params["destination"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FIXTURE, target)
        return {"path": str(target), "sha256": ""}

    host = Host(
        data,
        {
            "settings.get": {"values": {"githubMirror": "always"}},
            "instances.ofType": {
                "instances": [
                    {
                        "id": "i1",
                        "name": "llama",
                        "type": TYPE,
                        "path": str(instance),
                        "running": False,
                    }
                ]
            },
            "download.fetch": fetch,
        },
    )
    initialized = host.call(
        "initialize",
        {
            "protocolVersion": 1,
            "extension": {"id": "io.github.licyk.llama-cpp", "version": "0.1.0"},
            "dataDir": str(data),
            "locale": "en",
        },
    )
    assert initialized["result"]["protocolVersion"] == 1
    yield host
    assert host.close() == 0


def test_ping_and_hooks(host):
    assert host.call("ping")["result"] == {}


def test_validate_spec(host):
    good = host.call("provider.validateSpec", {"type": TYPE, "values": {"variant": "cpu"}})
    assert good["result"] == {"issues": {}}
    bad = host.call(
        "provider.validateSpec", {"type": TYPE, "values": {"variant": "cpu", "version": "x"}}
    )
    assert list(bad["result"]["issues"]) == ["version"]


def test_install_plan_runs_the_script_on_the_managed_python(host):
    values = {
        "version": "b12287",
        "variant": "vulkan",
        "modelSource": "huggingface",
        "hfRepo": "ggml-org/gemma-3-1b-it-GGUF",
        "hfFile": "gemma-3-1b-it-Q4_K_M.gguf",
    }
    response = host.call(
        "provider.installPlan",
        {"type": TYPE, "kind": "reinstall", "values": values, "instance": {"id": "i1"}},
    )
    write, run = response["result"]["steps"]
    assert write["kind"] == "write-file"
    assert write["path"] == "{component.models}/README.txt"
    assert run["kind"] == "run"
    assert run["cwd"] == "{component.models}"
    assert run["command"] == [
        "{host.python}",
        "{extension.root}/scripts/install.py",
        "--core",
        "{component.core}",
        "--models",
        "{component.models}",
        "--version",
        "b12287",
        "--variant",
        "vulkan",
        "--github-mirror",
        "always",
        "--hf-mirror",
        "auto",
        "--hf-repo",
        "ggml-org/gemma-3-1b-it-GGUF",
        "--hf-file",
        "gemma-3-1b-it-Q4_K_M.gguf",
        "--force",
    ]


def test_launch_plan(host):
    response = host.call(
        "provider.launchPlan",
        {
            "type": TYPE,
            "role": "ui",
            "port": 4321,
            "values": {"modelSource": "directory", "contextSize": 8192, "extraArgs": "--jinja"},
            "instance": {"id": "i1"},
        },
    )
    program = "llama-server.exe" if host_platform()[0] == "windows" else "llama-server"
    assert response["result"] == {
        "argv": [
            "{component.core}/" + program,
            "--host",
            "127.0.0.1",
            "--port",
            "{port}",
            "--models-dir",
            "{component.models}",
            "--ctx-size",
            "8192",
            "--jinja",
        ],
        "env": {},
    }
    unknown = host.call(
        "provider.launchPlan", {"type": TYPE, "role": "api", "values": {}, "instance": {}}
    )
    assert unknown["error"]["code"] == -32602


def test_the_status_panel_and_the_update_check(host, instance):
    rows = host.call(
        "ui.panel.query", {"panel": "status", "block": "install", "context": {"instanceId": "i1"}}
    )["result"]["rows"]
    assert rows == [{"label": "Installed version", "value": "Not installed"}]

    os_name, arch = host_platform()
    write_record(
        instance / "core",
        {
            "tag": "b12284",
            "variant": "cpu",
            "asset": "llama-b12284-bin.tar.gz",
            "installedAt": "now",
        },
    )
    notice = host.call(
        "ui.panel.action",
        {"panel": "status", "action": "check", "values": {}, "context": {"instanceId": "i1"}},
    )["result"]["notice"]
    if (os_name, arch) == ("linux", "x64"):
        assert "b12289 is available" in notice
        rows = host.call(
            "ui.panel.query",
            {"panel": "status", "block": "install", "context": {"instanceId": "i1"}},
        )["result"]["rows"]
        assert {"label": "Latest release", "value": "b12289"} in rows
        assert {
            "label": "Status",
            "value": "b12289 is available: run Update in Maintenance",
        } in rows
        operation = host.call(
            "provider.manage.check-update",
            {"type": TYPE, "operation": "check-update", "instance": {"id": "i1"}},
        )
        assert operation["result"]["notice"] == notice
    fetches = [call for call in host.calls if call[0] == "download.fetch"]
    assert len(fetches) == 1

    models = host.call(
        "ui.panel.query", {"panel": "status", "block": "models", "context": {"instanceId": "i1"}}
    )["result"]["rows"]
    assert [(row["name"], row["size"]) for row in models] == [
        ("a.gguf", "2.9 KiB"),
        ("sub/b.gguf", "1 B"),
    ]


def test_an_instance_of_another_extension_is_refused(host):
    response = host.call(
        "ui.panel.query", {"panel": "status", "block": "install", "context": {"instanceId": "x"}}
    )
    assert response["error"]["code"] == -32602


def test_the_releases_cache_is_written_in_the_data_directory(host, tmp_path, instance):
    write_record(instance / "core", {"tag": "b12289", "variant": "cpu"})
    host.call(
        "ui.panel.action",
        {"panel": "status", "action": "check", "values": {}, "context": {"instanceId": "i1"}},
    )
    cached = json.loads((tmp_path / "data" / "cache" / "releases.json").read_text())
    assert cached[0]["tag_name"] == "b12289"
