"""The llama.cpp extension's backend.

It plans the ``io.github.licyk.llama-cpp:server`` type (the install step that
runs ``scripts/install.py``, the ``llama-server`` command), checks its values
for this platform, and answers the instance's status panel and the
application's update check (``provider.checkUpdate``). It never writes into
an instance: the host runs the plan's steps under the instance's permits.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hanafubuki_extension import INVALID_PARAMS, REQUEST_FAILED, Handler, Host, RpcError, run

from hanafubuki_llama import releases as rel
from hanafubuki_llama.record import list_models, read_record
from hanafubuki_llama.spec import escape, server_argv, validate

TYPE = "io.github.licyk.llama-cpp:server"
PANEL = "status"
MIRROR_MODES = ("auto", "always", "off")
MODELS_README = (
    "Put GGUF model files here. With the model source 'Models folder', the server's\n"
    "web page lists every .gguf file in this folder and loads the one you pick.\n"
)

_TEXT = {
    "en": {
        "version": "Installed version",
        "variant": "Build",
        "asset": "Archive",
        "installedAt": "Installed at",
        "latest": "Latest release",
        "status": "Status",
        "none": "Not installed",
        "checking": "Checking…",
        "unknown": "Unknown ({error})",
        "current": "Up to date",
        "available": "{tag} is available: run Update in Maintenance",
        "notice": "llama.cpp {installed} ({variant}) is installed; the latest release is {latest}.",
        "noticeNew": "llama.cpp {latest} is available ({variant}); {installed} is installed. "
        "Run Update to install it.",
        "noticeNone": "llama.cpp is not installed in this instance yet.",
    },
    "zh": {
        "version": "已安装版本",
        "variant": "构建",
        "asset": "压缩包",
        "installedAt": "安装时间",
        "latest": "最新版本",
        "status": "状态",
        "none": "未安装",
        "checking": "正在检查…",
        "unknown": "未知（{error}）",
        "current": "已是最新",
        "available": "{tag} 可用：在维护中运行更新",
        "notice": "已安装 llama.cpp {installed}（{variant}），最新版本为 {latest}。",
        "noticeNew": "llama.cpp {latest} 可用（{variant}），已安装 {installed}。运行更新即可安装。",
        "noticeNone": "此实例尚未安装 llama.cpp。",
    },
}


class Releases:
    """llama.cpp's recent releases, fetched through the host and cached in the data directory.

    The backend's environment carries no proxy, so the listing is fetched with
    ``download.fetch``, which goes through the application's. A panel never
    waits for it: a stale listing is refreshed on a thread and the panel shows
    what is cached meanwhile.
    """

    TTL_SECONDS = 15 * 60
    WAIT_SECONDS = 20

    def __init__(self, data_dir: Path) -> None:
        """Builds the cache.

        Args:
            data_dir: The extension's data directory.
        """
        self._dir = data_dir / "cache"
        self._lock = threading.Lock()
        self.error: str | None = None

    def get(self, host: Host, wait: bool) -> list[dict[str, Any]] | None:
        """The releases, refreshed when the cache is stale.

        Args:
            host: The application.
            wait: Wait for a refresh (an action the user pressed) rather than
                start one in the background.

        Returns:
            The cached releases, or ``None`` when none were fetched yet.

        Raises:
            RpcError: Waiting, GitHub did not answer within ``WAIT_SECONDS``.
        """
        if not self._fresh():
            refresh = threading.Thread(target=self._refresh, args=(host,), daemon=True)
            refresh.start()
            # Within the host's 30 s bound for the call that waits: a late
            # answer leaves the refresh running and says so.
            if wait:
                refresh.join(self.WAIT_SECONDS)
                if refresh.is_alive():
                    raise RpcError(REQUEST_FAILED, "GitHub has not answered yet; try again shortly")
        return self._read()

    def _refresh(self, host: Host) -> None:
        if not self._lock.acquire(blocking=False):
            # Another refresh runs: wait for it rather than fetch twice.
            with self._lock:
                return
        try:
            partial = f"cache/releases-{os.getpid()}-{threading.get_ident()}.json"
            result = host.call("download.fetch", {"url": rel.RELEASES_URL, "destination": partial})
            os.replace(result["path"], self._dir / "releases.json")
            self.error = None
        except (RpcError, OSError, KeyError, TypeError) as error:
            self.error = str(error)
            host.log("warn", f"cannot fetch llama.cpp's releases: {error}")
        finally:
            self._lock.release()

    def _fresh(self) -> bool:
        try:
            age = time.time() - (self._dir / "releases.json").stat().st_mtime
        except OSError:
            return False
        return age < self.TTL_SECONDS

    def _read(self) -> list[dict[str, Any]] | None:
        try:
            value = json.loads((self._dir / "releases.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return value if isinstance(value, list) else None


class Backend(Handler):
    """The type's provider, its status panel and its update check."""

    def __init__(self) -> None:
        """Builds the backend; the data directory is known once the host initializes it."""
        self.releases: Releases | None = None

    def initialize(self, host: Host, params: dict[str, Any]) -> dict[str, Any]:
        """Keeps the data directory, then answers as the SDK does.

        Args:
            host: The application.
            params: What the host sent.

        Returns:
            The answer.
        """
        data = params.get("dataDir") or os.environ.get("HANAFUBUKI_EXTENSION_DATA", ".")
        self.releases = Releases(Path(str(data)))
        return super().initialize(host, params)

    def validate_spec(
        self,
        host: Host,
        type_id: str,
        values: dict[str, Any],
        instance: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Checks the values against what llama.cpp builds for this platform.

        Args:
            host: The application.
            type_id: The type.
            values: The install parameters.
            instance: The instance whose values change, when it exists; not read.

        Returns:
            Problems by parameter.
        """
        return validate(values, *rel.host_platform())

    def install_plan(
        self, host: Host, type_id: str, kind: str, values: dict[str, Any], instance: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """One ``run`` step: ``scripts/install.py`` on the managed Python.

        The release is resolved by the script, not here: this call has 30 s.

        Args:
            host: The application.
            type_id: The type.
            kind: ``install``, ``reinstall`` or ``update``.
            values: The install parameters.
            instance: The instance.

        Returns:
            The steps.
        """
        settings = host.call("settings.get").get("values", {})
        command = [
            "{host.python}",
            "{extension.root}/scripts/install.py",
            "--core",
            "{component.core}",
            "--models",
            "{component.models}",
            "--version",
            escape(str(values.get("version") or "latest")),
            "--variant",
            escape(str(values.get("variant") or "cpu")),
            "--github-mirror",
            _mirror_mode(settings.get("githubMirror")),
            "--hf-mirror",
            _mirror_mode(settings.get("huggingfaceMirror")),
        ]
        if values.get("modelSource") == "huggingface":
            command += [
                "--hf-repo",
                escape(str(values.get("hfRepo", ""))),
                "--hf-file",
                escape(str(values.get("hfFile", ""))),
            ]
        if kind == "reinstall":
            command.append("--force")
        return [
            # Creates the models folder, which the run step starts in.
            {
                "kind": "write-file",
                "path": "{component.models}/README.txt",
                "content": MODELS_README,
            },
            {"kind": "run", "command": command, "cwd": "{component.models}"},
        ]

    def launch_plan(
        self,
        host: Host,
        type_id: str,
        role: str,
        port: int,
        values: dict[str, Any],
        instance: dict[str, Any],
    ) -> dict[str, Any]:
        """The ``ui`` role's ``llama-server`` command.

        Args:
            host: The application.
            type_id: The type.
            role: The role; only ``ui`` is declared.
            port: Its port, which the command names as ``{port}``.
            values: The install parameters.
            instance: The instance.

        Returns:
            The argv and environment.

        Raises:
            RpcError: An undeclared role.
        """
        if role != "ui":
            raise RpcError(INVALID_PARAMS, f"the type has no server role `{role}`")
        return {"argv": server_argv(values, rel.host_platform()[0]), "env": {}}

    def check_update(
        self, host: Host, type_id: str, values: dict[str, Any], instance: dict[str, Any]
    ) -> dict[str, Any]:
        """``provider.checkUpdate``: the installed build and the one Update would install.

        Update installs the ``version`` value's release (``latest``: the
        newest that holds the build), so an instance pinned to the build it
        has has none.

        Args:
            host: The application.
            type_id: The type.
            values: The install parameters.
            instance: The instance.

        Returns:
            ``current``, ``latest`` and ``hasUpdate``.

        Raises:
            RpcError: Nothing is installed, or the releases cannot be read.
        """
        record = read_record(_component(instance, "core"))
        if record is None:
            raise RpcError(REQUEST_FAILED, _strings(host)["noticeNone"])
        wanted = str(values.get("version") or "latest")
        variant = str(values.get("variant") or record.get("variant") or "cpu")
        target = self._release(host, variant, wanted, wait=True)
        if target is None:
            raise RpcError(REQUEST_FAILED, f"cannot read llama.cpp's releases: {self._error()}")
        installed = str(record.get("tag", ""))
        changes = _newer(target, installed) if wanted == "latest" else target != installed
        return {"current": installed, "latest": target, "hasUpdate": changes}

    def panel_query(
        self, host: Host, panel: str, block: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """The status panel's blocks: the installed build, and the models.

        Args:
            host: The application.
            panel: The panel.
            block: ``install`` or ``models``.
            context: The instance.

        Returns:
            The block's rows.

        Raises:
            RpcError: An unknown panel or block.
        """
        if panel != PANEL:
            raise RpcError(INVALID_PARAMS, f"no panel `{panel}`")
        instance = self._instance(host, str(context.get("instanceId", "")))
        if block == "install":
            return {"rows": self._install_rows(host, _component(instance, "core"))}
        if block == "models":
            return {
                "rows": [
                    {
                        "name": model.name,
                        "size": _size(model.size),
                        "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(model.modified)),
                    }
                    for model in list_models(_component(instance, "models"))
                ]
            }
        raise RpcError(INVALID_PARAMS, f"no block `{block}`")

    def panel_action(
        self,
        host: Host,
        panel: str,
        action: str,
        values: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        """``check``: fetches the releases now and says whether an update exists.

        Args:
            host: The application.
            panel: The panel.
            action: The action.
            values: Its form's values (none).
            context: The instance.

        Returns:
            The notice.

        Raises:
            RpcError: An unknown panel or action.
        """
        if panel != PANEL or action != "check":
            raise RpcError(INVALID_PARAMS, f"no action `{action}` in `{panel}`")
        instance = self._instance(host, str(context.get("instanceId", "")))
        return self._update_notice(host, _component(instance, "core"))

    def _update_notice(self, host: Host, core: Path) -> str:
        text = _strings(host)
        record = read_record(core)
        if record is None:
            return text["noticeNone"]
        latest = self._release(host, str(record.get("variant")), "latest", wait=True)
        if latest is None:
            raise RpcError(REQUEST_FAILED, f"cannot read llama.cpp's releases: {self._error()}")
        key = "noticeNew" if _newer(latest, record.get("tag")) else "notice"
        return text[key].format(
            installed=record.get("tag"), variant=record.get("variant"), latest=latest
        )

    def _install_rows(self, host: Host, core: Path) -> list[dict[str, str]]:
        text = _strings(host)
        record = read_record(core)
        if record is None:
            return [{"label": text["version"], "value": text["none"]}]
        latest = self._release(host, str(record.get("variant")), "latest", wait=False)
        if latest is not None:
            status = (
                text["available"].format(tag=latest)
                if _newer(latest, record.get("tag"))
                else text["current"]
            )
        elif self.releases is not None and self.releases.error:
            status = text["unknown"].format(error=self.releases.error)
        else:
            status = text["checking"]
        return [
            {"label": text["version"], "value": str(record.get("tag", ""))},
            {"label": text["variant"], "value": str(record.get("variant", ""))},
            {"label": text["asset"], "value": str(record.get("asset", ""))},
            {"label": text["installedAt"], "value": str(record.get("installedAt", ""))},
            {"label": text["latest"], "value": latest or "—"},
            {"label": text["status"], "value": status},
        ]

    def _release(self, host: Host, variant: str, version: str, wait: bool) -> str | None:
        found = self.releases.get(host, wait) if self.releases else None
        if found is None:
            return None
        os_name, arch = rel.host_platform()
        try:
            return rel.select(found, os_name, arch, variant, version).tag
        except LookupError:
            return None

    def _error(self) -> str | None:
        return self.releases.error if self.releases else None

    def _instance(self, host: Host, instance_id: str) -> dict[str, Any]:
        # A panel's context names the instance only; `instances.ofType` lists
        # it with its components (Hanafubuki 0.3.31). `instances.get` would
        # need `instances = read`, which `own-types` is not.
        for instance in host.call("instances.ofType", {"type": TYPE}).get("instances", []):
            if instance.get("id") == instance_id:
                return instance
        raise RpcError(INVALID_PARAMS, f"no instance `{instance_id}` of {TYPE}")


def _component(instance: dict[str, Any], name: str) -> Path:
    """Where the host says the component ``name`` of a provider call's or a listed instance is.

    Raises:
        RpcError: The instance names no such component.
    """
    found = (instance.get("components") or {}).get(name)
    if not isinstance(found, dict) or not isinstance(found.get("path"), str) or not found["path"]:
        raise RpcError(INVALID_PARAMS, f"the instance has no `{name}` component")
    return Path(found["path"])


def _newer(latest: str, installed: Any) -> bool:
    try:
        return int(latest.lstrip("b")) > int(str(installed).lstrip("b"))
    except ValueError:
        return False


def _mirror_mode(value: Any) -> str:
    return value if value in MIRROR_MODES else "auto"


def _strings(host: Host) -> dict[str, str]:
    locale = str(host.initialize_params.get("locale", "auto"))
    if locale == "auto":
        locale = "zh" if os.environ.get("LANG", "").startswith("zh") else "en"
    return _TEXT["zh" if locale == "zh" else "en"]


def _size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    value = float(size)
    for unit in ("KiB", "MiB", "GiB"):
        value /= 1024
        if value < 1024 or unit == "GiB":
            break
    return f"{value:.1f} {unit}"


if __name__ == "__main__":
    run(Backend())
