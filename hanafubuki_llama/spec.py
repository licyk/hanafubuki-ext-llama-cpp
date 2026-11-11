"""An instance's install parameters: what the backend checks, and the server command they make.

The host reads the values against the type's form first (kinds, bounds,
options, required fields), so this checks only what the form cannot say.
"""

from __future__ import annotations

import posixpath
import re
import shlex
from typing import Any

from .releases import supported_variants

_VERSION = re.compile(r"^(latest|b\d+)$")
_HF_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
# Set by the host, so the extra arguments may not set them again.
_RESERVED = ("--host", "--port")

DEFAULTS: dict[str, Any] = {
    "version": "latest",
    "variant": "cpu",
    "modelSource": "directory",
}


def validate(values: dict[str, Any], os_name: str, arch: str) -> dict[str, str]:
    """What is wrong with an instance's values on this platform.

    Args:
        values: The install parameters, after the host's form checks.
        os_name: The platform's operating system.
        arch: Its architecture.

    Returns:
        Problems by parameter; empty when the values are good.
    """
    issues: dict[str, str] = {}
    version = _text(values, "version")
    if not _VERSION.match(version):
        issues["version"] = "use `latest` or a release tag such as `b12287`"
    variant = _text(values, "variant")
    available = supported_variants(os_name, arch)
    if not available:
        issues["variant"] = f"llama.cpp publishes no builds for {os_name}-{arch}"
    elif variant not in available:
        issues["variant"] = (
            f"no {variant} build for {os_name}-{arch}; choose one of: {', '.join(available)}"
        )
    source = _text(values, "modelSource")
    if source == "file" and not _text(values, "model").lower().endswith(".gguf"):
        issues["model"] = "choose a .gguf file"
    if source == "huggingface":
        if not _HF_REPO.match(_text(values, "hfRepo")):
            issues["hfRepo"] = "write the repository as `owner/name`"
        hf_file = _text(values, "hfFile")
        if not _hf_file_ok(hf_file):
            issues["hfFile"] = "write the .gguf file's path in the repository"
    try:
        extra = split_args(_text(values, "extraArgs"), os_name)
    except ValueError as error:
        issues["extraArgs"] = f"cannot be split into arguments: {error}"
    else:
        reserved = [arg for arg in extra if arg.split("=", 1)[0] in _RESERVED]
        if reserved:
            issues["extraArgs"] = f"{', '.join(_RESERVED)} are set by Hanafubuki"
    return issues


def server_argv(values: dict[str, Any], os_name: str) -> list[str]:
    """The ``ui`` role's command, as a plan template.

    Placeholders (``{component.core}``, ``{port}``) are the host's to fill;
    every value from the user has its braces doubled so it stays literal.

    Args:
        values: The install parameters.
        os_name: The platform's operating system.

    Returns:
        The argv.
    """
    program = "{component.core}/llama-server" + (".exe" if os_name == "windows" else "")
    argv = [program, "--host", "127.0.0.1", "--port", "{port}"]
    source = _text(values, "modelSource")
    if source == "file":
        argv += ["--model", escape(_text(values, "model"))]
    elif source == "huggingface":
        argv += ["--model", "{component.models}/" + escape(model_file_name(values))]
    else:
        # Router mode: the web UI lists every model in the folder and loads
        # the one picked, so the server starts without one.
        argv += ["--models-dir", "{component.models}"]
    for name, flag in (("contextSize", "--ctx-size"), ("gpuLayers", "--gpu-layers")):
        value = values.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            argv += [flag, str(int(value))]
    argv += [escape(arg) for arg in split_args(_text(values, "extraArgs"), os_name)]
    return argv


def model_file_name(values: dict[str, Any]) -> str:
    """Where a Hugging Face model is saved in the models folder: its file's name.

    Args:
        values: The install parameters.

    Returns:
        The file name (the first part's, for a split model).
    """
    return posixpath.basename(_text(values, "hfFile"))


def split_args(text: str, os_name: str) -> list[str]:
    """Splits extra arguments as a POSIX shell would, keeping Windows backslashes.

    Args:
        text: The arguments as written.
        os_name: The platform's operating system.

    Returns:
        The arguments.

    Raises:
        ValueError: A quote is not closed.
    """
    if os_name == "windows":
        text = text.replace("\\", "\\\\")
    return shlex.split(text)


def escape(text: str) -> str:
    """Doubles braces, so a plan template keeps ``text`` as it is.

    Args:
        text: Any text.

    Returns:
        The text, as a template that renders to it.
    """
    return text.replace("{", "{{").replace("}", "}}")


def _hf_file_ok(path: str) -> bool:
    parts = path.split("/")
    return (
        path.lower().endswith(".gguf")
        and not path.startswith("/")
        and "\\" not in path
        and all(part not in ("", ".", "..") for part in parts)
    )


def _text(values: dict[str, Any], name: str) -> str:
    value = values.get(name, DEFAULTS.get(name))
    return value.strip() if isinstance(value, str) else ""
