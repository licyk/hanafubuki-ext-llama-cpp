"""Helpers shared by the tests: the paths, and a subprocess environment.

``HANAFUBUKI_SDK`` names the extension SDK's directory (``sdk/python`` of a
Hanafubuki checkout) and ``SD_WEBUI_ALL_IN_ONE`` a checkout of
sd-webui-all-in-one; both default to sibling checkouts of this repository.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SDK = Path(os.environ.get("HANAFUBUKI_SDK", ROOT.parent / "Hanafubuki-dev" / "sdk" / "python"))
SD_WEBUI_ALL_IN_ONE = Path(
    os.environ.get("SD_WEBUI_ALL_IN_ONE", ROOT.parent / "sd-webui-all-in-one")
)
PYTHON = sys.executable


def environment(*paths: object) -> dict[str, str]:
    """The test's environment with ``paths`` as the whole ``PYTHONPATH`` and no proxy.

    Args:
        *paths: The directories, in order.

    Returns:
        The environment.
    """
    proxies = ("http_proxy", "https_proxy", "all_proxy", "no_proxy")
    env = {key: value for key, value in os.environ.items() if key.lower() not in proxies}
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def known(value: Any) -> Any:
    """``value``, which must not be ``None``.

    Args:
        value: What a lookup returned.

    Returns:
        The value.
    """
    assert value is not None
    return value
