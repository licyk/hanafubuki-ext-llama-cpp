"""llama.cpp's GitHub releases: which asset is which build, and which one to install.

Every release of ``ggml-org/llama.cpp`` is marked a prerelease and its assets
are uploaded one by one after the tag is published, so "latest" is the newest
``bNNNN`` release that already holds the asset a platform and variant need,
never ``/releases/latest``.

A build's asset is ``llama-<tag>-bin-<os>[-<build>]-<arch>.<tar.gz|zip>``
(``llama-b12287-bin-ubuntu-cuda-12.8-x64.tar.gz``); a CUDA build's runtime
libraries are ``cudart-llama[-<tag>]-bin-<os>-cuda-<version>-<arch>.<ext>``,
with the tag only on Linux.
"""

from __future__ import annotations

import platform
import re
from dataclasses import dataclass
from typing import Any

REPOSITORY = "ggml-org/llama.cpp"
RELEASES_URL = f"https://api.github.com/repos/{REPOSITORY}/releases?per_page=10"

# The variants a user picks, in the order the form lists them.
VARIANTS = ("cpu", "cuda-12", "cuda-13", "vulkan", "rocm", "sycl", "openvino", "metal")

# What the releases build for each platform (checked against a recorded
# release by the tests, so a change upstream shows up there).
SUPPORTED: dict[tuple[str, str], tuple[str, ...]] = {
    ("linux", "x64"): ("cpu", "cuda-12", "cuda-13", "vulkan", "rocm", "sycl", "openvino"),
    ("linux", "arm64"): ("cpu", "cuda-13", "vulkan"),
    ("windows", "x64"): ("cpu", "cuda-12", "cuda-13", "vulkan", "rocm", "sycl", "openvino"),
    ("windows", "arm64"): ("cpu", "cuda-13", "vulkan"),
    ("macos", "arm64"): ("cpu", "metal"),
    ("macos", "x64"): ("cpu",),
}

_TAG = re.compile(r"^b(\d+)$")
_ASSET = re.compile(
    r"^(?P<cudart>cudart-)?llama-(?:(?P<tag>b\d+)-)?bin-(?P<os>ubuntu|win|macos)"
    r"(?:-(?P<build>.+?))?-(?P<arch>x64|arm64)\.(?:tar\.gz|zip)$"
)
_OS = {"ubuntu": "linux", "win": "windows", "macos": "macos"}


@dataclass(frozen=True)
class AssetName:
    """What an asset's name says about it."""

    cudart: bool
    tag: str | None
    os: str
    arch: str
    # The part between the OS and the architecture (``cuda-12.4``), or "".
    build: str

    @property
    def variants(self) -> tuple[str, ...]:
        """The variants this build serves.

        Returns:
            Them; none for a build this extension does not offer.
        """
        return _variants(self.os, self.arch, self.build)


@dataclass(frozen=True)
class Selection:
    """The release and the assets an install downloads."""

    tag: str
    asset: dict[str, Any]
    cudart: dict[str, Any] | None


def host_platform() -> tuple[str, str]:
    """This machine's operating system and architecture, as the assets name them.

    Returns:
        ``(os, arch)``, e.g. ``("linux", "x64")``; an unknown one keeps its own name.
    """
    system = {"Linux": "linux", "Windows": "windows", "Darwin": "macos"}.get(
        platform.system(), platform.system().lower()
    )
    machine = platform.machine().lower()
    arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(
        machine, machine
    )
    return system, arch


def supported_variants(os_name: str, arch: str) -> tuple[str, ...]:
    """The variants a platform has builds of.

    Args:
        os_name: ``linux``, ``windows`` or ``macos``.
        arch: ``x64`` or ``arm64``.

    Returns:
        Its variants; none for a platform llama.cpp does not build for.
    """
    return SUPPORTED.get((os_name, arch), ())


def parse_asset(name: str) -> AssetName | None:
    """Reads an asset's name.

    Args:
        name: The asset's file name.

    Returns:
        What it says, or ``None`` for an asset that is no build of a desktop platform.
    """
    match = _ASSET.match(name)
    if match is None:
        return None
    return AssetName(
        cudart=bool(match["cudart"]),
        tag=match["tag"],
        os=_OS[match["os"]],
        arch=match["arch"],
        build=match["build"] or "",
    )


def select(
    releases: list[dict[str, Any]], os_name: str, arch: str, variant: str, version: str
) -> Selection:
    """Picks the release and the assets to install.

    Args:
        releases: The GitHub API's releases, in any order.
        os_name: The platform's operating system.
        arch: Its architecture.
        variant: One of ``VARIANTS``.
        version: ``latest`` or a release tag (``b12287``).

    Returns:
        The release, its build for the platform and variant, and for a CUDA
        build on Linux or Windows its runtime libraries.

    Raises:
        LookupError: No release (or not the one named) holds that build.
    """
    candidates = sorted(
        (release for release in releases if _usable(release)),
        key=lambda release: int(release["tag_name"][1:]),
        reverse=True,
    )
    if version != "latest":
        candidates = [release for release in candidates if release["tag_name"] == version]
        if not candidates:
            raise LookupError(f"llama.cpp has no release {version}")
    for release in candidates:
        selection = _select_in(release, os_name, arch, variant)
        if selection is not None:
            return selection
    where = "the newest releases" if version == "latest" else f"release {version}"
    raise LookupError(f"{where} of llama.cpp hold no {variant} build for {os_name}-{arch}")


def digest(asset: dict[str, Any]) -> str | None:
    """An asset's SHA-256 as GitHub reports it (``sha256:<hex>``).

    Args:
        asset: The API's asset.

    Returns:
        The hex digest, or ``None`` when GitHub gives none.
    """
    value = asset.get("digest")
    if isinstance(value, str) and value.startswith("sha256:"):
        return value.removeprefix("sha256:").lower()
    return None


def _usable(release: dict[str, Any]) -> bool:
    tag = release.get("tag_name")
    return not release.get("draft") and isinstance(tag, str) and _TAG.match(tag) is not None


def _select_in(release: dict[str, Any], os_name: str, arch: str, variant: str) -> Selection | None:
    tag = release["tag_name"]
    builds: list[tuple[AssetName, dict[str, Any]]] = []
    runtimes: list[tuple[AssetName, dict[str, Any]]] = []
    for asset in release.get("assets") or []:
        parsed = parse_asset(str(asset.get("name", "")))
        if parsed is None or parsed.os != os_name or parsed.arch != arch:
            continue
        if parsed.cudart:
            runtimes.append((parsed, asset))
        elif parsed.tag == tag and variant in parsed.variants:
            builds.append((parsed, asset))
    if not builds:
        return None
    build, asset = builds[0]
    if not variant.startswith("cuda-") or os_name == "macos":
        return Selection(tag=tag, asset=asset, cudart=None)
    # The runtime is the one built for the same CUDA version; a release still
    # uploading may hold the build before it.
    for runtime, runtime_asset in runtimes:
        if runtime.build == build.build and runtime.tag in (None, tag):
            return Selection(tag=tag, asset=asset, cudart=runtime_asset)
    return None


def _variants(os_name: str, arch: str, build: str) -> tuple[str, ...]:
    if build in ("", "cpu"):
        # Only the Apple silicon build has Metal.
        return ("cpu", "metal") if (os_name, arch) == ("macos", "arm64") else ("cpu",)
    if build in ("vulkan", "sycl", "sycl-fp32"):
        return ("sycl",) if build.startswith("sycl") else (build,)
    for prefix, variant in (
        ("cuda-12.", "cuda-12"),
        ("cuda-13.", "cuda-13"),
        ("rocm-", "rocm"),
        ("openvino-", "openvino"),
    ):
        if build.startswith(prefix):
            return (variant,)
    return ()
