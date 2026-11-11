"""Choosing a release and its assets, against releases recorded from GitHub's API."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from support import known

from hanafubuki_llama import releases as rel

FIXTURE = Path(__file__).parent / "fixtures" / "releases.json"


@pytest.fixture
def recorded() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("os_name", "arch", "variant", "asset", "runtime"),
    [
        ("linux", "x64", "cpu", "llama-b12289-bin-ubuntu-x64.tar.gz", None),
        ("linux", "x64", "vulkan", "llama-b12289-bin-ubuntu-vulkan-x64.tar.gz", None),
        (
            "linux",
            "x64",
            "cuda-12",
            "llama-b12289-bin-ubuntu-cuda-12.8-x64.tar.gz",
            "cudart-llama-b12289-bin-ubuntu-cuda-12.8-x64.tar.gz",
        ),
        ("linux", "x64", "sycl", "llama-b12289-bin-ubuntu-sycl-fp32-x64.tar.gz", None),
        ("linux", "x64", "rocm", "llama-b12289-bin-ubuntu-rocm-10.0-x64.tar.gz", None),
        ("linux", "arm64", "cpu", "llama-b12289-bin-ubuntu-arm64.tar.gz", None),
        ("windows", "x64", "cpu", "llama-b12289-bin-win-cpu-x64.zip", None),
        (
            "windows",
            "x64",
            "cuda-12",
            "llama-b12289-bin-win-cuda-12.4-x64.zip",
            "cudart-llama-bin-win-cuda-12.4-x64.zip",
        ),
        ("windows", "x64", "openvino", "llama-b12289-bin-win-openvino-2026.4.1-x64.zip", None),
        ("macos", "arm64", "metal", "llama-b12289-bin-macos-arm64.tar.gz", None),
        ("macos", "arm64", "cpu", "llama-b12289-bin-macos-arm64.tar.gz", None),
        ("macos", "x64", "cpu", "llama-b12289-bin-macos-x64.tar.gz", None),
    ],
)
def test_latest_picks_the_newest_build_of_the_variant(
    recorded, os_name, arch, variant, asset, runtime
):
    selection = rel.select(recorded, os_name, arch, variant, "latest")
    assert selection.tag == "b12289"
    assert selection.asset["name"] == asset
    assert (selection.cudart or {}).get("name") == runtime
    assert len(known(rel.digest(selection.asset))) == 64


def test_a_release_still_uploading_is_skipped_for_what_it_lacks(recorded):
    newest = recorded[0]
    newest["assets"] = [
        asset
        for asset in newest["assets"]
        if "win-vulkan" not in asset["name"] and not asset["name"].startswith("cudart")
    ]
    assert rel.select(recorded, "windows", "x64", "vulkan", "latest").tag == "b12287"
    # A CUDA build without its runtime yet is skipped too.
    assert rel.select(recorded, "linux", "x64", "cuda-12", "latest").tag == "b12287"
    assert rel.select(recorded, "linux", "x64", "cpu", "latest").tag == "b12289"


def test_order_and_drafts_do_not_mislead_latest(recorded):
    shuffled = [recorded[2], recorded[0], recorded[1]]
    assert rel.select(shuffled, "linux", "x64", "cpu", "latest").tag == "b12289"
    drafted = copy.deepcopy(recorded)
    drafted[0]["draft"] = True
    assert rel.select(drafted, "linux", "x64", "cpu", "latest").tag == "b12287"


def test_a_named_release(recorded):
    assert rel.select(recorded, "linux", "x64", "cpu", "b12284").asset["name"] == (
        "llama-b12284-bin-ubuntu-x64.tar.gz"
    )
    with pytest.raises(LookupError, match="no release b1"):
        rel.select(recorded, "linux", "x64", "cpu", "b1")


def test_a_variant_without_a_build_is_refused(recorded):
    with pytest.raises(LookupError, match="no rocm build for macos-arm64"):
        rel.select(recorded, "macos", "arm64", "rocm", "latest")


def test_the_supported_table_matches_a_recorded_release(recorded):
    """Every offered variant has a build, and every build of a desktop platform is offered."""
    built: dict[tuple[str, str], set[str]] = {}
    for asset in recorded[1]["assets"]:
        parsed = rel.parse_asset(asset["name"])
        if parsed is not None and not parsed.cudart:
            built.setdefault((parsed.os, parsed.arch), set()).update(parsed.variants)
    assert {platform: set(variants) for platform, variants in rel.SUPPORTED.items()} == built
    assert set().union(*rel.SUPPORTED.values()) == set(rel.VARIANTS)


def test_asset_names_are_read_by_their_parts():
    parsed = known(rel.parse_asset("llama-b12287-bin-ubuntu-cuda-13.5-arm64.tar.gz"))
    assert parsed == rel.AssetName(False, "b12287", "linux", "arm64", "cuda-13.5")
    assert parsed.variants == ("cuda-13",)
    runtime = rel.parse_asset("cudart-llama-bin-win-cuda-12.4-x64.zip")
    assert runtime == rel.AssetName(True, None, "windows", "x64", "cuda-12.4")
    for other in (
        "llama-b12287-bin-ubuntu-26.04-s390x.tar.gz",
        "llama-b12287-bin-android-arm64.tar.gz",
        "llama-b12287-bin-linux-arm64-snapdragon.tar.gz",
        "llama-b12287-xcframework.zip",
        "llama-b12287-ui.tar.gz",
    ):
        assert rel.parse_asset(other) is None
    for unoffered in (
        "llama-b12287-bin-ubuntu-sycl-fp16-x64.tar.gz",
        "llama-b12287-bin-win-opencl-adreno-arm64.zip",
    ):
        assert known(rel.parse_asset(unoffered)).variants == ()
