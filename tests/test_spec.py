"""The backend's checks of an instance's values, and the server command they make."""

from __future__ import annotations

import pytest

from hanafubuki_llama.spec import escape, server_argv, split_args, validate

DEFAULTS = {"version": "latest", "variant": "cpu", "modelSource": "directory", "contextSize": 4096}


def test_the_defaults_are_good_on_every_platform():
    for os_name, arch in (("linux", "x64"), ("windows", "arm64"), ("macos", "x64")):
        assert validate(DEFAULTS, os_name, arch) == {}


@pytest.mark.parametrize(
    ("values", "platform", "field", "message"),
    [
        ({"version": "1.0"}, ("linux", "x64"), "version", "release tag"),
        ({"variant": "metal"}, ("linux", "x64"), "variant", "no metal build for linux-x64"),
        ({"variant": "cuda-12"}, ("linux", "arm64"), "variant", "cpu, cuda-13, vulkan"),
        ({"variant": "cpu"}, ("linux", "riscv64"), "variant", "no builds for linux-riscv64"),
        ({"modelSource": "file", "model": "/m/a.bin"}, ("linux", "x64"), "model", ".gguf"),
        (
            {"modelSource": "huggingface", "hfRepo": "no-slash", "hfFile": "a.gguf"},
            ("linux", "x64"),
            "hfRepo",
            "owner/name",
        ),
        (
            {"modelSource": "huggingface", "hfRepo": "o/r", "hfFile": "../a.gguf"},
            ("linux", "x64"),
            "hfFile",
            ".gguf",
        ),
        ({"extraArgs": "--port 9 --jinja"}, ("linux", "x64"), "extraArgs", "set by Hanafubuki"),
        ({"extraArgs": "--host=0.0.0.0"}, ("linux", "x64"), "extraArgs", "set by Hanafubuki"),
        ({"extraArgs": "--alias 'open"}, ("linux", "x64"), "extraArgs", "cannot be split"),
    ],
)
def test_what_the_form_cannot_say_is_refused(values, platform, field, message):
    issues = validate({**DEFAULTS, **values}, *platform)
    assert list(issues) == [field]
    assert message in issues[field]


def test_a_folder_of_models_starts_the_router():
    assert server_argv(DEFAULTS, "linux") == [
        "{component.core}/llama-server",
        "--host",
        "127.0.0.1",
        "--port",
        "{port}",
        "--models-dir",
        "{component.models}",
        "--ctx-size",
        "4096",
    ]


def test_a_downloaded_model_is_loaded_from_the_models_folder():
    argv = server_argv(
        {
            **DEFAULTS,
            "modelSource": "huggingface",
            "hfRepo": "ggml-org/gemma-3-1b-it-GGUF",
            "hfFile": "Q4/gemma-00001-of-00002.gguf",
            "gpuLayers": 20,
            "contextSize": 0,
        },
        "windows",
    )
    assert argv[0] == "{component.core}/llama-server.exe"
    assert argv[5:] == [
        "--model",
        "{component.models}/gemma-00001-of-00002.gguf",
        "--ctx-size",
        "0",
        "--gpu-layers",
        "20",
    ]


def test_user_text_stays_literal_in_the_template():
    argv = server_argv(
        {
            **DEFAULTS,
            "modelSource": "file",
            "model": "/models/{odd}.gguf",
            "extraArgs": "--jinja --chat-template-kwargs '{\"enable_thinking\": false}'",
        },
        "linux",
    )
    assert argv[5:7] == ["--model", "/models/{{odd}}.gguf"]
    assert argv[-3:] == [
        "--jinja",
        "--chat-template-kwargs",
        '{{"enable_thinking": false}}',
    ]


def test_windows_paths_keep_their_backslashes():
    assert split_args(r'--lora "C:\loras\a b.gguf" -t 4', "windows") == [
        "--lora",
        r"C:\loras\a b.gguf",
        "-t",
        "4",
    ]
    assert split_args(r"--x a\ b", "linux") == ["--x", "a b"]


def test_escape_doubles_braces():
    assert escape("{a}}") == "{{a}}}}"
