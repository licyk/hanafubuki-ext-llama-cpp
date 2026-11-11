# llama.cpp for Hanafubuki

[中文](#中文)

A [Hanafubuki](https://github.com/licyk/Hanafubuki) extension that installs
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`, runs it
as a supervised server whose chat page opens in Hanafubuki, and keeps it
up to date.

- **One instance type**, *llama.cpp server* (`io.github.licyk.llama-cpp:server`),
  in the install page with its own preview.
- **Every published build**: CPU, CUDA 12, CUDA 13, Vulkan, ROCm, SYCL,
  OpenVINO and Metal, offered per platform as llama.cpp publishes them
  (Linux and Windows on x64 and arm64, macOS). A CUDA build comes with its
  CUDA runtime.
- **Three ways to give it a model**: a folder of GGUF files to pick from on
  the server's page (the default: llama-server's router mode), a model
  downloaded from Hugging Face at install, or a GGUF file on the computer.
- **Downloads with sd-webui-all-in-one**: GitHub and Hugging Face mirrors,
  SHA-256 checks, resumable downloads, the instance's proxy.
- **Update** to the newest release from the instance's Maintenance section,
  a **status panel** with the installed build, the latest release and the
  models, a **terminal** with llama.cpp's programs on `PATH`, and
  **discovery** of instances in its layout.

## Requirements

- Hanafubuki 0.3.28 or later (below 0.4), with the extension-validation host
  features: types an extension's backend provides with `{host.python}`,
  `{extension.root}` and `[environment]`, the declarative terminal, preview
  images, discovery signatures, and `sdWebuiAllInOne` for Python backends.
- Nothing else: the backend and the install script run on Hanafubuki's
  managed Python, and llama.cpp's builds need no installation. A GPU build
  needs the GPU's driver (and for ROCm, SYCL or OpenVINO their runtimes).

## Installing the extension

- **From an archive**: in Settings > Extensions, install
  `io.github.licyk.llama-cpp-<version>.zip` (see [Packaging](#packaging)).
- **As a development link**: in Settings > Extensions, link a clone of this
  repository (the repository's root is the package). Hanafubuki reads the folder in place; after a
  change, press the extension's Reload. A linked folder may hold no
  symbolic link, so keep virtual environments and `node_modules` out of it.

Turn the extension on; *llama.cpp server* is then in the install page.

## How it works

| Step | What happens |
| --- | --- |
| Create | Hanafubuki checks the form, then asks the backend (`provider.validateSpec`), which refuses a build that is not published for this platform, a malformed version or repository, and `--host`/`--port` in the extra arguments. |
| Install | The backend's plan (`provider.installPlan`) is two steps: write `models/README.txt`, then run `scripts/install.py` on the managed Python (`{host.python}`), which imports `sd_webui_all_in_one`. The script resolves the release, downloads the build (and the CUDA runtime) into `core/.staging`, checks GitHub's SHA-256, extracts it with `extract_archive`, swaps it in for the previous build, writes `core/.hanafubuki-llama.json`, and downloads the Hugging Face model into `models`. It writes only in `core` and `models`. |
| Start | The `ui` role runs `core/llama-server --host 127.0.0.1 --port {port}` with the model arguments (`--models-dir models`, or `--model <file>`), `--ctx-size`, `--gpu-layers` and the extra arguments (`provider.launchPlan`). It is ready when `/health` answers (up to 10 minutes, for large models), opens in Hanafubuki's browser, and restarts if it fails. |
| Update | Maintenance > Update runs the same script. With version `latest` it installs the newest release; a build already installed is not downloaded again. To switch builds or pin a version, update with other values. |
| Status | The *llama.cpp* section of the instance's management (a schema panel, `ui/status.json`) shows the installed build, the latest release and the models; *Check for updates* asks GitHub now. The `check-update` operation answers the same for the management tools (the Agent, MCP clients). |
| Terminal | Opens in `core` with it first on `PATH` (`llama-cli`, `llama-quantize`, ...), and `LLAMA_CACHE` set to `models`. |
| Discover | A folder holding `core/llama-server` (or `.exe`) and `models` is found by Hanafubuki's instance discovery and imported with the default parameters. |

**Choosing "latest".** Every llama.cpp release is marked a prerelease, so
GitHub's "latest release" is never one of them, and a release's files are
uploaded for a while after it is published. *Latest* is the newest `bNNNN`
release that already holds the build for this platform and variant.

**Why a schema panel, not a frame.** What the section shows is a few facts
and a table, which Hanafubuki renders natively, themed and localized, with
no web code to build or ship. The one action that changes the instance,
updating, is Hanafubuki's own Update with its permits and progress; a panel
action may not change an instance's files anyway. A frame would add a Vue
build (whose `node_modules` cannot live in a linked folder) for nothing the
panel cannot show.

### Settings

| Setting | Values |
| --- | --- |
| GitHub mirrors | *When the direct download fails* (default), *First*, *Never*: sd-webui-all-in-one's GitHub mirror list, for the release archives. |
| Hugging Face mirror | The same, with hf-mirror.com, for models. |

### Permissions

- `instances = "own-types"`: the status panel finds its instances' folders
  with `instances.ofType`.
- `network = ["api.github.com"]`: the backend lists releases with
  `download.fetch`, so the request goes through Hanafubuki's proxy (the
  backend's own environment carries none). The listing is cached for 15
  minutes in the extension's data directory.

The install script runs as an instance task, not through the host API, so
its downloads need no permission. As for every extension, the backend is a
process with the user's rights; the permissions gate only the host API.

### Limits

- Gated Hugging Face repositories are not supported (the downloader sends no
  token); download such a model yourself into `models`.
- The status panel reads `<instance>/core` and `<instance>/models`: an
  instance imported with other component folders shows "Not installed".
- A discovered instance takes the default parameters (the CPU build): set
  the build with Update's values before updating it.
- A CUDA build on Linux downloads its runtime too (about 600 MB).

## Development

```text
backend/main.py           the backend (Python SDK): validateSpec, installPlan, launchPlan, the panel, check-update
scripts/install.py        the install and update step, on the managed Python with sd_webui_all_in_one
hanafubuki_llama/         what both share: release selection, the spec checks and the server command, the record
types/server.toml         the type: components, parameters, environment, operations, server role, discovery
ui/status.json            the status panel
schemas/settings.json     the settings
locales/{en,zh}.json      the strings
images/preview.png        the type's preview (tools/make_preview.py screenshots a running server)
tests/                    the tests
```

The tests use a Hanafubuki checkout for the extension SDK and a
sd-webui-all-in-one checkout, by default the sibling folders
`../Hanafubuki-dev` and `../sd-webui-all-in-one`; set `HANAFUBUKI_SDK` (the
SDK's `sdk/python` folder) and `SD_WEBUI_ALL_IN_ONE` to use others.

```sh
python -m pytest                          # Python 3.11 or later
LLAMA_CPP_REAL_TEST=1 python -m pytest tests/test_real.py   # downloads the real CPU build and starts it
python -m ruff format --check . && python -m ruff check .
python -m ty check --python "$(command -v python)" .
```

- `test_releases.py`: release and asset selection on releases recorded from
  GitHub's API (`tests/fixtures/releases.json`), including a release still
  uploading, and that the platform table matches what is published.
- `test_spec.py`: the backend's checks and the server command.
- `test_install.py`: `scripts/install.py` in a process of its own with the
  real sd_webui_all_in_one, against archives built in the test (tar.gz with
  a top directory and a symbolic link, flat zip, a CUDA build and its
  runtime) and a Hugging Face model, served over local HTTP: staging, the
  swap, the record, digests, the urllib downloader.
- `test_backend.py`: `backend/main.py` spawned as Hanafubuki spawns it,
  over the protocol.
- `test_package.py`: the manifest, the type, the strings in both locales,
  the preview, no links in the repository, and the archive.

## Packaging

```sh
python scripts/package.py      # writes dist/io.github.licyk.llama-cpp-<version>.zip
```

The archive holds the package only (the manifest, `backend`, `scripts/install.py`,
`hanafubuki_llama`, `types`, `ui`, `schemas`, `locales`, `images`, the README
and the license), not the tests, tools or development files.

## License

GPL-3.0-only.

---

## 中文

为 [Hanafubuki](https://github.com/licyk/Hanafubuki) 提供
[llama.cpp](https://github.com/ggml-org/llama.cpp) 的 `llama-server`：安装、作为受监管的服务器运行（聊天页面在 Hanafubuki 中打开）并保持更新。

- **实例类型** *llama.cpp 服务器*：可选 CPU、CUDA 12/13、Vulkan、ROCm、SYCL、OpenVINO、Metal 构建，按平台提供 llama.cpp 实际发布的构建；CUDA 构建会附带 CUDA 运行库。
- **模型**：默认使用模型文件夹（在服务器页面中选择模型），也可在安装时从 Hugging Face 下载，或使用本机的 GGUF 文件。
- **下载**：安装脚本 `scripts/install.py` 在 Hanafubuki 的托管 Python 上运行，使用 sd-webui-all-in-one 的下载器（GitHub / Hugging Face 镜像、SHA-256 校验、断点续传、实例代理）与 `extract_archive`。
- **更新**：在实例的维护中运行“更新”；版本为 `latest` 时安装最新发布版本。实例管理中的 *llama.cpp* 面板显示已安装构建、最新版本与模型列表。
- **终端**：llama.cpp 的程序在 `PATH` 上；**发现**：含 `core/llama-server` 与 `models` 的文件夹可被实例发现导入。

需要 Hanafubuki 0.3.28 及以上（低于 0.4）。可通过“设置 > 扩展”安装打包文件，或以开发链接方式链接本仓库文件夹（文件夹内不能有符号链接）。打包：`python scripts/package.py`。
