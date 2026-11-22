<div align="center">
<img src="./Logo.png" alt="logo" width="100"/>
<h2>ModelTranslator</h2>
<h3>Multi-format Model Conversion Tool (uv environment)</h3>
</div>


### 1. Introduction

`ModelTranslator` is a **multi-format model conversion tool** managed by [uv](https://docs.astral.sh/uv/), supporting conversion between common deep learning model formats. All conversion work is **delegated to existing mature libraries** (torch / onnx / onnxruntime / tensorflow / tf2onnx / openvino / coremltools / trtexec / llama.cpp, etc.); this project only handles format detection and conversion orchestration, without reinventing the wheel.

- ✅ Supports **12 common model formats and 11 declared conversion paths** (not every pair); run `model-translator list` for every reachable path
- ✅ Declares **5 deployment targets** (rknn / Ascend OM / Cambricon DLC / Horizon BPU / Apple ANE) with routing
- ✅ Dependencies **pulled on demand**: the CLI core is only a few MB, backends installed as needed
- ✅ `--auto-install` automatically installs missing dependencies
- ✅ Format detection reads **file contents** (magic bytes / zip layout / directory marker files); extensions are only a fallback
- ✅ Docker-ready, works out of the box
- ⚠️ NPU paths are **not verified in this repository** (no vendor SDK on the dev machine); `list` says so explicitly

### 2. Quick Start

```bash
# 1. Install (only the lightweight CLI by default)
uv sync

# 2. List supported formats and conversions (shows the format x deployment-target matrix)
uv run model-translator list

# 3. Inspect one deployment target
uv run model-translator list --target rknn

# 4. Convert: PyTorch -> ONNX (auto-installs missing deps with --auto-install)
uv run model-translator convert model.pt model.onnx --input-shape 1,3,224,224 --auto-install

# 5. Deploy to an NPU target (dry-run the route first, run for real in the target environment)
uv run model-translator convert model.onnx yolo.rknn --target rknn --dry-run

# 6. Docker
docker build -t model-translator .
docker run --rm -v $(pwd):/workspace model-translator list
```

### 3. Supported Formats

| Format key | Name | Extensions | Description |
|------------|------|------------|-------------|
| `pytorch` | PyTorch | `.pt` `.pth` | PyTorch weights/model |
| `torchscript` | TorchScript | `.pt` | TorchScript serialized model |
| `onnx` | ONNX | `.onnx` | Open Neural Network Exchange |
| `tensorflow` | TensorFlow SavedModel | `.pb` / directory | SavedModel |
| `keras` | Keras | `.h5` `.keras` | Keras HDF5 model |
| `tflite` | TensorFlow Lite | `.tflite` | Lightweight inference format |
| `openvino` | OpenVINO IR | `.xml` | OpenVINO intermediate representation |
| `tensorrt` | TensorRT | `.engine` `.trt` | NVIDIA inference engine |
| `coreml` | CoreML | `.mlmodel` `.mlpackage` | Apple CoreML |
| `ggml` | GGUF | `.gguf` `.bin` | Quantized LLM format |
| `safetensors` | Safetensors | `.safetensors` | HuggingFace safe weights (source of the GGUF path) |
| `mindspore` | MindSpore | `.ckpt` `.mindir` | Huawei MindSpore model |

> Use `uv run model-translator info <path>` to detect the format of any file. It also
> prints the **detection evidence** (magic bytes / zip layout / header structure /
> directory marker / extension fallback) so you can check why it decided what it decided.
>
> There are **12** formats but only **11** conversion edges: not every pair of formats
> can be converted. Reachable combinations are exactly the "conversion paths" section
> printed by `uv run model-translator list`.

### 4. Supported Conversion Paths

| Source | Target | Library | Underlying implementation |
|--------|--------|---------|---------------------------|
| PyTorch | ONNX | torch + onnx | `torch.onnx.export` |
| PyTorch | TorchScript | torch | `torch.jit.trace/script` |
| ONNX | OpenVINO | openvino | `ov.convert_model` |
| ONNX | CoreML | coremltools | `coremltools.convert` |
| ONNX | TensorRT | trtexec | invokes NVIDIA `trtexec` |
| SavedModel | TFLite | tensorflow | `TFLiteConverter` |
| Keras | TFLite | tensorflow | `TFLiteConverter` |
| SavedModel | ONNX | tensorflow + tf2onnx | `tf2onnx.convert` |
| Keras | ONNX | tensorflow + tf2onnx | `tf2onnx.convert` |
| CoreML | ONNX | coremltools | coremltools 7+ to ONNX |
| safetensors | GGUF | llama.cpp | `convert_hf_to_gguf.py` |

### 5. Deployment Targets (format × target)

`uv run model-translator list --target <key>` shows a single target's entry formats,
toolchain, acquisition route and verification status.

| Target key | Name | Vendor | Toolchain | Entry format | How to get | Verified |
|-----------|------|--------|-----------|--------------|-----------|----------|
| `rknn` | Rockchip RKNN | Rockchip | rknn-toolkit2 | ONNX | Official wheel (not on PyPI, x86 Linux only) | **No** |
| `om` | Ascend OM | Huawei Ascend | `atc` (CANN) | ONNX | Install CANN Toolkit, add `$ASCEND_HOME/bin` to PATH | **No** |
| `dlc` | Cambricon DLC | Cambricon | `cnmodel` (CNToolkit) | ONNX | Install CNToolkit, add bin to PATH | **No** |
| `bin` | Horizon BPU bin (Horizon Journey) | Horizon | `hb_mapper` (OpenExplorer) | ONNX | Install OpenExplorer (horizon_tc) | **No** |
| `mlpackage` | CoreML mlpackage (Apple ANE) | Apple | coremltools | ONNX | `uv sync --extra coreml` | **No** (reuses onnx->coreml) |

**Why "not verified"**: this repository's development environment (Windows) has no
CANN / rknn-toolkit2 / CNToolkit / OpenExplorer, so these paths **have never run
through in this repo**. Parameters are built from each vendor's public docs; both
`list` and `convert` state this explicitly rather than implying it works. Verify by
actually running them in the target environment.

> All 5 targets have `verified=False`, and this is not merely a documentation promise:
> `test_no_target_claims_verified` in `tests/test_registry.py` asserts that **no target
> may claim to be verified**. Setting `verified=True` without real evidence turns CI red
> immediately, so the unverified status is enforced, not asserted in prose.

**Routing rule**: if the source format has a direct path to the target, it is used;
otherwise the tool first converts to the intermediate format (ONNX by default) and
then to the target. For example `pytorch -> rknn` becomes two steps, with the
intermediate artifact named `out.rknn.step0.onnx` so it is never mistaken for the
final product.

```bash
# Show the route without running anything
uv run model-translator convert model.pt out.rknn --target rknn --input-shape 1,3,224,224 --dry-run

# Ascend: the target SoC must be given (atc never guesses; a wrong one yields an .om that won't run)
uv run model-translator convert model.onnx out.om --target om --opt soc_version=Ascend310P3

# RKNN: INT8 quantization only when a calibration set is given, otherwise explicitly disabled
uv run model-translator convert model.onnx out.rknn --target rknn \
  --opt input_shape=1,3,224,224 --opt dataset=calib.txt

# Horizon: hb_mapper's YAML config is mandatory (output name/input shape/calib live in the yaml)
uv run model-translator convert model.onnx out.bin --target bin --opt hb_config=config.yaml
```

`--opt KEY=VALUE` is repeatable and passes parameters straight to the vendor
toolchain; CLI-native options (like `--input-shape`) still feed the format converters.

### 6. CLI Reference

#### `list`

Lists supported formats and all conversion paths, plus the format × deployment-target matrix.

```
uv run model-translator list
uv run model-translator list --target rknn
```

#### `info`

Detects the model format of a file/directory.

```
uv run model-translator info <path>
```

#### `convert`

The core model conversion command.

```
uv run model-translator convert <src> <dst> [options]
```

| Option | Description |
|--------|-------------|
| `--to / -t <key>` | Target format key (defaults to output extension) |
| `--target / -g <key>` | Deployment target key (`rknn/om/dlc/bin/mlpackage`), routes via the format × target matrix |
| `--input-shape <N,C,H,W>` | Example input shape, e.g. `1,3,224,224` (required for PyTorch) |
| `--opset <n>` | ONNX opset version (default 13) |
| `--input-names <a,b>` | ONNX input names, comma-separated |
| `--output-names <a,b>` | ONNX output names, comma-separated |
| `--dynamic-axes` | Export dynamic axes (batch dimension) in ONNX |
| `--fp16` | Use FP16 for TensorRT |
| `--quantize int8` | TFLite quantization mode |
| `--script <path>` | Custom Python script (defines `load_model(path)` returning a model) to load PyTorch state_dict |
| `--allow-pickle` | **Off by default**: allow pickle deserialization of full PyTorch models (unsafe, trusted models only; without it `torch.load(weights_only=True)` rejects pickle so malicious `.pt` cannot run code via `__reduce__`) |
| `--auto-install` | Auto-install missing dependencies on demand (`uv sync --extra <needed groups>`) |
| `--opt KEY=VALUE` | Repeatable pass-through for deployment-target toolchains (see section 5) |
| `--dry-run` | Print the route, implementations and output paths without running anything |

**Examples:**

```bash
# PyTorch -> ONNX
uv run model-translator convert model.pt model.onnx --input-shape 1,3,224,224

# PyTorch -> TorchScript
uv run model-translator convert model.pt model_scripted.pt --input-shape 1,3,224,224

# PyTorch(state_dict) -> ONNX, restore structure via script
uv run model-translator convert weights.pth model.onnx \
  --script ./load_my_model.py --input-shape 1,3,224,224

# ONNX -> OpenVINO / CoreML / TensorRT
uv run model-translator convert model.onnx model.xml
uv run model-translator convert model.onnx model.mlpackage
uv run model-translator convert model.onnx model.engine --fp16

# SavedModel/Keras -> TFLite / ONNX
uv run model-translator convert saved_model_dir model.tflite
uv run model-translator convert model.h5 model.tflite --quantize int8
uv run model-translator convert saved_model_dir model.onnx

# Deployment targets (dry-run the route first)
uv run model-translator convert model.onnx yolo.rknn --target rknn --dry-run

# Auto-install missing deps
uv run model-translator convert model.pt model.onnx --input-shape 1,3,224,224 --auto-install
```

> When the target format cannot be inferred (e.g. Horizon's `.bin` collides with
> GGUF's `.bin`), pass `--target` explicitly; this tool never guesses from the extension.

### 7. On-demand Dependencies

The CLI core is the only mandatory dependency (click + filetype, just a few MB); all conversion backends live in the `optional-dependencies` groups of [pyproject.toml](pyproject.toml):

| Group | Packages | Size |
|-------|----------|------|
| `onnx` | onnx / onnxruntime / onnxscript | Medium |
| `torch` | torch | Large (hundreds of MB+) |
| `tensorflow` | tensorflow / tf2onnx | Largest (1GB+) |
| `openvino` | openvino | Medium |
| `coreml` | coremltools (macOS only) | Large |
| `dev` | pytest (tests only, no conversion backend) | Tiny |
| `all` | everything above | Very large (not recommended) |

NPU toolchains (rknn-toolkit2 / CANN atc / CNToolkit / OpenExplorer) are **not on
PyPI** and cannot be installed via an extra group; see the section 5 table and
`model-translator list --target <key>` for how to obtain them.

**Two ways to install on demand:**

```bash
# Way 1: install manually
uv sync --extra torch --extra onnx

# Way 2: auto-install during conversion (--auto-install)
uv run model-translator convert model.pt model.onnx --input-shape 1,3,224,224 --auto-install
```

> When `convert` finds a missing dependency, it prints the **precise on-demand install command** (e.g. `uv sync --extra torch --extra onnx`) — you never need to pull all backends at once.

### 8. Docker

```bash
cd ModelTranslator

# Build image (only the onnx backend by default, lightweight)
docker build -t model-translator .

# Customize backends
docker build --build-arg 'UV_EXTRAS=onnx openvino' -t model-translator .

# All backends (very large, usually not needed)
docker build --build-arg 'UV_EXTRAS=all' -t model-translator-full .

# List supported formats
docker run --rm -v $(pwd):/workspace model-translator list

# Mount directory and convert (input/output under /workspace)
docker run --rm -v $(pwd):/workspace model-translator \
  convert /workspace/model.pt /workspace/model.onnx --input-shape 1,3,224,224

# Or use docker-compose
docker compose run --rm translator \
  convert /workspace/model.pt /workspace/model.onnx --input-shape 1,3,224,224
```

> On Windows, mount with `docker run --rm -v %cd%:/workspace ...` (cmd) or `-v ${PWD}:/workspace` (PowerShell).

#### Dockerfile notes

- Base image: `ghcr.io/astral-sh/uv:python3.11-bookworm-slim` (official uv image with uv built in)
- Use `--build-arg UV_EXTRAS` to choose which dependency groups to install (space-separated)
- Working directory `/workspace`; mount your model inputs/outputs there

### 9. Project Structure

```
ModelTranslator/
├── pyproject.toml              # uv project definition + optional dependency groups (incl. dev)
├── Dockerfile                  # Docker image (on-demand build)
├── docker-compose.yml          # one-command compose run
├── .github/workflows/
│   ├── docker-image.yml        # image build
│   ├── release.yml             # Release packaging
│   └── python-test.yml         # pytest + list-matrix gate on PR/push
├── src/model_translator/
│   ├── cli.py                  # CLI entry (list/info/convert, incl. --target routing)
│   ├── registry.py             # format registry + conversion paths + deployment targets/routing
│   └── converters/
│       ├── __init__.py         # format -> format converters (call existing libraries)
│       └── targets.py          # deployment-target converters (NPU/ANE; arg building is unit-testable)
└── tests/test_registry.py      # registry / detection / arg-building / CLI unit tests
```

### 10. Development

```bash
uv sync --extra dev           # install pytest (no conversion backend)
uv run model-translator list
uv run pytest -q              # pytest step matches CI's python-test.yml; CI also validates the list matrix
```

To add a new format / conversion path:

1. Register the new format in `src/model_translator/registry.py` (`register_format`)
2. Implement the converter in `src/model_translator/converters/__init__.py` and register it in `register_all()`
3. Add the corresponding optional dependency group in `pyproject.toml` (the group names in `requires` must match the group keys)
4. New deployment target: register it in `registry.py` with `register_target(...)` (set
   `verified` honestly), implement arg-building (`build_*` pure functions, unit-testable)
   plus the execution function in `src/model_translator/converters/targets.py`, and
   finally `register_target_path(...)` in `register_targets()`
5. Add tests for the new behavior and run `uv run pytest` locally (CI's
   `python-test.yml` runs the same pytest command, then additionally validates the
   format × deployment-target matrix in the `list` output — all 5 target keys must appear)

### 11. Tests

```bash
uv sync --extra dev
uv run pytest -q
```

Tests cover registry/detection logic plus vendor toolchain **argument construction**.
None of them import torch/tensorflow or any NPU SDK, so CI needs no heavy backend.
**All 53 tests currently pass** (`53 passed`, measured on v1.0.20261123):

- `detect_format` adversarial fixtures: magic bytes (GGUF `GGUF`, TFLite `TFL3`), zip
  layout (TorchScript `code/*.py`, PyTorch `data.pkl`, Keras `metadata.json`+`config.json`),
  safetensors header (8-byte little-endian length followed by a JSON metadata object),
  directory markers (`saved_model.pb`, `Manifest.json`)
- Target matrix and routing: all 5 targets registered, `verified` must be false, every
  target has an `install_hint`, direct/two-step routing, no-route returns `None`.
  `test_no_target_claims_verified` **enforces** the `verified=False` of all five
  NPU/ANE targets, so an unproven path can never be claimed as verified.
- Argument construction: `atc` strips the `.om` suffix and uses `--framework=5`, missing
  `soc_version` errors out, rknn disables quantization without a calibration set,
  `cnmodel`/`hb_mapper` arguments, trtexec's min/opt/max shapes
- CLI: `list` shows every target, unknown target exits 2, no route exits 2, missing
  tool exits 3 with acquisition hints, `--dry-run` writes nothing

### 12. FAQ

- **Missing library**: the tool prints the precise on-demand install command (e.g. `uv sync --extra torch --extra onnx`), or add `--auto-install` to install automatically
- **PyTorch->ONNX asks for an example input**: add `--input-shape`, e.g. `--input-shape 1,3,224,224`
- **PyTorch weights are a state_dict and won't load directly**: write a loader script (defining `load_model(path)` returning the model) and pass it with `--script`
- **ONNX->TensorRT says trtexec not found**: install NVIDIA TensorRT separately (`trtexec` is not distributed via pip)
- **GGUF conversion fails**: install llama.cpp and provide `convert_hf_to_gguf.py`
- **`--target` says the tool was not found**: NPU toolchains are not on PyPI — install them as described by `list --target <key>` and retry; rknn-toolkit2 and CANN do not work on Windows, run on x86 Linux
- **`--target` reports wrong arguments**: add `--dry-run` first to see the route and where each parameter comes from, then override with `--opt KEY=VALUE` (e.g. `--opt soc_version=Ascend310P3`)
- **`--target bin` vs GGUF's `.bin`**: Horizon artifacts are never inferred from the extension; pass `--target bin` and `--opt hb_config=...` explicitly
