<div align="center">
<img src="./Logo.png" alt="logo" width="100"/>
<h2>ModelTranslator</h2>
<h3>Multi-format Model Conversion Tool (uv environment)</h3>
</div>


### 1. Introduction

`ModelTranslator` is a **multi-format model conversion tool** managed by [uv](https://docs.astral.sh/uv/), supporting conversion between common deep learning model formats. All conversion work is **delegated to existing mature libraries** (torch / onnx / onnxruntime / tensorflow / tf2onnx / openvino / coremltools / trtexec / llama.cpp, etc.); this project only handles format detection and conversion orchestration, without reinventing the wheel.

- ✅ Supports 10+ common model formats, convertible in any combination
- ✅ Dependencies **pulled on demand**: the CLI core is only a few MB, backends installed as needed
- ✅ `--auto-install` automatically installs missing dependencies
- ✅ Docker-ready, works out of the box

### 2. Quick Start

```bash
# 1. Install (only the lightweight CLI by default)
uv sync

# 2. List supported formats and conversions
uv run model-translator list

# 3. Convert: PyTorch -> ONNX (auto-installs missing deps with --auto-install)
uv run model-translator convert model.pt model.onnx --input-shape 1,3,224,224 --auto-install

# 4. Docker
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
| `mindspore` | MindSpore | `.ckpt` `.mindir` | Huawei MindSpore model |

> Use `uv run model-translator info <path>` to detect the format of any file.

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

### 5. CLI Reference

#### `list`

Lists supported formats and all conversion paths.

```
uv run model-translator list
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
| `--input-shape <N,C,H,W>` | Example input shape, e.g. `1,3,224,224` (required for PyTorch) |
| `--opset <n>` | ONNX opset version (default 13) |
| `--input-names <a,b>` | ONNX input names, comma-separated |
| `--output-names <a,b>` | ONNX output names, comma-separated |
| `--dynamic-axes` | Export dynamic axes (batch dimension) in ONNX |
| `--fp16` | Use FP16 for TensorRT |
| `--quantize int8` | TFLite quantization mode |
| `--script <path>` | Custom Python script (defines `load_model(path)` returning a model) to load PyTorch state_dict |
| `--auto-install` | Auto-install missing dependencies on demand (`uv sync --extra <needed groups>`) |

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

# Auto-install missing deps
uv run model-translator convert model.pt model.onnx --input-shape 1,3,224,224 --auto-install
```

### 6. On-demand Dependencies

The CLI core is the only mandatory dependency (click + filetype, just a few MB); all conversion backends live in the `optional-dependencies` groups of [pyproject.toml](pyproject.toml):

| Group | Packages | Size |
|-------|----------|------|
| `onnx` | onnx / onnxruntime / onnxscript | Medium |
| `torch` | torch | Large (hundreds of MB+) |
| `tensorflow` | tensorflow / tf2onnx | Largest (1GB+) |
| `openvino` | openvino | Medium |
| `coreml` | coremltools (macOS only) | Large |
| `all` | everything above | Very large (not recommended) |

**Two ways to install on demand:**

```bash
# Way 1: install manually
uv sync --extra torch --extra onnx

# Way 2: auto-install during conversion (--auto-install)
uv run model-translator convert model.pt model.onnx --input-shape 1,3,224,224 --auto-install
```

> When `convert` finds a missing dependency, it prints the **precise on-demand install command** (e.g. `uv sync --extra torch --extra onnx`) — you never need to pull all backends at once.

### 7. Docker

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

### 8. Project Structure

```
ModelTranslator/
├── pyproject.toml              # uv project definition + optional dependency groups
├── Dockerfile                  # Docker image (on-demand build)
├── docker-compose.yml          # one-command compose run
├── src/model_translator/
│   ├── cli.py                  # CLI entry (list/info/convert)
│   ├── registry.py             # format registry + conversion path table
│   └── converters/             # converters per format (call existing libraries)
└── tests/test_registry.py      # core logic unit tests
```

### 9. Development

```bash
uv sync --all-extras        # install everything (dev only)
uv run model-translator list
```

To add a new format / conversion path:

1. Register the new format in `src/model_translator/registry.py` (`register_format`)
2. Implement the converter in `src/model_translator/converters/__init__.py` and register it in `register_all()`
3. Add the corresponding optional dependency group in `pyproject.toml` (the group names in `requires` must match the group keys)

### 10. FAQ

- **Missing library**: the tool prints the precise on-demand install command (e.g. `uv sync --extra torch --extra onnx`), or add `--auto-install` to install automatically
- **PyTorch->ONNX asks for an example input**: add `--input-shape`, e.g. `--input-shape 1,3,224,224`
- **PyTorch weights are a state_dict and won't load directly**: write a loader script (defining `load_model(path)` returning the model) and pass it with `--script`
- **ONNX->TensorRT says trtexec not found**: install NVIDIA TensorRT separately (`trtexec` is not distributed via pip)
- **GGUF conversion fails**: install llama.cpp and provide `convert_hf_to_gguf.py`
