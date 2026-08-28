<div align="center">
<img src="./Logo.png" alt="logo" width="100"/>
<h2>ModelTranslator</h2>
<h3>多格式模型转换工具（uv 环境）</h3>
</div>


### 一、简介

`ModelTranslator` 是一个基于 [uv](https://docs.astral.sh/uv/) 管理的 **多格式模型转换工具**，支持常见深度学习模型格式之间的互转。转换过程**全部委托给已有的成熟库**（torch / onnx / onnxruntime / tensorflow / tf2onnx / openvino / coremltools / trtexec / llama.cpp 等），本项目只负责格式识别与转换编排，不重复造轮子。

支持 Docker 运行，开箱即用。

### 二、支持的格式

| 格式 key | 名称 | 扩展名 | 说明 |
|---------|------|--------|------|
| `pytorch` | PyTorch | `.pt` `.pth` | PyTorch 权重/模型 |
| `torchscript` | TorchScript | `.pt` | TorchScript 序列化模型 |
| `onnx` | ONNX | `.onnx` | 开放神经网络交换格式 |
| `tensorflow` | TensorFlow SavedModel | `.pb` / 目录 | SavedModel 模型 |
| `keras` | Keras | `.h5` `.keras` | Keras HDF5 模型 |
| `tflite` | TensorFlow Lite | `.tflite` | 轻量推理格式 |
| `openvino` | OpenVINO IR | `.xml` | OpenVINO 中间表示 |
| `tensorrt` | TensorRT | `.engine` `.trt` | NVIDIA 推理引擎 |
| `coreml` | CoreML | `.mlmodel` `.mlpackage` | Apple CoreML |
| `ggml` | GGUF | `.gguf` `.bin` | 大语言模型量化格式 |

### 三、支持的转换路径

| 源格式 | 目标格式 | 依赖库 | 说明 |
|--------|---------|--------|------|
| PyTorch | ONNX | torch + onnx | `torch.onnx.export` |
| PyTorch | TorchScript | torch | `torch.jit.trace/script` |
| ONNX | OpenVINO | openvino | `ov.convert_model` |
| ONNX | CoreML | coremltools | `coremltools.convert` |
| ONNX | TensorRT | trtexec | 调用 NVIDIA `trtexec` |
| SavedModel | TFLite | tensorflow | `TFLiteConverter` |
| Keras | TFLite | tensorflow | `TFLiteConverter` |
| SavedModel | ONNX | tf2onnx | `tf2onnx.convert` |
| Keras | ONNX | tf2onnx | `tf2onnx.convert` |
| CoreML | ONNX | coremltools | coremltools 7+ 转 ONNX |
| safetensors | GGUF | llama.cpp | `convert_hf_to_gguf.py` |

### 四、使用方式

#### 方式 A：本地 uv 环境（依赖按需安装）

```bash
cd ModelTranslator

# 基础安装（只装 CLI 本身，很轻量，约几 MB）
uv sync

# 按需安装转换后端 —— 用到哪种转换就装哪个分组
uv sync --extra onnx       # ONNX 相关（onnx / onnxruntime）
uv sync --extra torch      # PyTorch 相关（较重）
uv sync --extra tensorflow # TensorFlow 相关（最重）
uv sync --extra openvino   # OpenVINO 相关
uv sync --extra coreml     # CoreML 相关（仅 macOS）
# 也可以临时拉取全部后端（不推荐，体积很大）
# uv sync --extra all

# 查看支持格式与转换路径
uv run model-translator list

# 检测文件格式
uv run model-translator info model.pt

# 转换：PyTorch -> ONNX（若缺依赖会提示精确的安装命令）
uv run model-translator convert model.pt model.onnx --input-shape 1,3,224,224

# 同一条命令自动按需安装缺失依赖
uv run model-translator convert model.pt model.onnx --input-shape 1,3,224,224 --auto-install

# 转换：PyTorch -> TorchScript
uv run model-translator convert model.pt model_scripted.pt --input-shape 1,3,224,224

# 转换：ONNX -> OpenVINO
uv run model-translator convert model.onnx model.xml

# 转换：ONNX -> CoreML
uv run model-translator convert model.onnx model.mlpackage

# 转换：ONNX -> TensorRT
uv run model-translator convert model.onnx model.engine --fp16

# 转换：SavedModel/Keras -> TFLite
uv run model-translator convert saved_model_dir model.tflite
uv run model-translator convert model.h5 model.tflite --quantize int8

# 转换：SavedModel/Keras -> ONNX
uv run model-translator convert saved_model_dir model.onnx
```

> **依赖按需拉取**：本项目只把 CLI 本身作为必装依赖（约几 MB），所有转换后端（torch / tensorflow / openvino 等，动辄数百 MB 到数 GB）都放在 optional 分组中。运行 `convert` 时若缺依赖，会提示**精确的按需安装命令**（如 `uv sync --extra torch --extra onnx`），或加 `--auto-install` 让工具自动安装该次转换所需的依赖，不会一次性拉取全部。

> PyTorch 的 state_dict 权重需要模型结构：编写一个加载脚本（定义 `def load_model(path) -> nn.Module`），通过 `--script` 传入。

```bash
uv run model-translator convert weights.pth model.onnx \
  --script ./load_my_model.py --input-shape 1,3,224,224
```

#### 方式 B：Docker 运行

```bash
cd ModelTranslator

# 构建镜像
docker build -t model-translator .

# 查看支持格式
docker run --rm -v $(pwd):/workspace model-translator list

# 挂载目录并转换（输入输出均在 /workspace 下）
docker run --rm -v $(pwd):/workspace model-translator \
  convert /workspace/model.pt /workspace/model.onnx --input-shape 1,3,224,224

# 或使用 docker-compose（编辑 compose 文件中的命令）
docker compose run --rm translator \
  convert /workspace/model.pt /workspace/model.onnx --input-shape 1,3,224,224
```

> Windows 下路径挂载：`docker run --rm -v %cd%:/workspace ...`（cmd）或 `-v ${PWD}:/workspace`（PowerShell）。

### 五、Docker 说明

- 基础镜像：`python:3.11-slim` + uv 官方安装脚本
- **依赖按需安装**：默认只装 `onnx` 相关后端（轻量），通过 `--build-arg UV_EXTRAS` 扩展
- 构建示例：

```bash
# 默认（仅 ONNX 相关）
docker build -t model-translator .

# 按需组合
docker build --build-arg 'UV_EXTRAS=onnx openvino' -t model-translator .

# 全部后端（体积很大）
docker build --build-arg 'UV_EXTRAS=all' -t model-translator-full .
```

### 六、开发

```bash
uv sync --all-extras
uv run model-translator list
```

添加新格式/转换路径的步骤：

1. 在 `src/model_translator/registry.py` 注册新格式（`register_format`）
2. 在 `src/model_translator/converters/__init__.py` 实现转换函数并在 `register_all()` 中登记
3. 在 `pyproject.toml` 添加对应的可选依赖分组（注意 `requires` 中的分组名必须与分组 key 一致）

### 七、常见问题

- **没有安装对应库**：运行时提示缺少依赖，会给出精确的按需安装命令（如 `uv sync --extra torch --extra onnx`），或加 `--auto-install` 自动安装
- **PyTorch 转 ONNX 报错需要示例输入**：加 `--input-shape`，例如 `--input-shape 1,3,224,224`
- **ONNX 转 TensorRT 报错找不到 trtexec**：需要另行安装 NVIDIA TensorRT（`trtexec` 不随 pip 分发）
- **GGUF 转换报错**：需要安装 llama.cpp 并提供 `convert_hf_to_gguf.py`