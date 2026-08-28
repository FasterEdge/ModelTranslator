<div align="center">
<img src="./Logo.png" alt="logo" width="100"/>
<h2>ModelTranslator</h2>
<h3>多格式模型转换工具（uv 环境）</h3>
</div>


### 一、简介

`ModelTranslator` 是一个基于 [uv](https://docs.astral.sh/uv/) 管理的 **多格式模型转换工具**，支持常见深度学习模型格式之间的互转。转换过程**全部委托给已有的成熟库**（torch / onnx / onnxruntime / tensorflow / tf2onnx / openvino / coremltools / trtexec / llama.cpp 等），本项目只负责格式识别与转换编排，不重复造轮子。

- ✅ 支持 10+ 常见模型格式，任意组合互转
- ✅ 依赖**按需拉取**：CLI 本体仅几 MB，转换后端按需安装
- ✅ 支持 `--auto-install` 自动安装缺失依赖
- ✅ 支持 Docker 运行，开箱即用

### 二、快速开始

```bash
# 1. 安装（默认只装 CLI，很轻量）
uv sync

# 2. 查看支持格式与转换路径
uv run model-translator list

# 3. 转换：PyTorch -> ONNX（缺依赖时自动提示，或加 --auto-install 自动装）
uv run model-translator convert model.pt model.onnx --input-shape 1,3,224,224 --auto-install

# 4. Docker 方式
docker build -t model-translator .
docker run --rm -v $(pwd):/workspace model-translator list
```

### 三、支持的格式

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
| `mindspore` | MindSpore | `.ckpt` `.mindir` | 华为 MindSpore 模型 |

> 用 `uv run model-translator info <path>` 可检测任意文件的格式。

### 四、支持的转换路径

| 源格式 | 目标格式 | 依赖库 | 底层实现 |
|--------|---------|--------|---------|
| PyTorch | ONNX | torch + onnx | `torch.onnx.export` |
| PyTorch | TorchScript | torch | `torch.jit.trace/script` |
| ONNX | OpenVINO | openvino | `ov.convert_model` |
| ONNX | CoreML | coremltools | `coremltools.convert` |
| ONNX | TensorRT | trtexec | 调用 NVIDIA `trtexec` |
| SavedModel | TFLite | tensorflow | `TFLiteConverter` |
| Keras | TFLite | tensorflow | `TFLiteConverter` |
| SavedModel | ONNX | tensorflow + tf2onnx | `tf2onnx.convert` |
| Keras | ONNX | tensorflow + tf2onnx | `tf2onnx.convert` |
| CoreML | ONNX | coremltools | coremltools 7+ 转 ONNX |
| safetensors | GGUF | llama.cpp | `convert_hf_to_gguf.py` |

### 五、CLI 命令参考

#### `list`

列出支持的格式与全部转换路径。

```
uv run model-translator list
```

#### `info`

检测文件/目录对应的模型格式。

```
uv run model-translator info <path>
```

#### `convert`

模型转换，是核心命令。

```
uv run model-translator convert <src> <dst> [选项]
```

| 选项 | 说明 |
|------|------|
| `--to / -t <key>` | 目标格式 key（默认按输出扩展名推断） |
| `--input-shape <N,C,H,W>` | 示例输入 shape，如 `1,3,224,224`（PyTorch 需要） |
| `--opset <n>` | ONNX opset 版本（默认 13） |
| `--input-names <a,b>` | ONNX 输入名，逗号分隔 |
| `--output-names <a,b>` | ONNX 输出名，逗号分隔 |
| `--dynamic-axes` | ONNX 导出动态轴（batch 维度） |
| `--fp16` | TensorRT 使用 FP16 |
| `--quantize int8` | TFLite 量化方式 |
| `--script <path>` | 自定义 Python 脚本（定义 `load_model(path)` 返回模型），用于加载 PyTorch state_dict |
| `--auto-install` | 缺依赖时自动按需安装（`uv sync --extra <所需分组>`） |

**示例：**

```bash
# PyTorch -> ONNX
uv run model-translator convert model.pt model.onnx --input-shape 1,3,224,224

# PyTorch -> TorchScript
uv run model-translator convert model.pt model_scripted.pt --input-shape 1,3,224,224

# PyTorch(state_dict) -> ONNX，用脚本恢复模型结构
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

# 缺依赖时自动按需安装
uv run model-translator convert model.pt model.onnx --input-shape 1,3,224,224 --auto-install
```

### 六、依赖按需拉取

本项目把 CLI 本体作为唯一必装依赖（click + filetype，仅几 MB），所有转换后端都放在 [pyproject.toml](pyproject.toml) 的 `optional-dependencies` 分组中：

| 分组 | 包含的库 | 体积 |
|------|---------|------|
| `onnx` | onnx / onnxruntime / onnxscript | 中等 |
| `torch` | torch | 较大（数百 MB+） |
| `tensorflow` | tensorflow / tf2onnx | 最大（1GB+） |
| `openvino` | openvino | 中等 |
| `coreml` | coremltools（仅 macOS） | 较大 |
| `all` | 以上全部 | 非常大（不推荐） |

**两种按需用法：**

```bash
# 方式 1：手动按需安装
uv sync --extra torch --extra onnx

# 方式 2：转换时自动安装（--auto-install）
uv run model-translator convert model.pt model.onnx --input-shape 1,3,224,224 --auto-install
```

> 运行 `convert` 时若缺依赖，会提示**精确的按需安装命令**（如 `uv sync --extra torch --extra onnx`），不会要求你一次拉取全部后端。

### 七、Docker 运行

```bash
cd ModelTranslator

# 构建镜像（默认只装 onnx 后端，轻量）
docker build -t model-translator .

# 按需组合后端
docker build --build-arg 'UV_EXTRAS=onnx openvino' -t model-translator .

# 全部后端（体积很大，一般不需要）
docker build --build-arg 'UV_EXTRAS=all' -t model-translator-full .

# 查看支持格式
docker run --rm -v $(pwd):/workspace model-translator list

# 挂载目录并转换（输入输出均在 /workspace 下）
docker run --rm -v $(pwd):/workspace model-translator \
  convert /workspace/model.pt /workspace/model.onnx --input-shape 1,3,224,224

# 或使用 docker-compose
docker compose run --rm translator \
  convert /workspace/model.pt /workspace/model.onnx --input-shape 1,3,224,224
```

> Windows 下路径挂载：`docker run --rm -v %cd%:/workspace ...`（cmd）或 `-v ${PWD}:/workspace`（PowerShell）。

#### Dockerfile 说明

- 基础镜像：`ghcr.io/astral-sh/uv:python3.11-bookworm-slim`（uv 官方镜像，自带 uv）
- 通过 `--build-arg UV_EXTRAS` 按需选择要安装的依赖分组（空格分隔多个）
- 工作目录 `/workspace`，挂载模型输入输出即可

### 八、项目结构

```
ModelTranslator/
├── pyproject.toml              # uv 项目定义 + 可选依赖分组
├── Dockerfile                  # Docker 镜像（按需构建）
├── docker-compose.yml          # compose 一键运行
├── src/model_translator/
│   ├── cli.py                  # 命令行入口（list/info/convert）
│   ├── registry.py             # 格式注册表 + 转换路径登记
│   └── converters/             # 各格式转换器（调用已有库）
└── tests/test_registry.py      # 基础逻辑单元测试
```

### 九、开发

```bash
uv sync --all-extras        # 安装全部依赖（仅开发时需要）
uv run model-translator list
```

添加新格式/转换路径的步骤：

1. 在 `src/model_translator/registry.py` 注册新格式（`register_format`）
2. 在 `src/model_translator/converters/__init__.py` 实现转换函数并在 `register_all()` 中登记
3. 在 `pyproject.toml` 添加对应的可选依赖分组（注意 `requires` 中的分组名必须与分组 key 一致）

### 十、常见问题

- **没有安装对应库**：运行时提示缺少依赖，会给出精确的按需安装命令（如 `uv sync --extra torch --extra onnx`），或加 `--auto-install` 自动安装
- **PyTorch 转 ONNX 报错需要示例输入**：加 `--input-shape`，例如 `--input-shape 1,3,224,224`
- **PyTorch 权重是 state_dict 无法直接加载**：编写加载脚本（定义 `load_model(path)` 返回模型），用 `--script` 传入
- **ONNX 转 TensorRT 报错找不到 trtexec**：需要另行安装 NVIDIA TensorRT（`trtexec` 不随 pip 分发）
- **GGUF 转换报错**：需要安装 llama.cpp 并提供 `convert_hf_to_gguf.py`
