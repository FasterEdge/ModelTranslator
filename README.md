<div align="center">
<img src="./Logo.png" alt="logo" width="100"/>
<h2>ModelTranslator</h2>
<h3>多格式模型转换工具（uv 环境）</h3>
</div>


### 一、简介

`ModelTranslator` 是一个基于 [uv](https://docs.astral.sh/uv/) 管理的 **多格式模型转换工具**，支持常见深度学习模型格式之间的互转。转换过程**全部委托给已有的成熟库**（torch / onnx / onnxruntime / tensorflow / tf2onnx / openvino / coremltools / trtexec / llama.cpp 等），本项目只负责格式识别与转换编排，不重复造轮子。

- ✅ 支持 **12 种常见模型格式、11 条已声明转换路径**（非任意组合互转）；用 `model-translator list` 可查看全部可达路径
- ✅ 支持 **5 个部署目标**（rknn / Ascend OM / Cambricon DLC / 地平线 BPU / Apple ANE）的路径声明与路由
- ✅ 依赖**按需拉取**：CLI 本体仅几 MB，转换后端按需安装
- ✅ 支持 `--auto-install` 自动安装缺失依赖
- ✅ 格式识别按**文件内容**（魔数 / zip 结构 / 目录标志文件），扩展名只作兜底
- ✅ 支持 Docker 运行，开箱即用
- ⚠️ NPU 路径**未在仓库环境实测**（开发机没有厂商 SDK），`list` 输出会明确标注

### 二、快速开始

```bash
# 1. 安装（默认只装 CLI，很轻量）
uv sync

# 2. 查看支持的格式与转换路径（含 格式 x 部署目标 矩阵）
uv run model-translator list

# 3. 只看某个部署目标的入口与获取方式
uv run model-translator list --target rknn

# 4. 转换：PyTorch -> ONNX（缺依赖时自动提示，或加 --auto-install 自动装）
uv run model-translator convert model.pt model.onnx --input-shape 1,3,224,224 --auto-install

# 5. 部署到 NPU 目标（先 dry-run 看路由，再在目标环境实跑）
uv run model-translator convert model.onnx yolo.rknn --target rknn --dry-run

# 6. Docker 方式
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
| `safetensors` | Safetensors | `.safetensors` | HuggingFace 安全权重（GGUF 转换的源） |
| `mindspore` | MindSpore | `.ckpt` `.mindir` | 华为 MindSpore 模型 |

> 用 `uv run model-translator info <path>` 可检测任意文件的格式，并给出**判定依据**
> （魔数 / zip 结构 / 头部结构 / 目录标志文件 / 扩展名兜底），便于核对识别是否正确。
>
> 共 **12** 种格式，但转换边只有 **11** 条：并非任意两种格式之间都能互转，
> 可达组合以 `uv run model-translator list` 的"支持的转换路径"一节为准。

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

> 12 种格式的两两组合有 66 种可能，其中**只有上表 11 条**被声明并注册。
> 未列出的组合（如 ONNX→TFLite、PyTorch→OpenVINO）会直接报"不支持的转换"，
> 不会按扩展名猜测乱跑。查全部可达路径：`uv run model-translator list`。

### 五、部署目标（格式 × 部署目标）

`model-translator list --target <key>` 查看单个目标的入口、工具链、获取方式与实测状态。

| 目标 key | 名称 | 厂商 | 工具链 | 入口格式 | 安装方式 | 实测状态 |
|---------|------|------|--------|---------|---------|---------|
| `rknn` | Rockchip RKNN | 瑞芯微 | rknn-toolkit2 | ONNX | 官方 wheel（不在 PyPI，仅 x86 Linux） | **未实测** |
| `om` | Ascend OM | 华为昇腾 | `atc`（CANN） | ONNX | 安装 CANN Toolkit，`$ASCEND_HOME/bin` 入 PATH | **未实测** |
| `dlc` | Cambricon DLC | 寒武纪 | `cnmodel`（CNToolkit） | ONNX | 安装 CNToolkit，bin 入 PATH | **未实测** |
| `bin` | Horizon BPU bin（地平线 Journey） | 地平线 | `hb_mapper`（OpenExplorer） | ONNX | 安装 OpenExplorer（horizon_tc） | **未实测** |
| `mlpackage` | CoreML mlpackage（Apple ANE） | Apple | coremltools | ONNX | `uv sync --extra coreml` | **未实测**（复用 onnx→coreml 同一实现） |

**为什么标"未实测"**：本仓库的开发环境（Windows）没有 CANN / rknn-toolkit2 /
CNToolkit / OpenExplorer，这些路径**从未在本仓库里跑通**。代码按各厂商公开文档
构造参数，`list` 与 `convert` 的输出会显式标注，不暗示可用性。请在目标环境自行实跑。

> 5 个目标的 `verified` 全部为 `False`，而且这不是"文档口头承诺"：`tests/test_registry.py`
> 里的 `test_no_target_claims_verified` 会断言"任何目标都不得声称已实测"，
> 谁把 `verified` 改成 `True` 而无实跑证据，CI 立刻变红。这条断言是论文里可复现的
> 诚实性指标，不是为了好看。

**路由规则**：源格式若有到该目标的直连路径则走直连；否则先转中间格式（默认 ONNX）
再落目标。例如 `pytorch -> rknn` 会拆成两步，中间产物命名为
`out.rknn.step0.onnx`，不会被误认成最终产物。

```bash
# 看路由（不执行）
uv run model-translator convert model.pt out.rknn --target rknn --input-shape 1,3,224,224 --dry-run

# 昇腾：必须指定目标芯片（atc 不自动推断，写错会生成跑不起来的 .om）
uv run model-translator convert model.onnx out.om --target om --opt soc_version=Ascend310P3

# RKNN：给了校准集才做 INT8 量化，否则明确关闭（不拿随机数据糊弄精度）
uv run model-translator convert model.onnx out.rknn --target rknn \
  --opt input_shape=1,3,224,224 --opt dataset=calib.txt

# 地平线：必须提供 hb_mapper 的 YAML 配置（产物名/输入 shape/校准集都在 yaml 里）
uv run model-translator convert model.onnx out.bin --target bin --opt hb_config=config.yaml
```

`--opt KEY=VALUE` 可重复，用于厂商工具链的透传参数；CLI 自身参数（如 `--input-shape`）
仍然是各格式转换器的输入。

### 六、CLI 命令参考

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
| `--target / -g <key>` | 部署目标 key（`rknn/om/dlc/bin/mlpackage`），按 格式 x 目标 矩阵路由 |
| `--input-shape <N,C,H,W>` | 示例输入 shape，如 `1,3,224,224`（PyTorch 需要） |
| `--opset <n>` | ONNX opset 版本（默认 13） |
| `--input-names <a,b>` | ONNX 输入名，逗号分隔 |
| `--output-names <a,b>` | ONNX 输出名，逗号分隔 |
| `--dynamic-axes` | ONNX 导出动态轴（batch 维度） |
| `--fp16` | TensorRT 使用 FP16 |
| `--quantize int8` | TFLite 量化方式 |
| `--script <path>` | 自定义 Python 脚本（定义 `load_model(path)` 返回模型），用于加载 PyTorch state_dict |
| `--allow-pickle` | **默认关闭**：允许 pickle 反序列化完整 PyTorch 模型（不安全，仅对可信模型使用；不加此开关时 `torch.load(weights_only=True)` 拒绝 pickle，防止恶意 `.pt` 经 `__reduce__` 执行任意代码） |
| `--auto-install` | 缺依赖时自动按需安装（`uv sync --extra <所需分组>`） |
| `--opt KEY=VALUE` | 部署目标工具链的透传参数，可重复（见第五节示例） |
| `--dry-run` | 只打印路由、实现与产物路径，不执行任何转换 |

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

# 部署目标（先 dry-run 看路由）
uv run model-translator convert model.onnx yolo.rknn --target rknn --dry-run

# 缺依赖时自动按需安装
uv run model-translator convert model.pt model.onnx --input-shape 1,3,224,224 --auto-install
```

> 目标格式推断不出时（如 `--target bin` 的地平线 `.bin` 与 GGUF `.bin` 撞扩展名），
> 必须显式给 `--target`，本工具不会按扩展名乱猜。

### 七、依赖按需拉取

本项目把 CLI 本体作为唯一必装依赖（click + filetype，仅几 MB），所有转换后端都放在 [pyproject.toml](pyproject.toml) 的 `optional-dependencies` 分组中：

| 分组 | 包含的库 | 体积 |
|------|---------|------|
| `onnx` | onnx / onnxruntime / onnxscript | 中等 |
| `torch` | torch | 较大（数百 MB+） |
| `tensorflow` | tensorflow / tf2onnx | 最大（1GB+） |
| `openvino` | openvino | 中等 |
| `coreml` | coremltools（仅 macOS） | 较大 |
| `dev` | pytest（测试用，不含任何转换后端） | 很小 |
| `all` | 以上全部 | 非常大（不推荐） |

NPU 工具链（rknn-toolkit2 / CANN atc / CNToolkit / OpenExplorer）**不在 PyPI**，
无法用 extra 分组安装，获取方式见第五节表格与 `model-translator list --target <key>`。

**两种按需用法：**

```bash
# 方式 1：手动按需安装
uv sync --extra torch --extra onnx

# 方式 2：转换时自动安装（--auto-install）
uv run model-translator convert model.pt model.onnx --input-shape 1,3,224,224 --auto-install
```

> 运行 `convert` 时若缺依赖，会提示**精确的按需安装命令**（如 `uv sync --extra torch --extra onnx`），不会要求你一次拉取全部后端。

### 八、Docker 运行

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

### 九、项目结构

```
ModelTranslator/
├── pyproject.toml              # uv 项目定义 + 可选依赖分组（含 dev 测试分组）
├── Dockerfile                  # Docker 镜像（按需构建）
├── docker-compose.yml          # compose 一键运行
├── .github/workflows/
│   ├── docker-image.yml        # 镜像构建
│   ├── release.yml             # Release 打包
│   └── python-test.yml         # pytest + list 矩阵校验（PR/推送门禁）
├── src/model_translator/
│   ├── cli.py                  # 命令行入口（list/info/convert，含 --target 路由）
│   ├── registry.py             # 格式注册表 + 转换路径 + 部署目标/路由
│   └── converters/
│       ├── __init__.py         # 格式->格式转换器（调用已有库）
│       └── targets.py          # 部署目标转换器（NPU/ANE，参数构造可单测）
└── tests/test_registry.py      # 注册表/探测/参数构造/CLI 单元测试
```

### 十、开发

```bash
uv sync --extra dev           # 装 pytest（不含任何转换后端）
uv run model-translator list
uv run pytest -q              # pytest 部分与 CI 的 python-test.yml 一致；CI 另有 list 矩阵校验步骤
```

添加新格式/转换路径的步骤：

1. 在 `src/model_translator/registry.py` 注册新格式（`register_format`）
2. 在 `src/model_translator/converters/__init__.py` 实现转换函数并在 `register_all()` 中登记
3. 在 `pyproject.toml` 添加对应的可选依赖分组（注意 `requires` 中的分组名必须与分组 key 一致）
4. 新增部署目标：在 `registry.py` 用 `register_target(...)` 登记（务必如实设置 `verified`），
   并在 `src/model_translator/converters/targets.py` 实现参数构造（`build_*` 纯函数，
   可单测）+ 执行函数，最后在 `register_targets()` 里 `register_target_path(...)`
5. 为新增行为补测试，本地跑 `uv run pytest`（CI 的 `python-test.yml` 跑同一 pytest 命令，
   并额外校验 `list` 输出的 格式 × 部署目标 矩阵，5 个目标 key 必须都在输出里）

### 十一、测试

```bash
uv sync --extra dev
uv run pytest -q
```

测试分为注册表/探测逻辑与厂商工具链的**参数构造**两类，都不导入 torch/tensorflow
或任何 NPU SDK，因此在 CI 上不需要装重后端。**当前 53 个用例全部通过**
（`53 passed`，实测于本仓库当前版本 v1.0.20261123）。

- `detect_format` 冲突构造：魔数（GGUF `GGUF`、TFLite `TFL3`）、zip 内容结构
  （TorchScript `code/*.py`、PyTorch `data.pkl`、Keras `metadata.json`+`config.json`）、
  safetensors 头部结构（8 字节小端长度 + 紧随的 JSON 元数据对象）、
  目录标志文件（`saved_model.pb`、`Manifest.json`）
- 目标矩阵与路由：5 个目标全部注册、`verified` 不得为真、每个目标有 `install_hint`、
  直连/两步路由、无路由返回 `None`。其中 `test_no_target_claims_verified`
  断言"任何目标都不得声称已实测"——5 个 NPU/ANE 目标的 `verified=False` 由测试
  **强制锁定**，没有实跑证据就不许虚报；这是论文里可直接引用的可度量指标。
- 参数构造：`atc` 剥 `.om` 后缀且 `--framework=5`、缺 `soc_version` 报错、
  rknn 无校准集时关闭量化、`cnmodel`/`hb_mapper` 参数、`trtexec` 的 shapes 三档
- CLI：`list` 输出含全部目标、`--target` 未知目标退出码 2、无路由退出码 2、
  缺工具退出码 3 且带获取方式、`--dry-run` 不落盘

### 十二、常见问题

- **没有安装对应库**：运行时提示缺少依赖，会给出精确的按需安装命令（如 `uv sync --extra torch --extra onnx`），或加 `--auto-install` 自动安装
- **PyTorch 转 ONNX 报错需要示例输入**：加 `--input-shape`，例如 `--input-shape 1,3,224,224`
- **PyTorch 权重是 state_dict 无法直接加载**：编写加载脚本（定义 `load_model(path)` 返回模型），用 `--script` 传入
- **ONNX 转 TensorRT 报错找不到 trtexec**：需要另行安装 NVIDIA TensorRT（`trtexec` 不随 pip 分发）
- **GGUF 转换报错**：需要安装 llama.cpp 并提供 `convert_hf_to_gguf.py`
- **`--target` 报找不到工具**：NPU 工具链不在 PyPI，按 `list --target <key>` 打印的"获取方式"安装后重试；Windows 上 rknn-toolkit2 与 CANN 均不可用，需在 x86 Linux 环境执行
- **`--target` 报参数不对**：先加 `--dry-run` 看拼出的路由与参数来源，再用 `--opt KEY=VALUE` 覆盖（如 `--opt soc_version=Ascend310P3`）
- **`--target bin` 与 GGUF 的 `.bin` 混淆**：地平线产物不会按扩展名自动推断，必须显式 `--target bin` 且提供 `--opt hb_config=...`
