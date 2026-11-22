# ─────────────────────────────────────────────────────────────
# FasterEdge 开源项目
# Github: https://github.com/FasterEdge
# Gitee:  https://gitee.com/FasterEdge
# ─────────────────────────────────────────────────────────────
"""格式注册表：识别文件格式、登记支持的转换路径与部署目标。

三个维度分开登记，互不冒充：

- ``FORMATS``：格式（扩展名 + 显示名），描述"文件是什么"。
- ``CONVERSIONS``：格式 <-> 格式 的转换边，描述"格式之间怎么互转"。
- ``TARGETS`` / ``TARGET_PATHS``：部署目标（NPU 等）与"某格式能否落到该目标"，
  描述"转换出来能不能跑在目标硬件上"。目标产物扩展名与格式扩展名不一定一一对应
  （如 ``.bin`` 既可能是 GGUF 也可能是地平线 BPU 产物），所以目标自成一维。

``DeployTarget.verified`` 表示该路径**是否在仓库环境里真正跑通**过厂商工具链。
没有厂商 SDK 的路径一律 ``verified=False``，CLI 里以 ``未实测`` 标出，不做成功暗示。
"""

from __future__ import annotations

import sys
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# 转换函数签名：fn(src: Path, dst: Path, **kwargs) -> Path
ConverterFn = Callable[[Path, Path], Path]


@dataclass
class ModelFormat:
    """一种模型格式。"""

    key: str
    name: str
    extensions: tuple[str, ...]
    description: str = ""


@dataclass
class Conversion:
    """一条转换路径：从 src_format 转到 dst_format。"""

    src_format: str
    dst_format: str
    fn: ConverterFn
    # 需要的可选依赖 key（对应 pyproject.optional-dependencies 中的分组名）
    requires: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class DeployTarget:
    """一个部署目标（推理后端 / NPU 平台）。

    requires 非空时表示能通过 ``uv sync --extra <分组>`` 装齐依赖
    （如 coremltools）；厂商 SDK 不在 PyPI 的目标 requires 为空，
    改用 install_hint 说明获取方式。
    """

    key: str
    name: str
    vendor: str
    tool: str
    # 产物扩展名（用于按输出路径推断目标）
    extensions: tuple[str, ...]
    requires: tuple[str, ...] = ()
    install_hint: str = ""
    # 中转格式：该目标的转换器消费哪个中间格式（NPU 工具链普遍吃 ONNX）
    via: str | None = "onnx"
    # 是否在仓库环境里实测跑通（无厂商 SDK 时恒为 False）
    verified: bool = False
    notes: str = ""


@dataclass
class TargetPath:
    """一条"格式 -> 部署目标"的路径声明。"""

    src_format: str
    target: str
    fn: ConverterFn
    requires: tuple[str, ...] = ()
    description: str = ""


# ========== 支持的格式 ==========

FORMATS: dict[str, ModelFormat] = {}


def register_format(key: str, name: str, extensions: tuple[str, ...], description: str = "") -> ModelFormat:
    fmt = ModelFormat(key=key, name=name, extensions=extensions, description=description)
    FORMATS[key] = fmt
    return fmt


# 常用格式
FMT_PYTORCH = register_format("pytorch", "PyTorch", (".pt", ".pth"),
                              "PyTorch 权重/状态字典或完整模型")
FMT_TORCHSCRIPT = register_format("torchscript", "TorchScript", (".pt",),
                                  "TorchScript 序列化模型（需 --dst-key torchscript 区分）")
FMT_ONNX = register_format("onnx", "ONNX", (".onnx",),
                           "Open Neural Network Exchange 标准格式")
FMT_TENSORFLOW = register_format("tensorflow", "TensorFlow SavedModel", (".pb", ""),
                                 "TensorFlow SavedModel（.pb 或目录）")
FMT_KERAS = register_format("keras", "Keras", (".h5", ".keras"),
                            "Keras HDF5 / .keras 模型")
FMT_TFLITE = register_format("tflite", "TensorFlow Lite", (".tflite",),
                             "TFLite 轻量推理格式")
FMT_OPENVINO = register_format("openvino", "OpenVINO IR", (".xml",),
                               "OpenVINO IR（.xml + .bin 成对）")
FMT_TENSORRT = register_format("tensorrt", "TensorRT", (".engine", ".trt"),
                               "TensorRT 引擎文件")
FMT_COREML = register_format("coreml", "CoreML", (".mlmodel", ".mlpackage"),
                             "Apple CoreML 模型")
FMT_GGML = register_format("ggml", "GGML/GGUF", (".gguf", ".bin"),
                           "GGUF 大语言模型格式")
FMT_MINDSPORE = register_format("mindspore", "MindSpore", (".ckpt", ".mindir"),
                                "MindSpore 模型")

# ========== 扩展名 -> 格式 ==========

EXT_TO_FORMAT: dict[str, str] = {}
for _fmt in FORMATS.values():
    # torchscript 与 pytorch 扩展名重叠（都是 .pt），默认按 pytorch 检测，
    # torchscript 作为转换目标通过 --to torchscript 显式指定
    if _fmt.key == "torchscript":
        continue
    for _ext in _fmt.extensions:
        # 空串是目录型 SavedModel 的特殊标记，跳过
        if _ext:
            EXT_TO_FORMAT[_ext] = _fmt.key


# ========== 格式探测 ==========
# 扩展名可以被随意改，所以能读内容就按内容判，判不出来才退回扩展名。

_ZIP_MAGIC = b"PK\x03\x04"
# (偏移, 魔数, 格式 key)
_MAGIC_PROBES: tuple[tuple[int, bytes, str], ...] = (
    (0, b"GGUF", "ggml"),     # GGUF 文件头 4 字节为 "GGUF"
    (4, b"TFL3", "tflite"),   # FlatBuffers file_identifier，TFLite 恒为 "TFL3"
)


def _read_head(path: Path, size: int = 8) -> bytes | None:
    try:
        with path.open("rb") as fp:
            return fp.read(size)
    except OSError:
        return None


def _detect_zip_kind(path: Path) -> str | None:
    """按 zip 目录结构区分"长得都像压缩包"的几种格式。

    - TorchScript：torch.jit.save 产物含 ``<archive>/code/xxxx.py``
    - PyTorch：torch.save 产物含 ``<archive>/data.pkl`` + ``version``
    - Keras 3：``.keras`` 是 zip，根含 ``metadata.json`` 与 ``config.json``
    """
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except (zipfile.BadZipFile, OSError, ValueError):
        return None
    if any("/code/" in n or n.startswith("code/") for n in names):
        return "torchscript"
    if any(n == "data.pkl" or n.endswith("/data.pkl") for n in names):
        return "pytorch"
    if "metadata.json" in names and "config.json" in names:
        return "keras"
    return None


def _detect_dir(path: Path) -> tuple[str | None, str]:
    """目录型模型的探测：先看标志文件，再看不出标志时沿用原有约定。

    原实现"只要是目录就返回 tensorflow"，这对 SavedModel 够用，但会把
    mlpackage / 任意目录也误判。这里先按标志文件识别，剩下的仍回退到
    tensorflow（保持既有行为，不擅自收窄用户可转换的输入）。
    """
    try:
        if (path / "saved_model.pb").is_file() or (path / "saved_model.pbtxt").is_file():
            return "tensorflow", "目录含 saved_model.pb"
        if (path / "Manifest.json").is_file():
            # .mlpackage：Manifest.json + Data/com.apple.CoreML
            return "coreml", "目录含 Manifest.json"
    except OSError:
        return None, "目录不可读"
    return "tensorflow", "目录(无标志文件，按 SavedModel 假定)"


def _detect(path: Path) -> tuple[str | None, str]:
    """返回 (格式 key, 判定依据)，判定依据用于 `info` 命令给出可核对证据。"""
    if path.is_dir():
        return _detect_dir(path)
    if path.is_file():
        head = _read_head(path)
        if head is not None:
            for offset, magic, key in _MAGIC_PROBES:
                if head[offset:offset + len(magic)] == magic:
                    return key, f"魔数 {magic!r} (offset {offset})"
            if head.startswith(_ZIP_MAGIC):
                key = _detect_zip_kind(path)
                if key is not None:
                    return key, "zip 内容结构"
    key = EXT_TO_FORMAT.get(path.suffix.lower())
    if key is not None:
        return key, f"扩展名 {path.suffix.lower()}"
    return None, "无匹配"


def detect_format(path: Path) -> str | None:
    """识别格式 key：先按文件内容（魔数 / zip 结构 / 目录结构），再退回扩展名。"""
    return _detect(path)[0]


def detect_format_with_reason(path: Path) -> tuple[str | None, str]:
    """同 detect_format，额外返回判定依据字符串。"""
    return _detect(path)


# ========== 转换路径注册 ==========

CONVERSIONS: list[Conversion] = []


def register_conversion(src: str, dst: str, fn: ConverterFn, requires=(), description=""):
    CONVERSIONS.append(Conversion(src, dst, fn, tuple(requires), description))


_LOADED = False


def _ensure_conversions_loaded() -> None:
    """确保转换路径与部署目标已注册（延迟导入 converters 模块，且只注册一次）。"""
    global _LOADED
    if _LOADED:
        return
    # 先置位再导入：converters 在导入期会回调 register_*，重入时不能死循环
    _LOADED = True
    from . import converters as _conv
    _conv.register_all(sys.modules[__name__])


def find_conversion(src: str, dst: str) -> Conversion | None:
    _ensure_conversions_loaded()
    for c in CONVERSIONS:
        if c.src_format == src and c.dst_format == dst:
            return c
    return None


def list_conversions() -> list[Conversion]:
    _ensure_conversions_loaded()
    return list(CONVERSIONS)


def format_name(key: str) -> str:
    fmt = FORMATS.get(key)
    return fmt.name if fmt else key


# ========== 部署目标注册 ==========

TARGETS: dict[str, DeployTarget] = {}
# 产物扩展名 -> 目标 key（.bin 歧义太大，不参与自动推断，需显式 --target bin）
TARGET_EXT_TO_TARGET: dict[str, str] = {}


def register_target(
    key: str,
    name: str,
    vendor: str,
    tool: str,
    extensions: tuple[str, ...],
    *,
    requires: tuple[str, ...] = (),
    install_hint: str = "",
    via: str | None = "onnx",
    verified: bool = False,
    notes: str = "",
) -> DeployTarget:
    target = DeployTarget(
        key=key, name=name, vendor=vendor, tool=tool, extensions=extensions,
        requires=requires, install_hint=install_hint, via=via,
        verified=verified, notes=notes,
    )
    TARGETS[key] = target
    return target


TARGET_RKNN = register_target(
    "rknn", "Rockchip RKNN", "瑞芯微 Rockchip", "rknn-toolkit2",
    (".rknn",),
    install_hint="rknn-toolkit2 只随 Rockchip 官方发布（不在 PyPI），"
                 "需从其发布页下载 wheel 后 pip install",
    notes="rknn-toolkit2 仅提供 x86 Linux wheel，Windows/macOS 无法安装",
)
TARGET_OM = register_target(
    "om", "Ascend OM", "华为昇腾 CANN", "atc",
    (".om",),
    install_hint="需安装华为 CANN Toolkit 并把 atc 加入 PATH",
    notes="atc 的 --soc_version 必须与目标芯片一致（如 Ascend310P3）",
)
TARGET_DLC = register_target(
    "dlc", "Cambricon DLC", "寒武纪 MLU", "cnmodel",
    (".dlc",),
    install_hint="需安装寒武纪 CNToolkit 并把模型编译命令行加入 PATH",
    notes="各版 CNToolkit 的命令名/参数差异较大，本仓库按统一入口调用",
)
TARGET_BIN = register_target(
    "bin", "Horizon BPU bin", "地平线 Journey", "hb_mapper",
    (".bin",),
    install_hint="需安装地平线 OpenExplorer 工具链（hb_mapper）并加入 PATH",
    notes="产物扩展名与 GGUF 权重同为 .bin，不会按扩展名自动推断",
)
TARGET_MLPACKAGE = register_target(
    "mlpackage", "CoreML mlpackage (Apple ANE)", "Apple", "coremltools",
    (".mlpackage",),
    requires=("coreml",),
    install_hint="coremltools 仅支持 macOS/Linux 转换，uv sync --extra coreml",
    notes="与本仓库 onnx -> coreml 同一条实现，产物可直接跑在 Apple Neural Engine",
)

TARGET_EXT_TO_TARGET.update({
    ".rknn": "rknn",
    ".om": "om",
    ".dlc": "dlc",
    ".mlpackage": "mlpackage",
})


TARGET_PATHS: list[TargetPath] = []


def register_target_path(src: str, target: str, fn: ConverterFn, requires=(), description=""):
    TARGET_PATHS.append(TargetPath(src, target, fn, tuple(requires), description))


def list_target_paths() -> list[TargetPath]:
    _ensure_conversions_loaded()
    return list(TARGET_PATHS)


def find_target_path(src: str, target: str) -> TargetPath | None:
    _ensure_conversions_loaded()
    for p in TARGET_PATHS:
        if p.src_format == src and p.target == target:
            return p
    return None


def target_name(key: str) -> str:
    t = TARGETS.get(key)
    return t.name if t else key


def resolve_target(name: str) -> str | None:
    """把目标名/别名/产物扩展名归一为 TARGETS 的 key。"""
    key = name.lower()
    if key in TARGETS:
        return key
    if key.startswith("."):
        return TARGET_EXT_TO_TARGET.get(key)
    return TARGET_EXT_TO_TARGET.get("." + key)


@dataclass(frozen=True)
class RouteStep:
    """路由中的一步：src -> dst 的转换调用。"""

    src: str
    dst: str
    fn: ConverterFn
    requires: tuple[str, ...] = ()
    description: str = ""


def plan_route(src_format: str, target: str) -> list[RouteStep] | None:
    """规划"源格式 -> 部署目标"的路由。

    优先直连（源格式有到该目标的目标路径）；否则若该目标声明了中转格式
    （通常 onnx），且源格式 -> 中转格式的转换存在，则返回两步路由。
    无可行路径返回 None。
    """
    direct = find_target_path(src_format, target)
    if direct is not None:
        return [RouteStep(src_format, target, direct.fn, direct.requires, direct.description)]

    tgt = TARGETS.get(target)
    if tgt is None or not tgt.via:
        return None
    first = find_conversion(src_format, tgt.via)
    if first is None:
        return None
    second = find_target_path(tgt.via, target)
    if second is None:
        return None
    return [
        RouteStep(src_format, tgt.via, first.fn, first.requires, first.description),
        RouteStep(tgt.via, target, second.fn, second.requires, second.description),
    ]
