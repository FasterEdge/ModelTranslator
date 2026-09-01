# ─────────────────────────────────────────────────────────────
# FasterEdge 开源项目
# Github: https://github.com/FasterEdge
# Gitee:  https://gitee.com/FasterEdge
# ─────────────────────────────────────────────────────────────
"""格式注册表：识别文件格式、登记支持的转换路径。

每种格式对应：
- extensions: 识别该格式的文件扩展名
- name: 格式显示名
- description: 一句话说明
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

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


def detect_format(path: Path) -> Optional[str]:
    """根据路径/扩展名识别格式 key。目录按 SavedModel 处理。"""
    if path.is_dir():
        return "tensorflow"
    ext = path.suffix.lower()
    return EXT_TO_FORMAT.get(ext)


# ========== 转换路径注册 ==========

CONVERSIONS: list[Conversion] = []


def register_conversion(src: str, dst: str, fn: ConverterFn, requires=(), description=""):
    CONVERSIONS.append(Conversion(src, dst, fn, tuple(requires), description))


def _ensure_conversions_loaded() -> None:
    """确保转换路径已注册（延迟导入 converters 模块）。"""
    if not CONVERSIONS:
        from . import converters as _conv
        _conv.register_all(sys.modules[__name__])


def find_conversion(src: str, dst: str) -> Optional[Conversion]:
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
