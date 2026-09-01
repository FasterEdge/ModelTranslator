# ─────────────────────────────────────────────────────────────
# FasterEdge 开源项目
# Github: https://github.com/FasterEdge
# Gitee:  https://gitee.com/FasterEdge
# ─────────────────────────────────────────────────────────────
"""转换器实现：每个转换委托给已有的成熟库。

约定：
- 每个函数签名为 convert_xxx(src: Path, dst: Path, **kwargs) -> Path
- 库缺失时抛出 MissingDependencyError，提示用户安装对应 optional 依赖
- 输出目录若不存在会自动创建
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class MissingDependencyError(RuntimeError):
    """缺少转换所需的第三方库。"""


class ConversionError(RuntimeError):
    """转换过程失败。"""


def _ensure_parent(dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)


def _import(name: str, pip_name: str | None = None):
    """延迟导入并给出友好错误。"""
    try:
        return __import__(name)
    except ImportError as e:
        hint = pip_name or name
        raise MissingDependencyError(
            f"缺少依赖 `{name}`（{hint}），该转换需要对应的 optional 依赖分组"
        ) from e


# ================= PyTorch -> ONNX / TorchScript =================


def convert_pytorch_to_onnx(src: Path, dst: Path, **kwargs) -> Path:
    torch = _import("torch")
    _ensure_parent(dst)
    # 加载模型：兼容完整模型 / state_dict / 自定义脚本三种
    model = _load_pytorch_model(torch, src, kwargs)
    # 动态输入：需要用户提供示例输入 shape
    input_shape = kwargs.get("input_shape") or kwargs.get("shape")
    opset = int(kwargs.get("opset", 13))
    if input_shape is None:
        raise ConversionError(
            "PyTorch->ONNX 需要示例输入，请通过 --input-shape 指定，"
            "例如 --input-shape '1,3,224,224'"
        )
    dummy = torch.randn(*[int(x) for x in str(input_shape).split(",")])
    model.eval()
    torch.onnx.export(
        model, dummy, str(dst),
        input_names=kwargs.get("input_names"),
        output_names=kwargs.get("output_names"),
        dynamic_axes=kwargs.get("dynamic_axes") or None,
        opset_version=opset,
    )
    return dst


def convert_pytorch_to_torchscript(src: Path, dst: Path, **kwargs) -> Path:
    torch = _import("torch")
    _ensure_parent(dst)
    model = _load_pytorch_model(torch, src, kwargs)
    model.eval()
    input_shape = kwargs.get("input_shape") or kwargs.get("shape")
    if input_shape is not None:
        dummy = torch.randn(*[int(x) for x in str(input_shape).split(",")])
        traced = torch.jit.trace(model, dummy)
        traced.save(str(dst))
    else:
        scripted = torch.jit.script(model)
        scripted.save(str(dst))
    return dst


def _load_pytorch_model(torch, src: Path, kwargs):
    """加载 PyTorch 模型。

    优先级：
    1. --script 提供的自定义脚本（应定义 load_model(path) -> nn.Module）
    2. 完整模型文件（torch.load 直接得到 module）
    3. state_dict -> 需要模型结构，引导用户用 --script
    """
    script = kwargs.get("script")
    if script is not None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("user_model_loader", str(script))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        loader = getattr(mod, "load_model", None)
        if loader is None:
            raise ConversionError("--script 脚本必须定义 load_model(path) 函数")
        return loader(str(src))

    ckpt = torch.load(str(src), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model" in ckpt):
        raise ConversionError(
            "检测到 state_dict 权重，需要模型结构定义。"
            "请编写一个 Python 脚本并通过 --script 提供，"
            "脚本中定义 def load_model(path) -> nn.Module 返回实例化好的模型。"
        )
    # 直接当作完整模型（module）加载
    return ckpt


# ================= ONNX -> OpenVINO / CoreML / TensorRT =================


def convert_onnx_to_openvino(src: Path, dst: Path, **kwargs) -> Path:
    ov = _import("openvino", "openvino")
    _ensure_parent(dst)
    model = ov.convert_model(str(src))
    ov.save_model(model, str(dst))
    return dst


def convert_onnx_to_coreml(src: Path, dst: Path, **kwargs) -> Path:
    ct = _import("coremltools", "coremltools")
    _ensure_parent(dst)
    mlmodel = ct.convert(str(src))
    mlmodel.save(str(dst))
    return dst


def convert_onnx_to_tensorrt(src: Path, dst: Path, **kwargs) -> Path:
    """ONNX -> TensorRT，调用系统 trtexec 工具。"""
    _ensure_parent(dst)
    trtexec = shutil.which("trtexec") or shutil.which("trtexec.exe")
    if not trtexec:
        raise MissingDependencyError(
            "未找到 trtexec，请安装 TensorRT（NVIDIA 官方）并加入 PATH"
        )
    cmd = [trtexec, f"--onnx={src}", f"--saveEngine={dst}"]
    if kwargs.get("fp16"):
        cmd.append("--fp16")
    if kwargs.get("int8"):
        cmd.append("--int8")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ConversionError(f"trtexec 失败: {result.stderr[-2000:]}")
    return dst


# ================= TensorFlow / Keras -> TFLite / ONNX =================


def convert_tf_to_tflite(src: Path, dst: Path, **kwargs) -> Path:
    _import("tensorflow", "tensorflow")
    import tensorflow as tf
    _ensure_parent(dst)

    def _convert(converter):
        if kwargs.get("quantize") == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        dst.write_bytes(tflite_model)
        return dst

    if src.is_dir():
        return _convert(tf.lite.TFLiteConverter.from_saved_model(str(src)))
    # .h5 / .keras
    model = tf.keras.models.load_model(str(src))
    return _convert(tf.lite.TFLiteConverter.from_keras_model(model))


def convert_tf_to_onnx(src: Path, dst: Path, **kwargs) -> Path:
    _import("tf2onnx", "tf2onnx")
    _import("tensorflow", "tensorflow")
    import tensorflow as tf
    _ensure_parent(dst)
    if src.is_dir():
        # SavedModel 目录 -> ONNX
        import tf2onnx.convert
        tf2onnx.convert.from_saved_model(str(src), output_path=str(dst))
    else:
        model = tf.keras.models.load_model(str(src))
        import tf2onnx.convert
        tf2onnx.convert.from_keras(model, output_path=str(dst))
    return dst


# ================= CoreML -> ONNX =================


def convert_coreml_to_onnx(src: Path, dst: Path, **kwargs) -> Path:
    _import("coremltools", "coremltools")
    _ensure_parent(dst)
    import coremltools as ct
    from coremltools.converters.mil.mil.ops.defs import (  # noqa: F401
        iOS15 as _,
    )
    # coremltools 7+ 提供 mlprogram 转 onnx
    mlmodel = ct.models.MLModel(str(src))
    model = mlmodel.convert_to("onnx", convert_to="onnx")
    model.save(str(dst))
    return dst


# ================= GGML/GGUF（LLM 权重格式） =================


def convert_safetensors_to_gguf(src: Path, dst: Path, **kwargs) -> Path:
    """safetensors -> GGUF，使用 llama.cpp 的 convert_hf_to_gguf.py。"""
    _ensure_parent(dst)
    script = shutil.which("convert_hf_to_gguf.py") or shutil.which("convert-hf-to-gguf")
    if not script:
        raise MissingDependencyError(
            "未找到 convert_hf_to_gguf.py，请安装 llama.cpp 并加入 PATH"
        )
    cmd = ["python3", script, str(src), "--outfile", str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ConversionError(f"GGUF 转换失败: {result.stderr[-2000:]}")
    return dst


# ================= 注册转换路径 =================


def register_all(reg) -> None:
    """把全部转换路径注册到 registry 模块（reg 为 model_translator.registry）。

    以参数传入避免模块间循环导入。
    requires 元组中的元素必须与 pyproject.toml 的 optional-dependencies 分组名一致，
    便于按需安装：uv sync --extra <name>。
    """
    reg.register_conversion("pytorch", "onnx", convert_pytorch_to_onnx,
                            requires=("torch", "onnx"), description="PyTorch -> ONNX")
    reg.register_conversion("pytorch", "torchscript", convert_pytorch_to_torchscript,
                            requires=("torch",), description="PyTorch -> TorchScript")
    reg.register_conversion("onnx", "openvino", convert_onnx_to_openvino,
                            requires=("onnx", "openvino"), description="ONNX -> OpenVINO IR")
    reg.register_conversion("onnx", "coreml", convert_onnx_to_coreml,
                            requires=("onnx", "coreml"), description="ONNX -> CoreML")
    reg.register_conversion("onnx", "tensorrt", convert_onnx_to_tensorrt,
                            requires=("onnx",), description="ONNX -> TensorRT (trtexec)")
    reg.register_conversion("tensorflow", "tflite", convert_tf_to_tflite,
                            requires=("tensorflow",), description="SavedModel -> TFLite")
    reg.register_conversion("keras", "tflite", convert_tf_to_tflite,
                            requires=("tensorflow",), description="Keras -> TFLite")
    reg.register_conversion("tensorflow", "onnx", convert_tf_to_onnx,
                            requires=("tensorflow",), description="SavedModel -> ONNX")
    reg.register_conversion("keras", "onnx", convert_tf_to_onnx,
                            requires=("tensorflow",), description="Keras -> ONNX")
    reg.register_conversion("coreml", "onnx", convert_coreml_to_onnx,
                            requires=("coreml",), description="CoreML -> ONNX")
    reg.register_conversion("ggml", "ggml", convert_safetensors_to_gguf,
                            requires=(), description="safetensors -> GGUF (llama.cpp)")
