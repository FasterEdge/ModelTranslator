"""model-translator 命令行入口。

用法示例:
    model-translator list
    model-translator convert model.pt model.onnx --input-shape 1,3,224,224
    model-translator convert saved_model_dir model.tflite
    model-translator convert yolo.onnx yolo.xml
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from . import __version__
from .converters import ConversionError, MissingDependencyError
from .registry import (
    FORMATS,
    detect_format,
    find_conversion,
    format_name,
    list_conversions,
)


# 常见格式 key 别名（便于 --to 参数）
KEY_ALIASES = {
    "pt": "pytorch",
    "pth": "pytorch",
    "torch": "pytorch",
    "onnx": "onnx",
    "tf": "tensorflow",
    "savedmodel": "tensorflow",
    "keras": "keras",
    "h5": "keras",
    "tflite": "tflite",
    "openvino": "openvino",
    "ov": "openvino",
    "tensorrt": "tensorrt",
    "trt": "tensorrt",
    "coreml": "coreml",
    "gguf": "ggml",
    "ggml": "ggml",
}


def resolve_key(name: str) -> str:
    key = name.lower()
    return KEY_ALIASES.get(key, key)


@click.group()
@click.version_option(__version__)
def main():
    """多格式模型转换工具（转换委托给已有成熟库）。"""


@main.command("list")
def list_formats():
    """列出支持的格式与转换路径。"""
    click.echo("支持的文件格式:")
    for fmt in FORMATS.values():
        exts = ", ".join(fmt.extensions) if any(fmt.extensions) else "(目录)"
        click.echo(f"  {fmt.key:<14} {fmt.name:<24} [{exts}]  {fmt.description}")
    click.echo()
    click.echo("支持的转换路径:")
    for c in list_conversions():
        req = f" (需要: {', '.join(c.requires)})" if c.requires else ""
        click.echo(f"  {format_name(c.src_format):<24} -> {format_name(c.dst_format):<24}{req}  {c.description}")


@main.command("info")
@click.argument("path", type=click.Path(exists=True))
def show_info(path):
    """检测文件/目录对应的模型格式。"""
    p = Path(path)
    fmt_key = detect_format(p)
    if fmt_key is None:
        click.echo(f"无法识别格式: {path}")
        sys.exit(1)
    fmt = FORMATS[fmt_key]
    click.echo(f"格式: {fmt.name} ({fmt.key})")
    click.echo(f"说明: {fmt.description}")
    if p.is_dir():
        click.echo("类型: 目录 (TensorFlow SavedModel)")


@main.command("convert")
@click.argument("src", type=click.Path(exists=True, path_type=Path))
@click.argument("dst", type=click.Path(path_type=Path))
@click.option("--to", "-t", default=None, help="目标格式 key（默认按输出扩展名推断）")
@click.option("--input-shape", default=None, help="示例输入 shape，如 1,3,224,224（PyTorch 需要）")
@click.option("--opset", default=13, type=int, help="ONNX opset 版本（默认 13）")
@click.option("--input-names", default=None, help="ONNX 输入名，逗号分隔")
@click.option("--output-names", default=None, help="ONNX 输出名，逗号分隔")
@click.option("--dynamic-axes", default=False, is_flag=True, help="ONNX 导出动态轴（batch 维度）")
@click.option("--fp16", default=False, is_flag=True, help="TensorRT 使用 FP16")
@click.option("--quantize", default=None, type=click.Choice(["int8"]),
              help="TFLite 量化方式")
@click.option("--script", default=None, type=click.Path(exists=True, path_type=Path),
              help="自定义 Python 脚本，用于加载 PyTorch 模型（定义 load_model() 返回模型）")
@click.option("--auto-install", default=False, is_flag=True,
              help="缺少依赖时自动按需安装（uv sync --extra <所需分组>）")
def convert(src: Path, dst: Path, to, input_shape, opset, input_names,
            output_names, dynamic_axes, fp16, quantize, script, auto_install):
    """模型转换: model-translator convert <src> <dst>"""
    src_key = detect_format(src)
    if src_key is None:
        click.echo(f"错误: 无法识别源格式 {src}", err=True)
        sys.exit(2)

    if to:
        dst_key = resolve_key(to)
    else:
        dst_key = detect_format(dst)
        if dst_key is None:
            click.echo(f"错误: 无法识别目标格式 {dst}，请用 --to 指定", err=True)
            sys.exit(2)

    conv = find_conversion(src_key, dst_key)
    if conv is None:
        click.echo(
            f"错误: 不支持的转换 {format_name(src_key)} -> {format_name(dst_key)}\n"
            f"用 `model-translator list` 查看支持的转换路径",
            err=True,
        )
        sys.exit(2)

    kwargs = {
        "input_shape": input_shape,
        "opset": opset,
        "input_names": input_names.split(",") if input_names else None,
        "output_names": output_names.split(",") if output_names else None,
        "dynamic_axes": dynamic_axes,
        "fp16": fp16,
        "quantize": quantize,
    }
    if script:
        kwargs["script"] = script

    click.echo(f"转换: {src_key} -> {dst_key}")
    click.echo(f"  {src} -> {dst}")
    try:
        result = _run_with_auto_install(conv, src, dst, kwargs, auto_install)
    except MissingDependencyError as e:
        # 按需安装提示：根据转换路径的 requires（即 pyproject 的 extra 分组）给出精确命令
        groups = " ".join(f"--extra {g}" for g in conv.requires)
        hint = f"  uv sync {groups}" if groups else "  (无对应的 pip 依赖)"
        click.echo(f"错误: {e}", err=True)
        click.echo(f"请按需安装依赖后重试：\n{hint}\n"
                   f"或直接加 --auto-install 自动安装", err=True)
        sys.exit(3)
    except ConversionError as e:
        click.echo(f"转换失败: {e}", err=True)
        sys.exit(4)
    except Exception as e:  # noqa: BLE001
        click.echo(f"转换异常: {type(e).__name__}: {e}", err=True)
        sys.exit(4)

    size = result.stat().st_size if result.exists() else 0
    click.echo(f"完成: {result} ({_fmt_size(size)})")


def _run_with_auto_install(conv, src, dst, kwargs, auto_install):
    """执行转换；若缺依赖且开了 --auto-install，按需安装后重试。"""
    if not auto_install:
        return conv.fn(src, dst, **kwargs)
    try:
        return conv.fn(src, dst, **kwargs)
    except MissingDependencyError:
        groups = list(conv.requires)
        if not groups:
            raise
        cmd = ["uv", "sync"] + [a for g in groups for a in ("--extra", g)]
        click.echo(f"缺少依赖，正在按需安装: {' '.join(cmd)} ...")
        import subprocess
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise MissingDependencyError(
                f"按需安装失败: {proc.stderr.strip()[-500:]}"
            )
        click.echo("依赖安装完成，重试转换 ...")
        return conv.fn(src, dst, **kwargs)


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n} B"


if __name__ == "__main__":
    main()