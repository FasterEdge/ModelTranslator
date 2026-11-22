# FasterEdge 开源项目 · https://github.com/FasterEdge · https://gitee.com/FasterEdge
"""model-translator 命令行入口。

用法示例:
    model-translator list
    model-translator list --target rknn
    model-translator convert model.pt model.onnx --input-shape 1,3,224,224
    model-translator convert saved_model_dir model.tflite
    model-translator convert yolo.onnx yolo.xml
    model-translator convert yolo.onnx yolo.rknn --target rknn --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from . import __version__
from .converters import ConversionError, MissingDependencyError
from .registry import (
    FORMATS,
    TARGETS,
    detect_format,
    detect_format_with_reason,
    find_conversion,
    format_name,
    list_conversions,
    list_target_paths,
    plan_route,
    resolve_target,
    target_name,
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
    "safetensors": "safetensors",
}

# 部署目标 key 别名
TARGET_ALIASES = {
    "rknn": "rknn",
    "rockchip": "rknn",
    "om": "om",
    "ascend": "om",
    "atc": "om",
    "dlc": "dlc",
    "mlu": "dlc",
    "cambricon": "dlc",
    "bpu": "bin",
    "bin": "bin",
    "horizon": "bin",
    "mlpackage": "mlpackage",
    "ane": "mlpackage",
}


def resolve_key(name: str) -> str:
    key = name.lower()
    return KEY_ALIASES.get(key, key)


def resolve_target_key(name: str) -> str | None:
    """目标名 -> TARGETS key；别名优先，再试扩展名/直接 key。"""
    key = name.lower()
    if key in TARGET_ALIASES:
        return TARGET_ALIASES[key]
    return resolve_target(key)


def _parse_input_shape(raw: str | None) -> tuple[int, ...] | None:
    """解析 --input-shape "1,3,224,224" 为整数元组; 非法输入给出友好错误(而非 raw traceback)。"""
    if not raw:
        return None
    try:
        parts = tuple(int(x.strip()) for x in raw.split(","))
    except ValueError:
        raise click.UsageError(
            f"无效的 --input-shape: {raw!r} (应为逗号分隔的整数, 如 1,3,224,224)"
        )
    if not parts or any(n <= 0 for n in parts):
        raise click.UsageError(
            f"无效的 --input-shape: {raw!r} (维度必须为正整数)"
        )
    return parts


def _parse_opts(pairs: tuple[str, ...]) -> dict[str, str]:
    """解析多个 `--opt k=v` 为字典；无 '=' 或 key 为空时报 UsageError。"""
    out: dict[str, str] = {}
    for item in pairs or ():
        if "=" not in item:
            raise click.UsageError(
                f"无效的 --opt: {item!r} (应为 key=value, 如 --opt soc_version=Ascend310P3)"
            )
        key, _, value = item.partition("=")
        key = key.strip()
        if not key:
            raise click.UsageError(f"无效的 --opt: {item!r} (key 不能为空)")
        out[key] = value.strip()
    return out


@click.group()
@click.version_option(__version__)
def main():
    """多格式模型转换工具（转换委托给已有成熟库）。"""


@main.command("list")
@click.option("--target", "-g", default=None, help="只看某个部署目标（如 rknn/om/dlc/bin/mlpackage）")
def list_formats(target):
    """列出支持的格式与转换路径（默认按 格式 x 部署目标 矩阵输出）。"""
    if target:
        tkey = resolve_target_key(target)
        if tkey is None:
            click.echo(f"错误: 未知部署目标 {target!r}，可用: {', '.join(sorted(TARGETS))}", err=True)
            sys.exit(2)
        _echo_target_matrix(tkey)
        return
    click.echo("支持的文件格式:")
    for fmt in FORMATS.values():
        exts = ", ".join(fmt.extensions) if any(fmt.extensions) else "(目录)"
        click.echo(f"  {fmt.key:<14} {fmt.name:<24} [{exts}]  {fmt.description}")
    click.echo()
    click.echo("支持的转换路径:")
    for c in list_conversions():
        req = f" (需要: {', '.join(c.requires)})" if c.requires else ""
        click.echo(f"  {format_name(c.src_format):<24} -> {format_name(c.dst_format):<24}{req}  {c.description}")
    click.echo()
    click.echo("支持的部署目标 (格式 x 部署目标):")
    click.echo(f"  {'目标':<10}{'厂商':<18}{'工具链':<22}{'输入格式':<10}状态")
    for t in TARGETS.values():
        via = t.via or "(仅 ONNX 直连)"
        status = "已实测" if t.verified else "未实测"
        hint = f" | 依赖: {', '.join(t.requires)}" if t.requires else ""
        click.echo(f"  {t.key:<10}{t.vendor:<18}{t.tool:<22}{via:<10}{status}{hint}")
    click.echo()
    click.echo("各源格式可到达的目标:")
    for p in list_target_paths():
        req = f" (需要: {', '.join(p.requires)})" if p.requires else ""
        click.echo(f"  {format_name(p.src_format):<24} -> {target_name(p.target):<24}{req}  {p.description}")


def _echo_target_matrix(tkey: str) -> None:
    t = TARGETS[tkey]
    status = "已实测" if t.verified else "未实测"
    click.echo(f"部署目标: {t.name} ({t.key})")
    click.echo(f"  厂商:   {t.vendor}")
    click.echo(f"  工具链: {t.tool}")
    click.echo(f"  输入格式: {t.via or '(仅 ONNX 直连)'}")
    click.echo(f"  实测状态: {status}")
    if t.requires:
        click.echo(f"  安装依赖: uv sync {' '.join('--extra ' + g for g in t.requires)}")
    if t.install_hint:
        click.echo(f"  获取方式: {t.install_hint}")
    if t.notes:
        click.echo(f"  备注:   {t.notes}")
    click.echo("  产物扩展名: " + (", ".join(t.extensions) or "(无)"))
    click.echo()
    click.echo("  可达的源格式:")
    paths = [p for p in list_target_paths() if p.target == tkey]
    for p in paths:
        direct = f"{format_name(p.src_format)} -> {t.name}  (直连)"
        click.echo(f"    {direct}")
    via = t.via
    if via:
        for conv in list_conversions():
            if conv.dst_format == via:
                route = plan_route(conv.src_format, tkey)
                if route and len(route) == 2:
                    click.echo(f"    {format_name(conv.src_format):<24} -> {format_name(via)} -> {t.name}")


@main.command("info")
@click.argument("path", type=click.Path(exists=True))
def show_info(path):
    """检测文件/目录对应的模型格式（按内容探测，附带判定依据）。"""
    p = Path(path)
    fmt_key, reason = detect_format_with_reason(p)
    if fmt_key is None:
        click.echo(f"无法识别格式: {path} (判定依据: {reason})")
        sys.exit(1)
    fmt = FORMATS[fmt_key]
    click.echo(f"格式: {fmt.name} ({fmt.key})")
    click.echo(f"说明: {fmt.description}")
    click.echo(f"判定依据: {reason}")
    if p.is_dir():
        click.echo("类型: 目录")


@main.command("convert")
@click.argument("src", type=click.Path(exists=True, path_type=Path))
@click.argument("dst", type=click.Path(path_type=Path))
@click.option("--to", "-t", default=None, help="目标格式 key（默认按输出扩展名推断）")
@click.option("--target", "-g", default=None,
              help="部署目标 key（rknn/om/dlc/bin/mlpackage），按 格式 x 目标 矩阵路由")
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
@click.option("--allow-pickle", default=False, is_flag=True,
              help="允许用 pickle 反序列化完整 PyTorch 模型(不安全, 仅对可信模型使用)")
@click.option("--auto-install", default=False, is_flag=True,
              help="缺少依赖时自动按需安装（uv sync --extra <所需分组>）")
@click.option("--opt", "-o", multiple=True, metavar="KEY=VALUE",
              help="部署目标工具链的透传参数，可重复，如 --opt soc_version=Ascend310P3")
@click.option("--dry-run", default=False, is_flag=True,
              help="只打印将要执行的路由与参数来源，不真正运行")
def convert(src: Path, dst: Path, to, target, input_shape, opset, input_names,
            output_names, dynamic_axes, fp16, quantize, script, allow_pickle,
            auto_install, opt, dry_run):
    """模型转换: model-translator convert <src> <dst>"""
    src_key, src_reason = detect_format_with_reason(src)
    if src_key is None:
        click.echo(f"错误: 无法识别源格式 {src} (判定依据: {src_reason})", err=True)
        sys.exit(2)

    opts = _parse_opts(opt)

    if target:
        _convert_to_target(src, dst, src_key, target, opts, input_shape, opset,
                           input_names, output_names, dynamic_axes, fp16, quantize,
                           script, allow_pickle, auto_install, dry_run)
        return

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

    kwargs = _build_kwargs(input_shape, opset, input_names, output_names,
                           dynamic_axes, fp16, quantize, allow_pickle, script, opts)

    if dry_run:
        click.echo(f"dry-run: {format_name(src_key)} -> {format_name(dst_key)}")
        click.echo(f"  实现: {conv.description}")
        if conv.requires:
            groups = " ".join(f"--extra {g}" for g in conv.requires)
            click.echo(f"  依赖: {', '.join(conv.requires)}  (uv sync {groups})")
        click.echo(f"  将会写入: {dst}")
        return

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


def _build_kwargs(input_shape, opset, input_names, output_names, dynamic_axes,
                  fp16, quantize, allow_pickle, script, opts) -> dict:
    kwargs = {
        "input_shape": _parse_input_shape(input_shape),
        "opset": opset,
        "input_names": input_names.split(",") if input_names else None,
        "output_names": output_names.split(",") if output_names else None,
        "dynamic_axes": dynamic_axes,
        "fp16": fp16,
        "quantize": quantize,
        "allow_pickle": allow_pickle,
        "opts": opts,
    }
    if script:
        kwargs["script"] = script
    return kwargs


def _convert_to_target(src, dst, src_key, target, opts, input_shape, opset,
                        input_names, output_names, dynamic_axes, fp16, quantize,
                        script, allow_pickle, auto_install, dry_run):
    """按"格式 x 部署目标"矩阵路由并逐步执行。

    路由长度 1 表示源格式直连目标；长度 2 表示先转中间格式（通常 ONNX）再落目标，
    中间产物写在 <dst>.step0.onnx 这类文件名上，不会被误认成最终产物。
    """
    tkey = resolve_target_key(target)
    if tkey is None:
        click.echo(f"错误: 未知部署目标 {target!r}，可用: {', '.join(sorted(TARGETS))}",
                   err=True)
        sys.exit(2)
    route = plan_route(src_key, tkey)
    if route is None:
        click.echo(
            f"错误: 没有 {format_name(src_key)} -> {target_name(tkey)} 的路由\n"
            f"用 `model-translator list --target {tkey}` 查看该目标的可用入口",
            err=True,
        )
        sys.exit(2)

    tgt = TARGETS[tkey]
    if not tgt.verified:
        click.echo(
            f"提示: {tgt.name} 路径未在本仓库环境实测（无 {tgt.tool}），"
            f"参数形状按厂商公开文档构造，请以目标环境实跑结果为准",
            err=True,
        )

    step_dsts = [Path(f"{dst}.step{i}.{step.dst}") for i, step in enumerate(route[:-1])]
    step_dsts.append(dst)
    step_src = src
    for i, (step, step_dst) in enumerate(zip(route, step_dsts, strict=True)):
        label = target_name(step.dst) if step.dst in TARGETS else format_name(step.dst)
        kwargs = _build_kwargs(input_shape, opset, input_names, output_names,
                               dynamic_axes, fp16, quantize, allow_pickle, script, opts)
        if dry_run:
            click.echo(f"dry-run 步骤 {i + 1}/{len(route)}: "
                       f"{format_name(step.src)} -> {label}")
            click.echo(f"  实现: {step.description}")
            if step.requires:
                groups = " ".join(f"--extra {g}" for g in step.requires)
                click.echo(f"  依赖: {', '.join(step.requires)}  (uv sync {groups})")
            else:
                hint = TARGETS[step.dst].install_hint if step.dst in TARGETS else ""
                click.echo(f"  依赖: {hint or '(无 pip 分组)'}")
            click.echo(f"  将会写入: {step_dst}")
            step_src = step_dst
            continue

        click.echo(f"步骤 {i + 1}/{len(route)}: {format_name(step.src)} -> {label}")
        try:
            step.fn(step_src, step_dst, **kwargs)
        except MissingDependencyError as e:
            groups = " ".join(f"--extra {g}" for g in step.requires)
            hint = f"  uv sync {groups}" if groups else (
                f"  {TARGETS[step.dst].install_hint}" if step.dst in TARGETS else "")
            click.echo(f"错误: {e}", err=True)
            click.echo(f"请按需安装依赖后重试：\n{hint}", err=True)
            sys.exit(3)
        except ConversionError as e:
            click.echo(f"转换失败: {e}", err=True)
            sys.exit(4)
        except Exception as e:  # noqa: BLE001
            click.echo(f"转换异常: {type(e).__name__}: {e}", err=True)
            sys.exit(4)
        step_src = step_dst

    if dry_run:
        click.echo("dry-run 结束：未执行任何转换")
        return
    size = dst.stat().st_size if dst.exists() else 0
    click.echo(f"完成: {dst} ({_fmt_size(size)})")


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
        project_dir = _find_project_dir()
        if project_dir is not None:
            # 容器/任意 cwd 场景下项目不在当前目录(如镜像 WORKDIR /workspace),
            # 必须显式指定项目根, 否则 uv sync 报 No pyproject.toml found。
            cmd += ["--project", str(project_dir)]
        click.echo(f"缺少依赖，正在按需安装: {' '.join(cmd)} ...")
        import subprocess
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise MissingDependencyError(
                f"按需安装失败: {proc.stderr.strip()[-500:]}"
            )
        click.echo("依赖安装完成，重试转换 ...")
        return conv.fn(src, dst, **kwargs)


def _find_project_dir() -> Path | None:
    """向上查找含 pyproject.toml 的项目根目录。"""
    # 1) 从当前目录向上找
    cur = Path.cwd()
    for p in (cur, *cur.parents):
        if (p / "pyproject.toml").is_file():
            return p
    # 2) 从本模块位置向上找（打包安装场景）
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").is_file():
            return p
    return None


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n} B"


if __name__ == "__main__":
    main()