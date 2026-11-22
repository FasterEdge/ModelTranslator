# ─────────────────────────────────────────────────────────────
# FasterEdge 开源项目
# Github: https://github.com/FasterEdge
# Gitee:  https://gitee.com/FasterEdge
# ─────────────────────────────────────────────────────────────
"""部署目标（NPU / 专用推理后端）转换器。

与 `converters/__init__.py` 的分工：
- 本模块只负责"把中间格式（通常 ONNX）交给厂商工具链"，不做模型改写。
- 参数构造（`build_*_args` / `build_*_kwargs`）是**纯函数**，不导入厂商 SDK、
  不起子进程，因此可以在没装任何 NPU 工具链的机器上单测参数拼装逻辑。
- 执行函数先构造参数、再查工具是否存在、最后跑子进程；工具缺失抛
  MissingDependencyError（带 `uv sync --extra <分组>` 或厂商 SDK 获取方式），
  工具返回非 0 或退出码为 0 但没产出文件都抛 ConversionError。

诚实声明：本仓库的开发环境（Windows + 无 CANN / 无 rknn-toolkit2 /
无 CNToolkit / 无 OpenExplorer）**没有实测**这些路径。注册表里
`DeployTarget.verified` 一律为 False，CLI 会标注"未实测"。参数形状按各厂商
公开文档构造；升级工具链后若参数改名，用 `--opt` 覆盖或改本文件。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from . import ConversionError, MissingDependencyError, _ensure_parent

# 厂商工具链里"实际被调用的可执行文件名"，与 registry.TARGETS[key].tool 必须一致
TOOL_RKNN_API = "rknn-toolkit2"   # 纯 Python API，无命令行
TOOL_ATC = "atc"
TOOL_CNMODEL = "cnmodel"
TOOL_HB_MAPPER = "hb_mapper"

INSTALL_HINT_RKNN = (
    "缺少 `rknn` 模块（rknn-toolkit2）。它不在 PyPI，请从瑞芯微官方发布页下载 "
    "wheel 后 pip install；或 uv pip install <下载的 .whl>。"
    "rknn-toolkit2 官方只提供 x86 Linux wheel，Windows/macOS 无法安装。"
)
INSTALL_HINT_ATC = (
    "未找到 atc（华为昇腾 CANN 离线模型编译器）。请安装 CANN Toolkit，"
    "并把 $ASCEND_HOME/bin 加入 PATH。"
)
INSTALL_HINT_CNMODEL = (
    "未找到 cnmodel（寒武纪 CNToolkit 离线模型编译器）。请安装 CNToolkit，"
    "并把其 bin 目录加入 PATH。"
)
INSTALL_HINT_HB_MAPPER = (
    "未找到 hb_mapper（地平线 OpenExplorer 工具链）。请安装 OpenExplorer（horizon_tc），"
    "并在容器/conda 环境内激活后使用。"
)


def _opts(kwargs: dict) -> dict[str, Any]:
    """取出 CLI `--opt k=v` 折叠成的字典（缺失时为空）。"""
    raw = kwargs.get("opts")
    return dict(raw) if isinstance(raw, dict) else {}


def _require_tool(name: str, hint: str) -> str:
    found = shutil.which(name) or shutil.which(name + ".exe") or shutil.which(name + ".cmd")
    if not found:
        raise MissingDependencyError(hint)
    return found


def _run_tool(tool: str, args: list[str], dst: Path, label: str) -> Path:
    """跑厂商 CLI，并校验产物真的落盘（防"退出码 0 但啥也没生成"的假成功）。"""
    result = subprocess.run([tool, *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-2000:]
        raise ConversionError(f"{label} 失败 (exit {result.returncode}): {tail}")
    if not dst.exists():
        raise ConversionError(
            f"{label} 退出码为 0，但未生成 {dst}。请核对工具链版本与参数"
            f"（日志尾部: {(result.stdout or '')[-500:]}）"
        )
    return dst


# ================= 华为昇腾 Ascend OM（atc） =================


def build_atc_args(src: Path, dst: Path, **kwargs) -> list[str]:
    """构造 `atc` 参数。

    atc 的 --output 是**不带扩展名的前缀**，成功后自动补 `.om`，所以这里先剥掉。
    --framework=5 表示 ONNX 输入（0=Caffe 1=MindSpore 3=TensorFlow 5=ONNX）。
    --soc_version 必须与目标芯片一致，无法给默认值，缺失即报错而不是猜。
    """
    opts = _opts(kwargs)
    soc = (opts.get("soc_version") or "").strip()
    if not soc:
        raise ConversionError(
            "ONNX -> OM 需要目标芯片型号：--opt soc_version=Ascend310P3 "
            "（atc 不会自动推断，写错会生成跑不起来的 .om）"
        )
    out_prefix = str(dst.with_suffix("")) if dst.suffix.lower() == ".om" else str(dst)
    args = [
        f"--model={src}",
        "--framework=5",
        f"--output={out_prefix}",
        f"--soc_version={soc}",
    ]
    if opts.get("input_shape"):
        args.append(f"--input_shape={opts['input_shape']}")
    if opts.get("input_format"):
        args.append(f"--input_format={opts['input_format']}")
    if opts.get("precision_mode"):
        args.append(f"--precision_mode={opts['precision_mode']}")
    if opts.get("batch"):
        args.append(f"--batch={opts['batch']}")
    return args


def convert_onnx_to_om(src: Path, dst: Path, **kwargs) -> Path:
    """ONNX -> 昇腾 OM，委托 CANN 的 atc。未实测（仓库环境无 CANN）。"""
    args = build_atc_args(src, dst, **kwargs)
    tool = _require_tool(TOOL_ATC, INSTALL_HINT_ATC)
    _ensure_parent(dst)
    return _run_tool(tool, args, dst, "atc")


# ================= 瑞芯微 RKNN（rknn-toolkit2 Python API） =================


def build_rknn_load_kwargs(**kwargs) -> dict[str, Any]:
    """构造 `RKNN.load_onnx()` 的关键字参数。

    input_size_list 只接受"固定 shape"的模型时使用；用户给了 --input-shape
    （如 1,3,224,224）就转成 [[1,3,224,224]]。
    """
    opts = _opts(kwargs)
    out: dict[str, Any] = {}
    shape = opts.get("input_shape")
    if shape:
        dims = [int(x) for x in str(shape).replace("x", ",").split(",") if str(x).strip()]
        out["input_size_list"] = [dims]
    if opts.get("input_names"):
        out["inputs"] = [n.strip() for n in str(opts["input_names"]).split(",") if n.strip()]
    return out


def build_rknn_build_kwargs(**kwargs) -> dict[str, Any]:
    """构造 `RKNN.build()` 的关键字参数。

    do_quantization 默认 True（NPU 上 INT8 是主路径）；给了 dataset（量化校准
    图片列表的 txt 路径）才允许量化，否则显式关掉，避免用具假数据集糊弄。
    """
    opts = _opts(kwargs)
    out: dict[str, Any] = {}
    dataset = opts.get("dataset")
    want_quant = str(opts.get("do_quantization", "true")).lower() not in ("0", "false", "no")
    if dataset:
        out["dataset"] = str(dataset)
        out["do_quantization"] = want_quant
    else:
        # 没给校准集时强制不量化：否则 rknn-toolkit2 会用随机数据校准，精度不可信
        out["do_quantization"] = False
    if opts.get("target_platform"):
        out["target_platform"] = str(opts["target_platform"])
    if opts.get("optimization_level") is not None:
        out["optimization_level"] = int(opts["optimization_level"])
    return out


def _import_rknn():
    """导入 rknn-toolkit2，缺库时给出"不在 PyPI"的具体获取方式。"""
    try:
        import rknn
    except ImportError as e:
        raise MissingDependencyError(INSTALL_HINT_RKNN) from e
    return rknn


def convert_onnx_to_rknn(src: Path, dst: Path, **kwargs) -> Path:
    """ONNX -> RKNN，委托 rknn-toolkit2 的 Python API（该工具链不提供 CLI）。"""
    load_kwargs = build_rknn_load_kwargs(**kwargs)
    build_kwargs = build_rknn_build_kwargs(**kwargs)
    _ensure_parent(dst)
    rknn = _import_rknn()
    # rknn-toolkit2 提供 rknn.api.RKNN，也允许直接 rknn.RKNN，两种都兼容
    api = rknn.api.RKNN if hasattr(rknn, "api") else rknn.RKNN
    model = api()
    ret = model.load_onnx(model=str(src), **load_kwargs)
    if ret != 0:
        raise ConversionError(f"rknn.load_onnx 返回 {ret}（非 0 表示失败）")
    ret = model.build(**build_kwargs)
    if ret != 0:
        raise ConversionError(f"rknn.build 返回 {ret}（非 0 表示失败）")
    ret = model.export_rknn(str(dst))
    if ret != 0:
        raise ConversionError(f"rknn.export_rknn 返回 {ret}（非 0 表示失败）")
    release = getattr(model, "release", None)
    if callable(release):
        release()
    if not dst.exists():
        raise ConversionError(f"rknn-toolkit2 未生成 {dst}")
    return dst


# ================= 寒武纪 MLU DLC（cnmodel） =================


def build_cnmodel_args(src: Path, dst: Path, **kwargs) -> list[str]:
    """构造寒武纪 CNToolkit 离线编译参数。

    不同大版本 CNToolkit 的子命令与参数名有差异，这里只拼最小必要集合
    （--model / --output），其余用 --opt 透传，避免伪造参数名。
    """
    opts = _opts(kwargs)
    args = ["--model", str(src), "--output", str(dst)]
    for key, flag in (
        ("precision", "--precision"),
        ("batch", "--batch"),
        ("core_number", "--core_number"),
        ("input_format", "--input_format"),
    ):
        if opts.get(key):
            args += [flag, str(opts[key])]
    return args


def convert_onnx_to_dlc(src: Path, dst: Path, **kwargs) -> Path:
    """ONNX -> 寒武纪 DLC，委托 CNToolkit 的 cnmodel。未实测（仓库环境无 CNToolkit）。"""
    args = build_cnmodel_args(src, dst, **kwargs)
    tool = _require_tool(TOOL_CNMODEL, INSTALL_HINT_CNMODEL)
    _ensure_parent(dst)
    return _run_tool(tool, args, dst, "cnmodel")


# ================= 地平线 BPU bin（hb_mapper makertbin） =================


def build_hb_mapper_args(config: Path, model_type: str = "onnx") -> list[str]:
    """构造 `hb_mapper makertbin` 参数。

    hb_mapper 只接受 YAML 配置（模型路径、输入 shape、输出目录都写在 yaml 里），
    不接受"src->dst"形式的直接参数，因此本函数只负责拼 yaml 与 model-type。
    """
    return ["makertbin", "--config", str(config), "--model-type", model_type]


def convert_onnx_to_bpu_bin(src: Path, dst: Path, **kwargs) -> Path:
    """ONNX -> 地平线 BPU .bin，委托 hb_mapper makertbin。

    需要用户提供 YAML 配置（`--opt hb_config=path/to/config.yaml`），因为产物名、
    输入 shape、校准数据都在 yaml 里声明。本仓库不生成该 yaml——模板随
    OpenExplorer 版本变化，凭猜写 yaml 等于伪造成功路径。
    未实测（仓库环境无 OpenExplorer）。
    """
    opts = _opts(kwargs)
    config = opts.get("hb_config")
    if not config:
        raise ConversionError(
            "ONNX -> 地平线 BPU bin 需要 hb_mapper 的 YAML 配置："
            "--opt hb_config=path/to/config.yaml（模型路径/输入 shape/校准集都在 yaml 里）"
        )
    args = build_hb_mapper_args(Path(config), str(opts.get("model_type") or "onnx"))
    tool = _require_tool(TOOL_HB_MAPPER, INSTALL_HINT_HB_MAPPER)
    _ensure_parent(dst)
    return _run_tool(tool, args, dst, "hb_mapper makertbin")


# ================= 注册"格式 -> 部署目标"路径 =================


def register_targets(reg) -> None:
    """把 NPU / 专用后端的路径声明登记到 registry。

    注意 onnx -> mlpackage 复用 converters.onnx_to_coreml 的实现（coremltools
    一条路径同时服务 CoreML 格式与 Apple ANE 目标），避免同一份逻辑写两遍。
    """
    from . import convert_onnx_to_coreml

    reg.register_target_path(
        "onnx", "rknn", convert_onnx_to_rknn, requires=(),
        description="ONNX -> Rockchip RKNN (rknn-toolkit2 Python API)",
    )
    reg.register_target_path(
        "onnx", "om", convert_onnx_to_om, requires=(),
        description="ONNX -> 昇腾 OM (atc --framework=5)",
    )
    reg.register_target_path(
        "onnx", "dlc", convert_onnx_to_dlc, requires=(),
        description="ONNX -> 寒武纪 DLC (cnmodel)",
    )
    reg.register_target_path(
        "onnx", "bin", convert_onnx_to_bpu_bin, requires=(),
        description="ONNX -> 地平线 BPU bin (hb_mapper makertbin)",
    )
    reg.register_target_path(
        "onnx", "mlpackage", convert_onnx_to_coreml, requires=("coreml",),
        description="ONNX -> CoreML mlpackage (coremltools, Apple ANE)",
    )
