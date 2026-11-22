# ─────────────────────────────────────────────────────────────
# FasterEdge 开源项目
# Github: https://github.com/FasterEdge
# Gitee:  https://gitee.com/FasterEdge
# ─────────────────────────────────────────────────────────────
"""registry / cli 基础逻辑测试（不依赖重库）。

测试分两类：
1. 注册表/探测逻辑：纯函数，不导入任何厂商 SDK。
2. 厂商工具链的参数构造：只测 `build_*` 纯函数与缺依赖报错，
   **不**真跑 NPU 转换（仓库环境没有 rknn/CANN/CNToolkit/OpenExplorer）。
"""

import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_translator.registry import (
    FORMATS,
    TARGETS,
    detect_format,
    detect_format_with_reason,
    find_conversion,
    find_target_path,
    list_conversions,
    list_target_paths,
    plan_route,
    resolve_target,
)

# ================= 原有断言（保持意图不变） =================


def test_formats_registered():
    assert "pytorch" in FORMATS
    assert "onnx" in FORMATS
    assert "tflite" in FORMATS
    assert "openvino" in FORMATS
    assert "coreml" in FORMATS
    assert "tensorrt" in FORMATS
    assert "ggml" in FORMATS


def test_detect_format():
    assert detect_format(Path("model.pt")) == "pytorch"
    assert detect_format(Path("model.pth")) == "pytorch"
    assert detect_format(Path("model.onnx")) == "onnx"
    assert detect_format(Path("model.h5")) == "keras"
    assert detect_format(Path("model.tflite")) == "tflite"
    assert detect_format(Path("model.xml")) == "openvino"
    assert detect_format(Path("model.engine")) == "tensorrt"
    assert detect_format(Path("model.mlpackage")) == "coreml"
    assert detect_format(Path("model.gguf")) == "ggml"
    # SavedModel 目录
    with tempfile.TemporaryDirectory() as d:
        assert detect_format(Path(d)) == "tensorflow"
    # 未知
    assert detect_format(Path("model.txt")) is None


def test_find_conversion():
    assert find_conversion("pytorch", "onnx") is not None
    assert find_conversion("pytorch", "torchscript") is not None
    assert find_conversion("onnx", "openvino") is not None
    assert find_conversion("onnx", "coreml") is not None
    assert find_conversion("onnx", "tensorrt") is not None
    assert find_conversion("tensorflow", "tflite") is not None
    assert find_conversion("keras", "tflite") is not None
    assert find_conversion("tensorflow", "onnx") is not None
    assert find_conversion("keras", "onnx") is not None
    assert find_conversion("coreml", "onnx") is not None
    assert find_conversion("safetensors", "ggml") is not None
    # 不支持的路径
    assert find_conversion("pytorch", "tflite") is None
    assert find_conversion("onnx", "tflite") is None
    assert find_conversion("pytorch", "openvino") is None


def test_list_conversions_nonempty():
    assert len(list_conversions()) >= 10


def test_find_project_dir_finds_pyproject():
    from model_translator.cli import _find_project_dir

    root = _find_project_dir()
    assert root is not None, "应在当前目录或模块位置的祖先中找到项目根"
    assert (root / "pyproject.toml").is_file()


def test_find_project_dir_ignores_foreign_cwd():
    """在无关目录(如容器挂载的 /workspace)执行时, 仍能定位到项目根。"""
    import os

    from model_translator.cli import _find_project_dir

    with tempfile.TemporaryDirectory() as d:
        old = os.getcwd()
        try:
            os.chdir(d)
            root = _find_project_dir()
            assert root is not None
            assert (root / "pyproject.toml").is_file()
        finally:
            os.chdir(old)


# ================= 新增：部署目标（格式 x 目标）注册表 =================


def test_npu_targets_registered():
    """用户关心的 NPU 目标必须全部在注册表里（H2 的核心诉求）。"""
    for key in ("rknn", "om", "dlc", "bin", "mlpackage"):
        assert key in TARGETS, f"缺少部署目标 {key}"


def test_npu_targets_consume_onnx_via():
    """NPU 工具链普遍吃 ONNX，目标必须声明 via=onnx，路由才成立。"""
    for key in ("rknn", "om", "dlc", "bin"):
        assert TARGETS[key].via == "onnx"


def test_no_target_claims_verified():
    """仓库环境没有厂商 SDK，任何目标都不得声称已实测（防伪造成功）。"""
    assert all(not t.verified for t in TARGETS.values())


def test_each_target_has_install_hint():
    """每个目标都必须给出"怎么拿到工具链"，否则用户无从下手。"""
    for t in TARGETS.values():
        if t.requires:
            continue
        assert t.install_hint.strip(), f"{t.key} 缺少 install_hint"


def test_target_paths_registered_for_every_target():
    registered = {p.target for p in list_target_paths()}
    assert {"rknn", "om", "dlc", "bin", "mlpackage"} <= registered


def test_find_target_path_onnx_to_npu():
    assert find_target_path("onnx", "rknn") is not None
    assert find_target_path("onnx", "om") is not None
    assert find_target_path("onnx", "dlc") is not None
    assert find_target_path("onnx", "bin") is not None


def test_find_target_path_unsupported_pair():
    assert find_target_path("tflite", "rknn") is None


def test_plan_route_direct():
    route = plan_route("onnx", "rknn")
    assert route is not None and len(route) == 1
    assert route[0].src == "onnx" and route[0].dst == "rknn"


def test_plan_route_two_step_via_onnx():
    """pytorch -> rknn 必须先 pytorch->onnx 再 onnx->rknn，共两步。"""
    route = plan_route("pytorch", "rknn")
    assert route is not None and len(route) == 2
    assert (route[0].src, route[0].dst) == ("pytorch", "onnx")
    assert (route[1].src, route[1].dst) == ("onnx", "rknn")


def test_plan_route_unsupported_returns_none():
    assert plan_route("ggml", "rknn") is None
    assert plan_route("onnx", "nonexistent") is None


def test_resolve_target_alias_and_extension():
    assert resolve_target("rknn") == "rknn"
    assert resolve_target(".rknn") == "rknn"
    assert resolve_target("Ascend".lower()) is None  # 别名在 cli 层，registry 只认 key/扩展名


# ================= 新增：detect_format 内容探测 =================


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_detect_gguf_by_magic_not_extension(tmp_path):
    """扩展名撒谎也要按内容判：GGUF 魔数 'GGUF' 在文件头。"""
    p = _write(tmp_path / "fake.bin", b"GGUF" + b"\x00" * 32)
    key, reason = detect_format_with_reason(p)
    assert key == "ggml"
    assert "GGUF" in reason


def test_detect_tflite_by_magic_not_extension(tmp_path):
    """TFLite FlatBuffers 标识 'TFL3' 在偏移 4。"""
    p = _write(tmp_path / "fake.onnx", b"\x00" * 4 + b"TFL3" + b"\x00" * 16)
    assert detect_format(p) == "tflite"


def test_detect_extension_used_when_no_magic(tmp_path):
    """没有可识别内容时退回扩展名，原有行为不变。"""
    p = _write(tmp_path / "model.onnx", b"\x08\x1a\x00" + b"\x00" * 32)
    key, reason = detect_format_with_reason(p)
    assert key == "onnx"
    assert "扩展名" in reason


def test_detect_torchscript_zip_contents(tmp_path):
    """TorchScript 是 zip，含 <archive>/code/xxxx.py，扩展名是 .pt。"""
    arc = tmp_path / "scripted.pt"
    with zipfile.ZipFile(arc, "w") as zf:
        zf.writestr("archive/code/version", "1")
        zf.writestr("archive/code/__model__.py", "class M: pass\n")
    assert detect_format(arc) == "torchscript"


def test_detect_pytorch_zip_contents(tmp_path):
    """torch.save 的 zip 布局含 data.pkl + version。"""
    p = tmp_path / "full_model.pt"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("data.pkl", b"\x80\x02")
        zf.writestr("version", "3")
    assert detect_format(p) == "pytorch"


def test_detect_keras_zip_contents(tmp_path):
    """Keras 3 的 .keras 是 zip，根含 metadata.json + config.json。"""
    p = tmp_path / "m.keras"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("metadata.json", "{}")
        zf.writestr("config.json", "{}")
    assert detect_format(p) == "keras"


def test_detect_zip_without_markers_falls_back_to_extension(tmp_path):
    """未知 zip 布局不瞎猜，退回扩展名（这里按 .pt -> pytorch）。"""
    p = tmp_path / "unknown.pt"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("whatever.txt", "x")
    assert detect_format(p) == "pytorch"


def test_detect_safetensors_by_header(tmp_path):
    """safetensors 没有魔数：8 字节小端元数据长度 + 紧随的 JSON 对象。"""
    header = b'{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    p = _write(tmp_path / "weights.safetensors",
               len(header).to_bytes(8, "little") + header + b"\x00" * 8)
    key, reason = detect_format_with_reason(p)
    assert key == "safetensors"
    assert "JSON" in reason


def test_detect_safetensors_shape_but_wrong_extension(tmp_path):
    """.bin 扩展名会落到 ggml，内容像 safetensors 时必须按内容优先判为 safetensors。"""
    header = b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    p = _write(tmp_path / "weird.bin",
               len(header).to_bytes(8, "little") + header + b"\x00" * 4)
    assert detect_format(p) == "safetensors"


def test_detect_plain_bin_still_ggml(tmp_path):
    """普通 .bin（非 safetensors 头部）仍按扩展名回退，不误判。"""
    p = _write(tmp_path / "plain.bin", b"\x00" * 32)
    assert detect_format(p) == "ggml"


def test_detect_saved_model_dir(tmp_path):
    """目录型 SavedModel：靠 saved_model.pb 判定。"""
    (tmp_path / "saved_model.pb").write_bytes(b"x")
    assert detect_format(tmp_path) == "tensorflow"


def test_detect_mlpackage_dir(tmp_path):
    """目录型 mlpackage：靠 Manifest.json 判定。"""
    (tmp_path / "Manifest.json").write_text("{}")
    assert detect_format(tmp_path) == "coreml"


def test_detect_empty_dir_keeps_legacy_saved_model_assumption(tmp_path):
    """空目录仍按 SavedModel 处理（保持既有行为），但判定依据要写明是"假定"。"""
    key, reason = detect_format_with_reason(tmp_path)
    assert key == "tensorflow"
    assert "假定" in reason


def test_detect_unknown_file_returns_none(tmp_path):
    p = _write(tmp_path / "x.dat", b"\x00" * 32)
    assert detect_format(p) is None


# ================= 新增：厂商工具链参数构造（纯函数单测） =================


def test_build_atc_args_requires_soc_version(tmp_path):
    from model_translator.converters.targets import build_atc_args

    with pytest.raises(Exception) as ei:
        build_atc_args(tmp_path / "m.onnx", tmp_path / "m.om")
    assert "soc_version" in str(ei.value)


def test_build_atc_args_strips_om_suffix_and_sets_framework(tmp_path):
    """atc 的 --output 是不带扩展名的前缀，.om 必须剥掉；framework=5 表示 ONNX。"""
    from model_translator.converters.targets import build_atc_args

    args = build_atc_args(tmp_path / "m.onnx", tmp_path / "out.om",
                          opts={"soc_version": "Ascend310P3"})
    joined = " ".join(args)
    assert f"--model={tmp_path / 'm.onnx'}" in joined
    assert "--framework=5" in joined
    assert f"--output={tmp_path / 'out'}" in joined
    assert "--soc_version=Ascend310P3" in joined
    assert not any(a.endswith(".om") for a in args if a.startswith("--output"))


def test_build_atc_args_extra_opt(tmp_path):
    from model_translator.converters.targets import build_atc_args

    args = build_atc_args(tmp_path / "m.onnx", tmp_path / "o.om",
                          opts={"soc_version": "Ascend310P3", "input_shape": "1,3,224,224"})
    assert "--input_shape=1,3,224,224" in args


def test_build_rknn_load_kwargs_converts_input_shape():
    from model_translator.converters.targets import build_rknn_load_kwargs

    out = build_rknn_load_kwargs(opts={"input_shape": "1,3,224,224"})
    assert out["input_size_list"] == [[1, 3, 224, 224]]


def test_build_rknn_build_kwargs_defaults_to_no_quantization_without_dataset():
    """没给校准集时必须关掉量化：否则会用随机数据校准，精度不可信。"""
    from model_translator.converters.targets import build_rknn_build_kwargs

    out = build_rknn_build_kwargs()
    assert out["do_quantization"] is False
    assert "dataset" not in out


def test_build_rknn_build_kwargs_with_dataset():
    from model_translator.converters.targets import build_rknn_build_kwargs

    out = build_rknn_build_kwargs(opts={"dataset": "calib.txt"})
    assert out["dataset"] == "calib.txt"
    assert out["do_quantization"] is True


def test_build_cnmodel_args_passthrough(tmp_path):
    from model_translator.converters.targets import build_cnmodel_args

    args = build_cnmodel_args(tmp_path / "m.onnx", tmp_path / "m.dlc",
                              opts={"precision": "int8"})
    assert "--precision" in args and "int8" in args


def test_build_hb_mapper_args_uses_yaml(tmp_path):
    from model_translator.converters.targets import build_hb_mapper_args

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model: {}")
    args = build_hb_mapper_args(cfg)
    assert args[0] == "makertbin"
    assert "--config" in args and str(cfg) in args


def test_onnx_to_rknn_raises_without_rknn_module(tmp_path):
    """本机没有 rknn-toolkit2：必须报 MissingDependencyError，且提示不在 PyPI。"""
    from model_translator.converters import MissingDependencyError
    from model_translator.converters.targets import convert_onnx_to_rknn

    with pytest.raises(MissingDependencyError) as ei:
        convert_onnx_to_rknn(tmp_path / "m.onnx", tmp_path / "m.rknn")
    assert "rknn-toolkit2" in str(ei.value)


def test_onnx_to_om_raises_without_atc(tmp_path):
    """本机没有 CANN：缺工具时报 MissingDependencyError，并说明装什么。"""
    from model_translator.converters import MissingDependencyError
    from model_translator.converters.targets import convert_onnx_to_om

    with pytest.raises(MissingDependencyError) as ei:
        convert_onnx_to_om(tmp_path / "m.onnx", tmp_path / "m.om",
                           opts={"soc_version": "Ascend310P3"})
    assert "atc" in str(ei.value)


def test_onnx_to_dlc_raises_without_cnmodel(tmp_path):
    from model_translator.converters import MissingDependencyError
    from model_translator.converters.targets import convert_onnx_to_dlc

    with pytest.raises(MissingDependencyError) as ei:
        convert_onnx_to_dlc(tmp_path / "m.onnx", tmp_path / "m.dlc")
    assert "cnmodel" in str(ei.value)


def test_onnx_to_bin_requires_yaml_config(tmp_path):
    """地平线必须先给 yaml，仓库不猜 yaml 内容（否则等于伪造成功路径）。"""
    from model_translator.converters import ConversionError
    from model_translator.converters.targets import convert_onnx_to_bpu_bin

    with pytest.raises(ConversionError) as ei:
        convert_onnx_to_bpu_bin(tmp_path / "m.onnx", tmp_path / "m.bin")
    assert "hb_config" in str(ei.value)


def test_build_trtexec_args_includes_shapes_when_given(tmp_path):
    """trtexec 参数构造抽成纯函数后可单测；给了 shape 就带三档 shapes。"""
    from model_translator.converters import build_trtexec_args

    args = build_trtexec_args(tmp_path / "m.onnx", tmp_path / "m.engine",
                              input_shape="1,3,224,224")
    assert any("minShapes=input:1x3x224x224" in a for a in args)
    assert any("maxShapes=input:1x3x224x224" in a for a in args)


# ================= 新增：CLI 层 =================


def _run_cli(*argv):
    from click.testing import CliRunner

    from model_translator.cli import main

    return CliRunner().invoke(main, list(argv))


def test_cli_list_shows_target_matrix():
    result = _run_cli("list")
    assert result.exit_code == 0, result.output
    for key in ("rknn", "om", "dlc", "bin", "mlpackage"):
        assert key in result.output


def test_cli_list_target_rknn():
    result = _run_cli("list", "--target", "rknn")
    assert result.exit_code == 0, result.output
    assert "rknn-toolkit2" in result.output
    assert "未实测" in result.output


def test_cli_convert_dry_run_to_rknn(tmp_path):
    """dry-run 不执行真转换，但要把路由和未实测标注打出来。"""
    src = _write(tmp_path / "m.onnx", b"\x08\x1a\x00" + b"\x00" * 16)
    dst = tmp_path / "out.rknn"
    result = _run_cli("convert", str(src), str(dst), "--target", "rknn", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "ONNX" in result.output
    assert "RKNN" in result.output
    assert "dry-run" in result.output
    assert not dst.exists()


def test_cli_convert_to_rknn_missing_tool_reports_hint(tmp_path):
    """真跑但工具缺失：必须失败并给出获取方式，且退出码为 3。"""
    src = _write(tmp_path / "m.onnx", b"\x08\x1a\x00" + b"\x00" * 16)
    result = _run_cli("convert", str(src), str(tmp_path / "out.rknn"), "--target", "rknn")
    assert result.exit_code == 3, result.output
    assert "rknn-toolkit2" in result.output


def test_cli_convert_unknown_target_fails(tmp_path):
    src = _write(tmp_path / "m.onnx", b"\x08\x1a\x00" + b"\x00" * 16)
    result = _run_cli("convert", str(src), str(tmp_path / "o.zzz"), "--target", "zzz")
    assert result.exit_code == 2, result.output


def test_cli_convert_no_route_fails(tmp_path):
    """ggml 到 rknn 没有路由，必须明确报错而不是瞎跑。"""
    src = _write(tmp_path / "m.gguf", b"GGUF" + b"\x00" * 16)
    result = _run_cli("convert", str(src), str(tmp_path / "o.rknn"), "--target", "rknn")
    assert result.exit_code == 2, result.output
    assert "路由" in result.output


def test_cli_parse_opts_rejects_bad_pair():
    from model_translator.cli import _parse_opts

    try:
        _parse_opts(("soc_version=Ascend310P3", "no_equals_sign"))
    except Exception as e:  # noqa: BLE001 - click.UsageError 即可
        assert "--opt" in str(e)
    else:  # pragma: no cover - 上面必须抛异常
        raise AssertionError("_parse_opts 应拒绝缺少 '=' 的项")


def test_cli_parse_opts_ok():
    from model_translator.cli import _parse_opts

    assert _parse_opts(("a=1", "b=x=y")) == {"a": "1", "b": "x=y"}


def test_cli_info_shows_detection_reason(tmp_path):
    p = _write(tmp_path / "m.gguf", b"GGUF" + b"\x00" * 16)
    result = _run_cli("info", str(p))
    assert result.exit_code == 0, result.output
    assert "GGUF" in result.output


if __name__ == "__main__":
    # 无 pytest 依赖的兜底运行：逐个执行不需要 tmp_path fixture 的 test_* 函数。
    # 带 fixture 的用例会跳过（提示用 uv run pytest 跑全套）。
    failures = 0
    skipped = 0
    for name in sorted(dir()):
        if not name.startswith("test_"):
            continue
        fn = globals()[name]
        try:
            fn()
            print(f"  PASS {name}")
        except TypeError as e:
            if "positional argument" in str(e):
                skipped += 1
                continue
            failures += 1
            print(f"  FAIL {name}: {e}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL {name}: {e}")
    print(f"\n{failures} 个失败, {skipped} 个跳过（需要 pytest fixture，用 uv run pytest 跑）")
    raise SystemExit(1 if failures else 0)
