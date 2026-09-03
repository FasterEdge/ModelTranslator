# ─────────────────────────────────────────────────────────────
# FasterEdge 开源项目
# Github: https://github.com/FasterEdge
# Gitee:  https://gitee.com/FasterEdge
# ─────────────────────────────────────────────────────────────
"""registry / cli 基础逻辑测试（不依赖重库）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_translator.registry import (
    FORMATS,
    detect_format,
    find_conversion,
    list_conversions,
)


def test_formats_registered():
    assert "pytorch" in FORMATS
    assert "onnx" in FORMATS
    assert "tflite" in FORMATS
    assert "openvino" in FORMATS
    assert "coreml" in FORMATS
    assert "tensorrt" in FORMATS
    assert "ggml" in FORMATS


def test_detect_format():
    import tempfile
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
    # 不支持的路径
    assert find_conversion("pytorch", "tflite") is None


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
    import tempfile

    from model_translator.cli import _find_project_dir

    old = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            root = _find_project_dir()
            assert root is not None
            assert (root / "pyproject.toml").is_file()
    finally:
        os.chdir(old)


if __name__ == "__main__":
    # 无 pytest 依赖的简单运行：逐个执行 test_* 函数
    failures = 0
    for name in sorted(dir()):
        if name.startswith("test_"):
            fn = globals()[name]
            try:
                fn()
                print(f"  ✓ {name}")
            except AssertionError as e:
                failures += 1
                print(f"  ✗ {name}: {e}")
    print(f"\n{failures} 个失败")
    raise SystemExit(1 if failures else 0)