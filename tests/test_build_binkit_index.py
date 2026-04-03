"""build_binkit_index 脚本单元测试。使用 mock 避免依赖 Ghidra。"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _make_lsir_raw(functions: list) -> dict:
    """构建 mock lsir_raw，与 extract_lsir_raw 输出格式一致。"""
    return {"functions": functions}


@patch("utils.ghidra_runner.require_ghidra_environment")
@patch("utils.ghidra_runner.run_ghidra_analysis")
def test_build_binkit_index_derives_from_lsir_raw(ghidra_mock, _require_ghidra_mock):
    """从 lsir_raw 推导 {name, entry}，输出格式与旧版 binkit_functions 一致。"""
    ghidra_mock.return_value = _make_lsir_raw([
        {"name": "main", "entry": "0x401000", "basic_blocks": []},
        {"name": "foo", "entry": "00402000", "basic_blocks": []},
    ])

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_binkit",
        PROJECT_ROOT / "scripts" / "sidechain" / "build_binkit_index.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with tempfile.TemporaryDirectory() as input_dir:
        (Path(input_dir) / "a.elf").write_bytes(b"\x7fELF")
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "out.json"
            old_argv = sys.argv
            try:
                sys.argv = [
                    "build_binkit_index",
                    "--input-dir", str(input_dir),
                    "--output", str(output_path),
                    "--temp-dir", str(Path(tmp) / "temp"),
                ]
                mod.main()
            finally:
                sys.argv = old_argv

            assert output_path.exists()
            with open(output_path, encoding="utf-8") as f:
                index = json.load(f)
    assert isinstance(index, list)
    assert len(index) == 1
    assert "binary" in index[0]
    assert "functions" in index[0]
    funcs = index[0]["functions"]
    assert len(funcs) == 2
    assert funcs[0]["name"] == "main" and funcs[0]["entry"] == "0x401000"
    assert funcs[1]["name"] == "foo" and funcs[1]["entry"] == "0x402000"
    ghidra_mock.assert_called_once()
    call_kw = ghidra_mock.call_args[1]
    assert call_kw["script_name"] == "extract_lsir_raw.java"
    assert call_kw["script_output_name"] == "lsir_raw.json"
    assert call_kw["return_dict"] is True
