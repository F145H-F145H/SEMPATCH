"""测试 Ghidra exit 0 但输出为空时的异常行为。"""

import os
import tempfile
from unittest.mock import patch


def test_empty_output_raises_runtime_error():
    """Ghidra exit 0 但输出文件为空时，run_ghidra_analysis 抛出 RuntimeError。"""
    from utils.ghidra_runner import run_ghidra_analysis

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = os.path.join(tmp, "output")
        os.makedirs(output_dir)
        # 创建空的输出文件（模拟 Ghidra exit 0 但没写内容）
        empty_output = os.path.join(output_dir, "lsir_raw.json")
        with open(empty_output, "w") as f:
            pass  # 空文件

        with patch("utils.ghidra_runner.validate_ghidra_environment"), \
             patch("utils.ghidra_runner.can_skip_ghidra", return_value=False), \
             patch("utils.ghidra_runner.read_from_binary_cache", return_value=None), \
             patch("utils.ghidra_runner.build_ghidra_command", return_value=["echo"]), \
             patch("utils.ghidra_runner.execute_ghidra_process", return_value=(0, "")), \
             patch("utils.ghidra_runner.write_to_binary_cache"):
            try:
                run_ghidra_analysis(
                    binary_path="/fake/bin",
                    output_dir=output_dir,
                    force=True,
                )
                raise AssertionError("Expected RuntimeError")
            except RuntimeError as e:
                assert "empty" in str(e).lower()


def test_nonempty_output_succeedes():
    """输出文件非空时正常返回路径。"""
    from utils.ghidra_runner import run_ghidra_analysis

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = os.path.join(tmp, "output")
        os.makedirs(output_dir)
        output_file = os.path.join(output_dir, "lsir_raw.json")
        with open(output_file, "w") as f:
            f.write('{"functions": []}')

        with patch("utils.ghidra_runner.validate_ghidra_environment"), \
             patch("utils.ghidra_runner.can_skip_ghidra", return_value=False), \
             patch("utils.ghidra_runner.read_from_binary_cache", return_value=None), \
             patch("utils.ghidra_runner.build_ghidra_command", return_value=["echo"]), \
             patch("utils.ghidra_runner.execute_ghidra_process", return_value=(0, "ok")), \
             patch("utils.ghidra_runner.write_to_binary_cache"):
            result = run_ghidra_analysis(
                binary_path="/fake/bin",
                output_dir=output_dir,
                force=True,
            )
            assert result == output_file
