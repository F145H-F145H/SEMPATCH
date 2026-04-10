"""测试：提交并执行 Ghidra headless 提取 P-code 的 DAG 任务。仅验证执行流程，不保留/断言输出。"""

import os
import tempfile

import pytest

# 确保 src 在 path 中（pyproject.toml pythonpath 已配置）
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dag.builders import build_ghidra_node
from dag.executor import run_dag
from dag.model import JobDAG


def _ghidra_available() -> bool:
    """检查 Ghidra 环境是否可用。"""
    try:
        from config import ANALYZE_HEADLESS, GHIDRA_HOME

        return (
            bool(GHIDRA_HOME)
            and os.path.isdir(GHIDRA_HOME)
            and os.path.isfile(ANALYZE_HEADLESS)
            and os.access(ANALYZE_HEADLESS, os.X_OK)
        )
    except Exception:
        return False


def _get_test_binary() -> str:
    """返回用于测试的小二进制路径。"""
    for p in ("/bin/true", "/usr/bin/true"):
        if os.path.isfile(p):
            return p
    return ""


@pytest.mark.integration
@pytest.mark.ghidra
class TestGhidraExtractDAG:
    """提交 Ghidra headless 提取 P-code 任务并执行。"""

    def test_run_ghidra_extract_dag(self):
        """构建单节点 DAG，执行 Ghidra 提取，仅验证执行完成。"""
        if not _ghidra_available():
            pytest.skip("Ghidra 未安装或不可用")
        binary = _get_test_binary()
        if not binary:
            pytest.skip("No /bin/true or /usr/bin/true found")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = os.path.join(tmp, "ghidra_out")
            os.makedirs(output_dir, exist_ok=True)

            dag = JobDAG()
            build_ghidra_node(
                dag,
                node_id="ghidra_1",
                binary_path=binary,
                output_dir=output_dir,
                deps=[],
                force=True,
            )

            run_dag(dag)

            node = dag.nodes["ghidra_1"]
            assert node.done, "Ghidra 节点应完成执行"
            assert not node.failed, "Ghidra 节点不应失败"
