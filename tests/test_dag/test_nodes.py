"""测试 DAG 节点（不含 Ghidra 集成）。"""
import os

import pytest

from dag.model import JobDAG
from dag.builders import build_lsir_build_node, build_feature_extract_node, build_embed_node
from dag.executor import run_dag
from dag.nodes import NODE_TYPE_REGISTRY


def test_run_dag_populates_passed_empty_ctx():
    """传入空 dict 时，节点写入必须反映在同一对象上（regression：勿使用 ctx or {}）。"""
    from dag.builders.fusion import build_load_db_node

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    db = os.path.join(root, "data", "vulnerability_db", "test_vuln_lsir.json")
    if not os.path.isfile(db):
        pytest.skip(f"Fixture 缺失: {db}")
    dag = JobDAG()
    build_load_db_node(dag, "load_1", db_path=db, deps=[], db_format="lsir")
    ctx: dict = {}
    run_dag(dag, ctx)
    assert "db_lsir" in ctx
    assert isinstance(ctx["db_lsir"].get("functions"), list)


def test_lsir_build_node():
    """LSIRBuildNode 从 ghidra_output 构建 lsir。"""
    dag = JobDAG()
    # 需要先有 ghidra_output，用 mock
    build_lsir_build_node(dag, "lsir_1", deps=[])
    # 无 deps 时 lsir_build 会直接就绪，但执行时需要 ctx[ghidra_output]
    ctx = {"ghidra_output": {"functions": []}}
    run_dag(dag, ctx)
    assert "lsir" in ctx
    assert ctx["lsir"]["functions"] == []


def test_unpack_node_binwalk_not_found(tmp_path):
    """UnpackNode 在 binwalk 未安装时抛出明确错误提示。"""
    firmware = tmp_path / "firmware.bin"
    firmware.write_bytes(b"\x00" * 64)
    UnpackNodeCls = NODE_TYPE_REGISTRY["unpack"]
    node = UnpackNodeCls(
        node_id="unpack_1",
        node_type="unpack",
        params={
            "firmware_path": str(firmware),
            "output_dir": str(tmp_path / "out"),
            "binwalk_cmd": "nonexistent_binwalk_xyz_12345",
        },
        deps=[],
    )
    with pytest.raises(RuntimeError) as exc_info:
        node.execute({})
    assert "binwalk 未安装" in str(exc_info.value) or "未在 PATH 中" in str(exc_info.value)
