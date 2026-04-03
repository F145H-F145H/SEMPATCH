"""不依赖 Ghidra 的 DAG 集成测试。使用预生成 fixture 替代 Ghidra 调用，验证完整流水线。"""
import json
import os

import pytest

from dag.builders import (
    build_diff_node,
    build_embed_node,
    build_feature_extract_node,
    build_load_db_node,
    build_lsir_build_node,
)
from dag.executor import run_dag
from dag.model import JobDAG

# Fixture 路径：相对于项目根
_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
_LSIR_RAW_MOCK_PATH = os.path.join(_FIXTURE_DIR, "lsir_raw_mock.json")
_DB_EMBEDDINGS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "vulnerability_db", "test_embeddings.json"
)


def _load_lsir_raw_mock() -> dict:
    """加载 lsir_raw mock fixture。"""
    with open(_LSIR_RAW_MOCK_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_dag_pipeline_without_ghidra():
    """使用 fixture 替代 Ghidra，运行完整 DAG：lsir_build → feature_extract → embed → load_db → diff。"""
    if not os.path.isfile(_LSIR_RAW_MOCK_PATH):
        pytest.skip(f"Fixture not found: {_LSIR_RAW_MOCK_PATH}")
    if not os.path.isfile(_DB_EMBEDDINGS_PATH):
        pytest.skip(f"DB embeddings not found: {_DB_EMBEDDINGS_PATH}")

    ctx = {"ghidra_output": _load_lsir_raw_mock()}

    dag = JobDAG()
    build_lsir_build_node(dag, "lsir_1", deps=[])
    build_feature_extract_node(dag, "feat_1", deps=["lsir_1"])
    build_embed_node(dag, "embed_1", deps=["feat_1"])
    build_load_db_node(dag, "load_1", db_path=_DB_EMBEDDINGS_PATH, deps=[])
    build_diff_node(dag, "diff_1", deps=["embed_1", "load_1"])

    run_dag(dag, ctx)

    assert "diff_result" in ctx
    assert "matches" in ctx["diff_result"]
    assert isinstance(ctx["diff_result"]["matches"], list)
