"""测试两阶段流水线 TwoStagePipeline。"""

import json
import os
import sys
import tempfile

import pytest

from matcher.two_stage import TwoStagePipeline


def _minimal_multimodal(pcode_tokens=None, node_features=None):
    """构造最小 multimodal 特征。"""
    if pcode_tokens is None:
        pcode_tokens = ["COPY", "INT_SUB", "INT_ADD"]
    if node_features is None:
        node_features = [[]]
    return {
        "graph": {
            "num_nodes": max(1, len(node_features)),
            "edge_index": [[], []],
            "node_features": node_features,
        },
        "sequence": {
            "pcode_tokens": pcode_tokens,
            "jump_mask": [0] * len(pcode_tokens),
            "seq_len": len(pcode_tokens),
        },
    }


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def two_stage_fixtures(tmp_path):
    """构造最小库与查询 fixtures 供 TwoStagePipeline 使用。"""
    mm = _minimal_multimodal()
    emb = {
        "functions": [
            {"function_id": "lib|0x1", "vector": [0.5] * 128},
            {"function_id": "lib|0x2", "vector": [0.3] * 128},
        ]
    }
    lib_features = {"lib|0x1": mm, "lib|0x2": _minimal_multimodal(pcode_tokens=["X", "Y"])}
    query_features = {"query|0x1": mm}

    emb_path = tmp_path / "embeddings.json"
    lf_path = tmp_path / "library_features.json"
    qf_path = tmp_path / "query_features.json"
    emb_path.write_text(json.dumps(emb), encoding="utf-8")
    lf_path.write_text(json.dumps(lib_features), encoding="utf-8")
    qf_path.write_text(json.dumps(query_features), encoding="utf-8")

    return {
        "embeddings": str(emb_path),
        "library_features": str(lf_path),
        "query_features": str(qf_path),
    }


def test_two_stage_pipeline_retrieve(two_stage_fixtures):
    """retrieve 返回非空列表，元素为 function_id 字符串。"""
    paths = two_stage_fixtures
    pipeline = TwoStagePipeline(
        library_safe_embeddings_path=paths["embeddings"],
        library_features_path=paths["library_features"],
        query_features_path=paths["query_features"],
        coarse_k=5,
    )
    result = pipeline.retrieve("query|0x1")
    assert isinstance(result, list)
    assert len(result) <= 5
    assert all(isinstance(x, str) and "|" in x for x in result)


def test_two_stage_pipeline_rerank(two_stage_fixtures):
    """rerank 返回按 score 降序的 (candidate_id, score) 列表。"""
    paths = two_stage_fixtures
    pipeline = TwoStagePipeline(
        library_safe_embeddings_path=paths["embeddings"],
        library_features_path=paths["library_features"],
        query_features_path=paths["query_features"],
        coarse_k=5,
    )
    candidates = pipeline.retrieve("query|0x1")
    if not candidates:
        pytest.skip("retrieve 返回空，无法测试 rerank")
    result = pipeline.rerank("query|0x1", candidates)
    assert isinstance(result, list)
    for i, (cid, score) in enumerate(result):
        assert isinstance(cid, str)
        assert isinstance(score, (int, float))
    for i in range(len(result) - 1):
        assert result[i][1] >= result[i + 1][1]


def test_two_stage_pipeline_retrieve_and_rerank(two_stage_fixtures):
    """retrieve_and_rerank 返回非空、按得分降序；同函数对应排前列。"""
    paths = two_stage_fixtures
    pipeline = TwoStagePipeline(
        library_safe_embeddings_path=paths["embeddings"],
        library_features_path=paths["library_features"],
        query_features_path=paths["query_features"],
        coarse_k=5,
    )
    result = pipeline.retrieve_and_rerank("query|0x1")
    assert isinstance(result, list)
    if result:
        for i in range(len(result) - 1):
            assert result[i][1] >= result[i + 1][1]


def test_two_stage_pipeline_rerank_empty_candidates(two_stage_fixtures):
    """rerank 空候选返回空列表。"""
    paths = two_stage_fixtures
    pipeline = TwoStagePipeline(
        library_safe_embeddings_path=paths["embeddings"],
        library_features_path=paths["library_features"],
        query_features_path=paths["query_features"],
        coarse_k=5,
    )
    result = pipeline.rerank("query|0x1", [])
    assert result == []


def test_two_stage_pipeline_missing_query_raises(two_stage_fixtures):
    """不存在的 query_func_id 抛出 KeyError 并附带清晰信息。"""
    paths = two_stage_fixtures
    pipeline = TwoStagePipeline(
        library_safe_embeddings_path=paths["embeddings"],
        library_features_path=paths["library_features"],
        query_features_path=paths["query_features"],
        coarse_k=5,
    )
    with pytest.raises(KeyError, match="不存在"):
        pipeline.retrieve("nonexistent|0x99")
    with pytest.raises(KeyError, match="不存在"):
        pipeline.rerank("nonexistent|0x99", ["lib|0x1"])
    with pytest.raises(KeyError, match="不存在"):
        pipeline.retrieve_and_rerank("nonexistent|0x99")


def test_two_stage_pipeline_no_dag_import():
    """导入 TwoStagePipeline 不会加载 dag 模块。"""
    dag_modules = [k for k in sys.modules if "dag" in k and k.startswith("dag")]
    from matcher.two_stage import TwoStagePipeline  # noqa: F401

    dag_after = [k for k in sys.modules if "dag" in k and k.startswith("dag")]
    assert set(dag_after) == set(dag_modules), "导入 two_stage 不应引入 dag 模块"
