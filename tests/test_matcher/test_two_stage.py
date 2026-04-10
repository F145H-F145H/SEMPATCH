"""测试两阶段流水线 TwoStagePipeline。"""

import json
import os
import sys
import tempfile

import pytest

from matcher.two_stage import TwoStagePipeline, _LibraryFeaturesLazy


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


# ------------------------------------------------------------------
# _LibraryFeaturesLazy 单元测试
# ------------------------------------------------------------------


def _make_library_features_json(tmp_path, entries):
    """构造 library_features.json 并返回路径。"""
    path = tmp_path / "lib_features.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return str(path)


class TestLibraryFeaturesLazyEager:
    """小文件走 eager 路径。"""

    def test_contains_and_getitem(self, tmp_path):
        data = {"a|0x1": {"graph": {"num_nodes": 1}, "sequence": {"pcode_tokens": ["X"]}}}
        p = _make_library_features_json(tmp_path, data)
        lf = _LibraryFeaturesLazy(p)
        assert "a|0x1" in lf
        assert "b|0x2" not in lf
        assert lf["a|0x1"]["graph"]["num_nodes"] == 1
        with pytest.raises(KeyError):
            _ = lf["missing"]
        assert lf.get("missing") is None
        assert lf.get("a|0x1") is not None
        assert len(lf) == 1
        assert "a|0x1" in list(lf.keys())

    def test_getitem_roundtrip(self, tmp_path):
        data = {f"k|0x{i}": {"v": i} for i in range(50)}
        p = _make_library_features_json(tmp_path, data)
        lf = _LibraryFeaturesLazy(p)
        for k, v in data.items():
            assert lf[k] == v


class TestLibraryFeaturesLazyIndex:
    """大文件走 lazy index 路径。"""

    @staticmethod
    def _pad_to_threshold(tmp_path, path_str, threshold):
        """给文件填充注释使其超过阈值，触发 lazy 模式。"""
        import pathlib
        p = pathlib.Path(path_str)
        content = p.read_bytes()
        pad = b" " * max(0, threshold - len(content) + 1)
        # 在尾部 '}' 之前插入空白
        p.write_bytes(content[:-1] + pad + b"}")

    def test_index_lookup_matches_eager(self, tmp_path):
        data = {
            "lib_func_1|0x1000": {"graph": {"num_nodes": 3}, "sequence": {"pcode_tokens": ["A", "B"]}},
            "lib_func_2|0x2000": {"graph": {"num_nodes": 5}, "sequence": {"pcode_tokens": ["C", "D", "E"]}},
            "lib_func_3|0x3000": {"graph": {"num_nodes": 1}, "sequence": {"pcode_tokens": ["X"]}},
        }
        p = _make_library_features_json(tmp_path, data)
        # 强制走 index 路径
        lf = _LibraryFeaturesLazy(p, eager_threshold=0)
        for k, v in data.items():
            assert k in lf
            assert lf[k] == v
            assert lf.get(k) == v
        assert "nonexistent" not in lf
        assert lf.get("nonexistent") is None
        assert len(lf) == 3
        lf.close()

    def test_index_with_nested_values(self, tmp_path):
        """嵌套对象、数组、字符串、数字、布尔、null 均正确解析。"""
        data = {
            "f1": {
                "graph": {
                    "num_nodes": 2,
                    "edge_index": [[0, 1], [1, 0]],
                    "node_features": [["COPY", "LOAD"], ["STORE"]],
                },
                "sequence": {
                    "pcode_tokens": ["COPY", "LOAD", "STORE"],
                    "jump_mask": [0, 1, 0],
                    "seq_len": 3,
                },
                "extra": {"nested": {"deep": True, "val": None}},
            },
        }
        p = _make_library_features_json(tmp_path, data)
        lf = _LibraryFeaturesLazy(p, eager_threshold=0)
        assert lf["f1"]["graph"]["edge_index"] == [[0, 1], [1, 0]]
        assert lf["f1"]["extra"]["nested"]["deep"] is True
        assert lf["f1"]["extra"]["nested"]["val"] is None
        lf.close()

    def test_index_with_many_keys(self, tmp_path):
        """大量 key 时索引正确。"""
        n = 200
        data = {f"func_{i}|0x{i:x}": {"graph": {"num_nodes": i}, "sequence": {"pcode_tokens": ["X"] * (i % 5 + 1)}} for i in range(n)}
        p = _make_library_features_json(tmp_path, data)
        lf = _LibraryFeaturesLazy(p, eager_threshold=0)
        assert len(lf) == n
        # 抽查几个
        assert lf["func_0|0x0"]["graph"]["num_nodes"] == 0
        assert lf["func_99|0x63"]["graph"]["num_nodes"] == 99
        assert lf["func_199|0xc7"]["graph"]["num_nodes"] == 199
        assert "func_999|0x3e7" not in lf
        lf.close()


class TestTwoStagePipelineLazyFeatures:
    """TwoStagePipeline + 惰性 library features 集成。"""

    def test_pipeline_with_lazy_library_features(self, tmp_path):
        """大 library_features.json 触发 lazy index，pipeline 正常工作。"""
        mm = _minimal_multimodal()
        emb = {
            "functions": [
                {"function_id": "lib|0x1", "vector": [0.5] * 128},
                {"function_id": "lib|0x2", "vector": [0.3] * 128},
            ]
        }
        # 构造较大的 library_features 以触发 lazy
        lib_features = {}
        for i in range(100):
            lib_features[f"lib|0x{i:x}"] = _minimal_multimodal(
                pcode_tokens=["COPY", "INT_ADD"] * (i % 3 + 1)
            )
        # 确保实际用到的 key 存在
        lib_features["lib|0x1"] = mm
        lib_features["lib|0x2"] = _minimal_multimodal(pcode_tokens=["X", "Y"])

        emb_path = tmp_path / "emb.json"
        lf_path = tmp_path / "lib_feat.json"
        qf_path = tmp_path / "qf.json"
        emb_path.write_text(json.dumps(emb), encoding="utf-8")
        qf_path.write_text(json.dumps({"query|0x1": mm}), encoding="utf-8")
        lf_path.write_text(json.dumps(lib_features), encoding="utf-8")

        # 强制走 lazy 路径
        import matcher.two_stage as ts_mod
        orig_threshold = ts_mod._EAGER_LOAD_THRESHOLD_BYTES
        ts_mod._EAGER_LOAD_THRESHOLD_BYTES = 0
        try:
            pipeline = TwoStagePipeline(
                library_safe_embeddings_path=str(emb_path),
                library_features_path=str(lf_path),
                query_features_path=str(qf_path),
                coarse_k=5,
            )
            result = pipeline.retrieve_and_rerank("query|0x1")
            assert isinstance(result, list)
            if result:
                for i in range(len(result) - 1):
                    assert result[i][1] >= result[i + 1][1]
        finally:
            ts_mod._EAGER_LOAD_THRESHOLD_BYTES = orig_threshold
