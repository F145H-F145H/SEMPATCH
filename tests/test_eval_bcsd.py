"""测试 eval_bcsd 评估脚本。"""
import importlib.util
import json
import os
import sys

import numpy as np

# 加载 scripts.eval_bcsd 模块
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "eval_bcsd",
    os.path.join(_project_root, "scripts", "eval_bcsd.py"),
)
assert _spec and _spec.loader
eval_bcsd = importlib.util.module_from_spec(_spec)
sys.modules["eval_bcsd"] = eval_bcsd
_spec.loader.exec_module(eval_bcsd)

load_embeddings = eval_bcsd.load_embeddings
compute_top_k = eval_bcsd.compute_top_k
build_relevant_pairs = eval_bcsd.build_relevant_pairs
build_relevant_pairs_by_cve = eval_bcsd.build_relevant_pairs_by_cve
compute_metrics = eval_bcsd.compute_metrics


def test_load_embeddings():
    """小型 JSON 验证 names 与 vectors 形状。"""
    # 使用项目内已有的 test_embeddings
    path = os.path.join(_project_root, "data", "vulnerability_db", "test_embeddings.json")
    if not os.path.isfile(path):
        # 若无真实文件，构造临时 JSON
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "functions": [
                    {"name": "foo", "vector": [0.1, 0.2, 0.3]},
                    {"name": "bar", "vector": [0.4, 0.5, 0.6]},
                ]
            }, f)
        path = f.name
    try:
        names, vectors, cves = load_embeddings(path)
        assert len(names) >= 2
        assert vectors.ndim == 2
        assert vectors.shape[0] == len(names)
        assert vectors.shape[1] >= 1
        assert len(cves) == len(names)
    finally:
        if path.startswith("/tmp") or "tempfile" in str(path):
            try:
                os.unlink(path)
            except OSError:
                pass


def test_load_embeddings_small_fixture(tmp_path):
    """用固定小数据验证 load_embeddings。"""
    data = {
        "functions": [
            {"name": "func_a", "vector": [1.0, 0.0, 0.0]},
            {"name": "func_b", "vector": [0.0, 1.0, 0.0]},
            {"name": "func_c", "vector": [0.0, 0.0, 1.0]},
        ]
    }
    p = tmp_path / "tiny.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    names, vectors, cves = load_embeddings(str(p))
    assert names == ["func_a", "func_b", "func_c"]
    assert vectors.shape == (3, 3)
    assert cves == [[], [], []]
    np.testing.assert_array_almost_equal(vectors[0], [1, 0, 0])
    np.testing.assert_array_almost_equal(vectors[1], [0, 1, 0])
    np.testing.assert_array_almost_equal(vectors[2], [0, 0, 1])


def test_load_embeddings_with_cve(tmp_path):
    """验证 load_embeddings 正确提取 cve 字段（规范为列表）。"""
    data = {
        "functions": [
            {"name": "func_a", "vector": [1.0, 0.0], "cve": "CVE-2021-1234"},
            {"name": "func_b", "vector": [0.0, 1.0]},
            {"name": "func_c", "vector": [0.5, 0.5], "cve": "CVE-2022-5678"},
        ]
    }
    p = tmp_path / "with_cve.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    names, vectors, cves = load_embeddings(str(p))
    assert names == ["func_a", "func_b", "func_c"]
    assert len(cves) == len(names)
    assert cves[0] == ["CVE-2021-1234"]
    assert cves[1] == []
    assert cves[2] == ["CVE-2022-5678"]


def test_load_embeddings_cve_as_list(tmp_path):
    """cve 为列表时保留多元素。"""
    data = {
        "functions": [
            {
                "name": "multi",
                "vector": [1.0, 0.0],
                "cve": ["CVE-2021-A", "CVE-2021-B", ""],
            },
        ]
    }
    p = tmp_path / "cve_list.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    names, vectors, cves = load_embeddings(str(p))
    assert names == ["multi"]
    assert cves[0] == ["CVE-2021-A", "CVE-2021-B"]


def test_load_embeddings_fixture_fake_cve():
    """tests/fixtures/fake_cve 含非空 cve 列表，加载无异常。"""
    path = os.path.join(_project_root, "tests", "fixtures", "fake_cve", "library_embeddings.json")
    assert os.path.isfile(path)
    names, vectors, cves = load_embeddings(path)
    assert len(names) >= 1
    assert any(c for c in cves)
    assert vectors.shape[0] == len(names)


def test_compute_top_k():
    """固定小向量验证索引与分数顺序。"""
    query = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    db = np.array([
        [1.0, 0.0, 0.0],   # 0: 与 query0 相同
        [0.9, 0.1, 0.0],   # 1: 与 query0 相似
        [0.0, 1.0, 0.0],   # 2: 与 query1 相同
    ])
    indices, scores = compute_top_k(query, db, k=2)
    assert indices.shape == (2, 2)
    assert scores.shape == (2, 2)
    # query0 最相似为 db0(1.0), 其次 db1
    assert indices[0, 0] == 0
    assert abs(scores[0, 0] - 1.0) < 1e-5
    # query1 最相似为 db2(1.0), 其次 db1(较小)
    assert indices[1, 0] == 2
    assert abs(scores[1, 0] - 1.0) < 1e-5
    # 分数降序
    assert scores[0, 0] >= scores[0, 1]
    assert scores[1, 0] >= scores[1, 1]


def test_compute_metrics():
    """
    固定 query/db 名称、top_k_indices、relevant_pairs，
    人工算 Recall@K、Precision@K、MRR，断言一致。
    """
    query_names = ["a", "b", "c"]
    db_names = ["x", "a", "b", "y"]
    relevant_pairs = build_relevant_pairs(query_names, db_names)
    assert (0, 1) in relevant_pairs  # query "a" -> db "a" at index 1
    assert (1, 2) in relevant_pairs  # query "b" -> db "b" at index 2
    # query "c" 无匹配

    # top_k_indices: 每个 query 的 top-2 检索结果
    # query0: 假设 top-2 为 [1, 0] -> 命中 db1(a), relevant
    # query1: 假设 top-2 为 [0, 2] -> 未命中 rank1, 命中 rank2(b), relevant
    # query2: 假设 top-2 为 [0, 3] -> 无 relevant
    top_k_indices = np.array([
        [1, 0],  # query0: rank1=db1(a) hit
        [0, 2],  # query1: rank2=db2(b) hit
        [0, 3],  # query2: no hit
    ])
    m = compute_metrics(query_names, db_names, top_k_indices, relevant_pairs, k=2)

    # Recall@2: query0 有 hit -> 1, query1 有 hit -> 1, query2 无 hit -> 0; avg = 2/3
    assert abs(m["recall_at_k"] - 2.0 / 3.0) < 1e-6
    # Precision@2: query0 1/2, query1 1/2, query2 0/2; avg = (0.5+0.5+0)/3 = 1/3
    assert abs(m["precision_at_k"] - 1.0 / 3.0) < 1e-6
    # MRR: query0 first at rank1 -> 1/1=1, query1 first at rank2 -> 1/2=0.5, query2 -> 0; avg = (1+0.5+0)/3
    assert abs(m["mrr"] - (1.0 + 0.5 + 0) / 3.0) < 1e-6


def test_build_relevant_pairs_by_cve():
    """CVE 模式：query[0] 与 db[1] 同 CVE，验证 relevant_pairs 与 compute_metrics。"""
    query_cves = [["CVE-2021-1234"], []]
    db_cves = [[], ["CVE-2021-1234"], []]
    pairs = build_relevant_pairs_by_cve(query_cves, db_cves)
    assert (0, 1) in pairs
    assert len(pairs) == 1

    # 构造 top_k_indices：query0 top-2=[1,0] 命中 db1；query1 top-2=[0,2] 无 CVE 匹配
    query_names = ["a", "b"]
    db_names = ["x", "y", "z"]
    top_k_indices = np.array([[1, 0], [0, 2]])
    m = compute_metrics(query_names, db_names, top_k_indices, pairs, k=2)
    # Recall@2: query0 有 hit -> 1, query1 无 hit -> 0; avg = 0.5
    assert abs(m["recall_at_k"] - 0.5) < 1e-6
    # Precision@2: query0 1/2, query1 0/2; avg = 0.25
    assert abs(m["precision_at_k"] - 0.25) < 1e-6
    # MRR: query0 first at rank1 -> 1, query1 -> 0; avg = 0.5
    assert abs(m["mrr"] - 0.5) < 1e-6


def test_build_relevant_pairs_by_cve_list_intersection():
    """多条 CVE 时，集合交集非空即相关。"""
    query_cves = [["CVE-A", "CVE-B"]]
    db_cves = [["CVE-B", "CVE-C"]]
    pairs = build_relevant_pairs_by_cve(query_cves, db_cves)
    assert pairs == {(0, 0)}


def test_compute_metrics_all_hit():
    """所有 query 均在 rank1 命中的情况。"""
    query_names = ["a", "b"]
    db_names = ["x", "a", "b"]
    relevant_pairs = build_relevant_pairs(query_names, db_names)
    top_k_indices = np.array([[1, 0], [2, 0]])
    m = compute_metrics(query_names, db_names, top_k_indices, relevant_pairs, k=2)
    assert abs(m["recall_at_k"] - 1.0) < 1e-6
    assert abs(m["precision_at_k"] - 0.5) < 1e-6
    assert abs(m["mrr"] - 1.0) < 1e-6


def test_compute_top_k_empty():
    """空输入的边界情况。"""
    indices, scores = compute_top_k(np.array([]).reshape(0, 3), np.zeros((2, 3)), 1)
    assert indices.shape == (0, 1)
    assert scores.shape == (0, 1)
