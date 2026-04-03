"""测试两阶段粗筛：LibraryFaissIndex 与 retrieve_coarse。"""
import json
import os
import tempfile

import pytest

from matcher.faiss_library import LibraryFaissIndex, retrieve_coarse


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_library_faiss_index_self_query():
    """用库中向量的原始形式查询，自身应出现在结果中且分数接近 1（L2 归一化后内积=余弦）。"""
    emb = {
        "functions": [
            {"function_id": "a|0x1", "vector": [1.0] * 64 + [0.0] * 64},
            {"function_id": "b|0x2", "vector": [0.0] * 64 + [1.0] * 64},
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(emb, f)
        path = f.name
    try:
        idx = LibraryFaissIndex(path)
        # 用第一个向量查询
        r = idx.search(emb["functions"][0]["vector"], k=2)
        assert len(r) == 2
        assert r[0][0] == "a|0x1"
        assert r[0][1] > 0.99
    finally:
        os.unlink(path)


def test_library_faiss_index_empty():
    """空库搜索返回空列表。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"functions": []}, f)
        path = f.name
    try:
        idx = LibraryFaissIndex(path)
        r = idx.search([1.0] * 128, k=10)
        assert r == []
    finally:
        os.unlink(path)


def test_library_faiss_index_k_larger_than_library():
    """k 大于库大小时返回全部结果。"""
    emb = {
        "functions": [
            {"function_id": "x|0x1", "vector": [0.1] * 128},
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(emb, f)
        path = f.name
    try:
        idx = LibraryFaissIndex(path)
        r = idx.search([0.2] * 128, k=100)
        assert len(r) == 1
    finally:
        os.unlink(path)


def test_retrieve_coarse_empty_library():
    """空库时 retrieve_coarse 返回空列表。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"functions": []}, f)
        path = f.name
    try:
        idx = LibraryFaissIndex(path)
        # 最小 multimodal 特征
        mm = {
            "graph": {"num_nodes": 1, "node_features": [[]]},
            "sequence": {"pcode_tokens": ["COPY", "INT_SUB"], "jump_mask": [], "seq_len": 2},
        }
        r = retrieve_coarse(mm, idx, k=10)
        assert r == []
    finally:
        os.unlink(path)


def test_retrieve_coarse_returns_function_ids():
    """retrieve_coarse 返回 function_id 列表，数量不超过 k。"""
    emb = {
        "functions": [
            {"function_id": "p|0x1", "vector": [0.5] * 128},
            {"function_id": "q|0x2", "vector": [0.3] * 128},
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(emb, f)
        path = f.name
    try:
        idx = LibraryFaissIndex(path)
        mm = {
            "graph": {"num_nodes": 1, "node_features": [[]]},
            "sequence": {"pcode_tokens": ["STORE", "LOAD"], "jump_mask": [], "seq_len": 2},
        }
        r = retrieve_coarse(mm, idx, k=1)
        assert len(r) <= 1
        assert all(isinstance(x, str) for x in r)
    finally:
        os.unlink(path)


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(_project_root(), "data", "two_stage", "library_safe_embeddings.json")),
    reason="需先构建 data/two_stage/library_safe_embeddings.json",
)
def test_retrieve_coarse_integration():
    """集成测试：对有正样本的查询，retrieve_coarse 可返回候选（不保证含正样本，因 SAFE 未训练）。"""
    root = _project_root()
    emb_path = os.path.join(root, "data", "two_stage", "library_safe_embeddings.json")
    qf_path = os.path.join(root, "data", "two_stage", "query_features.json")
    gt_path = os.path.join(root, "data", "two_stage", "ground_truth.json")
    if not os.path.isfile(qf_path) or not os.path.isfile(gt_path):
        pytest.skip("query_features 或 ground_truth 不存在")

    with open(qf_path, encoding="utf-8") as f:
        qf = json.load(f)
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    query_ids = [qid for qid in gt if qid in qf and len(gt[qid]) > 0]
    if not query_ids:
        pytest.skip("无有效查询")

    idx = LibraryFaissIndex(emb_path)
    qid = query_ids[0]
    mm = qf[qid]
    cands = retrieve_coarse(mm, idx, k=100)
    assert len(cands) <= 100
    assert all(isinstance(c, str) and "|" in c for c in cands)
