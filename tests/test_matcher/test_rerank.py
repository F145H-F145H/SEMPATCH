"""测试两阶段精排：load_candidate_features 与 compute_rerank_scores。"""

import json
import math
import os
import tempfile

import pytest

from matcher.rerank import compute_rerank_scores, load_candidate_features


def _minimal_multimodal(
    pcode_tokens=None,
    node_features=None,
):
    """构造最小 multimodal 特征，供 compute_rerank_scores 使用。"""
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


def test_compute_rerank_scores_same_function():
    """同一函数特征两次传入，得分应接近 1。"""
    mm = _minimal_multimodal()
    candidates = [("id_a", mm)]
    scores = compute_rerank_scores(mm, candidates)
    assert len(scores) == 1
    assert scores[0][0] == "id_a"
    assert 0.99 <= scores[0][1] <= 1.01


def test_compute_rerank_scores_positive_vs_negative():
    """正样本对（相同特征）得分应高于负样本对（不同特征）。"""
    mm_same = _minimal_multimodal(pcode_tokens=["COPY", "LOAD", "STORE"])
    mm_diff = _minimal_multimodal(
        pcode_tokens=["BRANCH", "CBRANCH", "RETURN"],
        node_features=[[], []],
    )
    pos_scores = compute_rerank_scores(mm_same, [("pos", mm_same)])
    neg_scores = compute_rerank_scores(mm_same, [("neg", mm_diff)])
    assert len(pos_scores) == 1 and len(neg_scores) == 1
    assert pos_scores[0][1] >= neg_scores[0][1]


def test_compute_rerank_scores_ordering():
    """10 个候选返回 10 个结果，且按 score 降序。"""
    base = _minimal_multimodal()
    candidates = [(f"cand_{i}", base) for i in range(10)]
    scores = compute_rerank_scores(base, candidates)
    assert len(scores) == 10
    assert all(s[0].startswith("cand_") for s in scores)
    for i in range(9):
        assert scores[i][1] >= scores[i + 1][1]


def test_compute_rerank_scores_empty():
    """空候选列表返回空列表。"""
    mm = _minimal_multimodal()
    scores = compute_rerank_scores(mm, [])
    assert scores == []


def test_compute_rerank_scores_empty_pcode_sequence():
    """无 pcode_tokens 时不应触发 Transformer 空序列 / to_padded_tensor 错误。"""
    mm = _minimal_multimodal(pcode_tokens=[])
    cand = _minimal_multimodal(pcode_tokens=["COPY", "INT_ADD"])
    scores = compute_rerank_scores(mm, [("c1", cand)])
    assert len(scores) == 1
    assert scores[0][0] == "c1"
    assert not math.isnan(scores[0][1])


def test_load_candidate_features_found():
    """对存在的 candidate_id 返回有效 multimodal。"""
    lib = {
        "a|0x1": _minimal_multimodal(pcode_tokens=["A", "B"]),
        "b|0x2": _minimal_multimodal(pcode_tokens=["C", "D"]),
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(lib, f)
        path = f.name
    try:
        result = load_candidate_features(["a|0x1", "b|0x2"], path)
        assert len(result) == 2
        ids = [r[0] for r in result]
        assert "a|0x1" in ids and "b|0x2" in ids
        for cid, mm in result:
            assert "graph" in mm and "sequence" in mm
            assert mm["sequence"]["pcode_tokens"]
    finally:
        os.unlink(path)


def test_load_candidate_features_missing():
    """对不存在的 id 跳过，返回仅包含存在的 id。"""
    lib = {"x|0x1": _minimal_multimodal()}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(lib, f)
        path = f.name
    try:
        result = load_candidate_features(["x|0x1", "nonexistent|0x99"], path)
        assert len(result) == 1
        assert result[0][0] == "x|0x1"
    finally:
        os.unlink(path)


def test_load_candidate_features_file_not_found():
    """文件不存在时抛出 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError, match="不存在"):
        load_candidate_features(["a|0x1"], "/nonexistent/path/library_features.json")


def test_load_candidate_features_invalid_format():
    """非 dict 格式时抛出 ValueError。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([{"a": 1}], f)
        path = f.name
    try:
        with pytest.raises(ValueError, match="格式应为"):
            load_candidate_features(["a|0x1"], path)
    finally:
        os.unlink(path)
