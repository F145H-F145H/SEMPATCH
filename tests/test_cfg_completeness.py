"""测试 CFG 为空时的行为与 warning 日志。"""

import logging

from utils.feature_extractors.graph_features import extract_graph_features


def test_empty_cfg_returns_empty_dict():
    """CFG 为 None 时返回空 cfg 子 dict，不崩溃。"""
    lsir = {"cfg": None, "dfg": None}
    out = extract_graph_features(lsir)
    assert out["cfg"] == {}
    assert out["dfg"] == {}


def test_empty_cfg_nonempty_dfg_warns(caplog):
    """CFG 为空但 DFG 非空时发出 WARNING 日志。"""
    lsir = {
        "cfg": None,
        "dfg": {"edges": [("a", "b"), ("b", "c")]},
    }
    with caplog.at_level(logging.WARNING, logger="utils.feature_extractors.graph_features"):
        out = extract_graph_features(lsir)
    assert out["cfg"] == {}
    assert out["dfg"]["num_nodes"] > 0
    assert any("部分提取" in r.message for r in caplog.records)


def test_both_present_no_warning(caplog):
    """CFG/DFG 都存在时不发出 warning。"""
    lsir = {
        "cfg": {"nodes": ["a", "b"], "edges": [("a", "b")]},
        "dfg": {"edges": [("a", "b")]},
    }
    with caplog.at_level(logging.WARNING, logger="utils.feature_extractors.graph_features"):
        out = extract_graph_features(lsir)
    assert out["cfg"]["num_nodes"] == 2
    assert not any("部分提取" in r.message for r in caplog.records)
