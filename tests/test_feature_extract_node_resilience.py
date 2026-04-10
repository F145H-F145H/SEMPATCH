"""测试 FeatureExtractNode 在部分函数提取失败时的容错行为。"""

from unittest.mock import MagicMock, patch


def test_feature_extract_skips_bad_function(caplog):
    """包含一个坏函数时，节点不崩溃，输出中跳过该函数。"""
    import logging

    from dag.nodes.feature_extract_node import FeatureExtractNode

    lsir = {
        "functions": [
            {
                "name": "good_func",
                "basic_blocks": [
                    {"instructions": [{"pcode": [{"opcode": "COPY"}]}]},
                ],
                "dfg": {"edges": []},
            },
            {
                "name": "bad_func",
                "basic_blocks": [
                    {"instructions": [{"pcode": [{"opcode": "INT_ADD"}]}]},
                ],
                "dfg": {"edges": []},
            },
        ]
    }

    # 先导入真实函数，再构造 mock
    from utils.feature_extractors import extract_graph_features as real_extract_graph

    def _mock_graph(fn):
        if fn.get("name") == "bad_func":
            raise ValueError("模拟 CFG 提取失败")
        return real_extract_graph(fn)

    ctx = {"lsir": lsir}
    node = FeatureExtractNode(node_id="test_fe", node_type="feature_extract", params={}, deps=[])

    # patch execute 内部的本地导入 —— 由于 execute 用 from-import，
    # 需要在 utils.feature_extractors 模块级替换
    with caplog.at_level(logging.WARNING, logger="dag.nodes.feature_extract_node"), \
         patch("utils.feature_extractors.extract_graph_features", side_effect=_mock_graph):
        node.execute(ctx)

    assert node.done
    output = ctx["features"]
    func_names = [f["name"] for f in output["functions"]]
    assert "good_func" in func_names
    assert "bad_func" not in func_names
    assert any("bad_func" in r.message for r in caplog.records)


def test_feature_extract_empty_funcs():
    """空函数列表不崩溃。"""
    from dag.nodes.feature_extract_node import FeatureExtractNode

    lsir = {"functions": []}
    ctx = {"lsir": lsir}
    node = FeatureExtractNode(node_id="test_fe", node_type="feature_extract", params={}, deps=[])
    node.execute(ctx)
    assert node.done
    assert ctx["features"]["functions"] == []
