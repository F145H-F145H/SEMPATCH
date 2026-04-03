"""特征提取模块单元测试。验证输出符合 FeaturesItem.multimodal schema。"""
import pytest

from utils.ir_builder import build_lsir
from utils.pcode_normalizer import normalize_lsir_raw
from utils.feature_extractors import (
    extract_acfg_features,
    extract_graph_features,
    extract_sequence_features,
    fuse_features,
    extract_multimodal_from_lsir_raw,
)


def _make_lsir_func():
    """构建最小 LSIR 函数（含 cfg、dfg、basic_blocks、pcode）。"""
    raw = {
        "functions": [
            {
                "name": "main",
                "entry": "0x1000",
                "basic_blocks": [
                    {
                        "start": "0x1000",
                        "end": "0x1010",
                        "instructions": [
                            {
                                "address": "0x1000",
                                "mnemonic": "COPY",
                                "pcode": [
                                    {"opcode": "COPY", "output": "(register, 0x0, 8)", "inputs": ["(const, 0x0, 8)"]},
                                ],
                            },
                            {
                                "address": "0x1008",
                                "mnemonic": "CBRANCH",
                                "pcode": [
                                    {"opcode": "CBRANCH", "output": None, "inputs": ["(unique, 0x100, 1)"]},
                                ],
                            },
                        ],
                    },
                    {
                        "start": "0x1010",
                        "end": "0x1020",
                        "instructions": [
                            {
                                "address": "0x1010",
                                "mnemonic": "RETURN",
                                "pcode": [{"opcode": "RETURN", "output": None, "inputs": []}],
                            },
                        ],
                    },
                ],
            },
        ],
    }
    lsir = build_lsir(raw)
    return lsir["functions"][0]


def test_extract_graph_features():
    """extract_graph_features 返回 cfg/dfg 的 num_nodes、adjacency、node_list。"""
    fn = _make_lsir_func()
    out = extract_graph_features(fn)
    assert "cfg" in out
    assert "dfg" in out
    assert "num_nodes" in out["cfg"] or "adjacency" in out["cfg"]
    assert "node_list" in out["cfg"]
    assert "cfg_weight" in out
    assert "dfg_weight" in out
    assert isinstance(out["cfg"].get("adjacency", []), list)
    assert isinstance(out["cfg"].get("node_list", []), list)


def test_extract_sequence_features():
    """extract_sequence_features 返回 pcode_seq、mnemonic_seq、seq_len。"""
    fn = _make_lsir_func()
    out = extract_sequence_features(fn)
    assert "pcode_seq" in out
    assert "mnemonic_seq" in out
    assert "seq_len" in out
    assert isinstance(out["pcode_seq"], list)
    assert isinstance(out["mnemonic_seq"], list)
    assert out["seq_len"] == len(out["pcode_seq"])
    assert "COPY" in out["pcode_seq"]
    assert "CBRANCH" in out["pcode_seq"]
    assert "RETURN" in out["pcode_seq"]


def test_extract_acfg_features():
    """extract_acfg_features 返回 node_features（含 pcode_opcodes）。"""
    fn = _make_lsir_func()
    out = extract_acfg_features(fn)
    assert "num_nodes" in out
    assert "num_edges" in out
    assert "node_features" in out
    assert isinstance(out["node_features"], list)
    assert len(out["node_features"]) >= 1
    for nf in out["node_features"]:
        assert "inst_count" in nf
        assert "pcode_opcodes" in nf
        assert isinstance(nf["pcode_opcodes"], list)


def test_fuse_features_multimodal_schema():
    """fuse_features 返回 multimodal.graph / dfg / sequence 符合 schema。"""
    fn = _make_lsir_func()
    gf = extract_graph_features(fn)
    sf = extract_sequence_features(fn)
    acfg = extract_acfg_features(fn)
    fused = fuse_features(gf, sf, acfg_feats=acfg)

    assert "multimodal" in fused
    mm = fused["multimodal"]
    assert "graph" in mm
    assert "sequence" in mm
    assert "dfg" in mm

    dfg = mm["dfg"]
    assert "num_nodes" in dfg
    assert "edge_index" in dfg
    assert "node_list" in dfg
    assert "node_features" in dfg
    assert len(dfg["edge_index"]) == 2
    assert dfg["num_nodes"] == len(dfg["node_list"])
    assert len(dfg["node_features"]) == dfg["num_nodes"]
    for nid in dfg["node_features"]:
        assert isinstance(nid, int)
        assert 2 <= nid < 512

    graph = mm["graph"]
    assert "num_nodes" in graph
    assert "edge_index" in graph
    assert "node_list" in graph
    assert "node_features" in graph
    assert isinstance(graph["edge_index"], list)
    assert len(graph["edge_index"]) == 2
    assert isinstance(graph["node_features"], list)
    assert len(graph["node_features"]) == graph["num_nodes"]

    seq = mm["sequence"]
    assert "pcode_tokens" in seq
    assert "jump_mask" in seq
    assert "seq_len" in seq
    assert len(seq["jump_mask"]) == seq["seq_len"]
    assert len(seq["pcode_tokens"]) == seq["seq_len"]


def test_fuse_features_without_acfg():
    """fuse_features 无 acfg_feats 时仍产出有效 multimodal。"""
    fn = _make_lsir_func()
    gf = extract_graph_features(fn)
    sf = extract_sequence_features(fn)
    fused = fuse_features(gf, sf, acfg_feats=None)
    graph = fused["multimodal"]["graph"]
    assert "num_nodes" in graph
    assert "edge_index" in graph
    assert "node_features" in graph
    assert graph["num_nodes"] >= 0


def test_fuse_features_include_dfg_false():
    """include_dfg=False 时不写入 multimodal.dfg。"""
    fn = _make_lsir_func()
    gf = extract_graph_features(fn)
    sf = extract_sequence_features(fn)
    fused = fuse_features(gf, sf, acfg_feats=None, include_dfg=False)
    assert "dfg" not in fused["multimodal"]


def test_fuse_features_empty_dfg():
    """无 DFG 图数据时 multimodal.dfg 为空图。"""
    fn = _make_lsir_func()
    gf = extract_graph_features(fn)
    gf = {**gf, "dfg": {}}
    sf = extract_sequence_features(fn)
    fused = fuse_features(gf, sf, acfg_feats=None)
    dfg = fused["multimodal"]["dfg"]
    assert dfg["num_nodes"] == 0
    assert dfg["node_list"] == []
    assert dfg["edge_index"] == [[], []]


def test_extract_multimodal_from_lsir_raw_found():
    """extract_multimodal_from_lsir_raw 从 lsir_raw 正确提取 multimodal。"""
    raw_func = {
        "name": "main",
        "entry": "0x1000",
        "basic_blocks": [
            {
                "start": "0x1000",
                "end": "0x1010",
                "instructions": [
                    {
                        "address": "0x1000",
                        "mnemonic": "COPY",
                        "pcode": [
                            {"opcode": "COPY", "output": "(register, 0x0, 8)", "inputs": ["(const, 0x0, 8)"]},
                        ],
                    },
                    {
                        "address": "0x1008",
                        "mnemonic": "CBRANCH",
                        "pcode": [
                            {"opcode": "CBRANCH", "output": None, "inputs": ["(unique, 0x100, 1)"]},
                        ],
                    },
                ],
            },
        ],
    }
    raw = normalize_lsir_raw({"functions": [raw_func]})
    lsir_raw_funcs = raw["functions"]

    mm = extract_multimodal_from_lsir_raw(lsir_raw_funcs, "0x1000")
    assert "graph" in mm
    assert "sequence" in mm
    assert "pcode_tokens" in mm["sequence"]
    assert len(mm["sequence"]["pcode_tokens"]) >= 2


def test_extract_multimodal_from_lsir_raw_entry_variants():
    """extract_multimodal_from_lsir_raw 接受多种 entry 格式。"""
    raw_func = {
        "name": "f",
        "entry": "0x101000",
        "basic_blocks": [
            {
                "start": "0x101000",
                "end": "0x101010",
                "instructions": [
                    {
                        "address": "0x101000",
                        "mnemonic": "COPY",
                        "pcode": [{"opcode": "COPY", "output": "(register, 0x0, 8)", "inputs": []}],
                    },
                ],
            },
        ],
    }
    raw = normalize_lsir_raw({"functions": [raw_func]})
    lsir_raw_funcs = raw["functions"]

    mm1 = extract_multimodal_from_lsir_raw(lsir_raw_funcs, "0x101000")
    mm2 = extract_multimodal_from_lsir_raw(lsir_raw_funcs, "101000")
    assert mm1["sequence"]["pcode_tokens"] == mm2["sequence"]["pcode_tokens"]


def test_extract_multimodal_from_lsir_raw_not_found():
    """extract_multimodal_from_lsir_raw 未找到时抛出 ValueError。"""
    raw_func = {"name": "f", "entry": "0x1000", "basic_blocks": []}
    raw = normalize_lsir_raw({"functions": [raw_func]})
    lsir_raw_funcs = raw["functions"]

    with pytest.raises(ValueError, match="未找到"):
        extract_multimodal_from_lsir_raw(lsir_raw_funcs, "0x9999")
