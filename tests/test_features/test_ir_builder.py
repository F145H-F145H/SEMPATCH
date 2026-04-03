"""测试 ir_builder。"""
from utils.ir_builder import build_lsir


def test_build_lsir_empty():
    out = build_lsir({"functions": []})
    assert "functions" in out
    assert out["functions"] == []


def test_build_lsir_simple():
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
                                "pcode": [{"opcode": "COPY", "output": "v1", "inputs": ["v0"]}],
                            }
                        ],
                    }
                ],
            }
        ]
    }
    out = build_lsir(raw)
    assert len(out["functions"]) == 1
    fn = out["functions"][0]
    assert "cfg" in fn
    assert "dfg" in fn


def test_build_lsir_dfg_cross_instruction():
    """DFG 沿基本块内指令顺序传播：后一条指令的 input 连到前一条的 def 节点。"""
    raw = {
        "functions": [
            {
                "name": "f",
                "entry": "0x1000",
                "basic_blocks": [
                    {
                        "start": "0x1000",
                        "instructions": [
                            {
                                "address": "0x1000",
                                "mnemonic": "X",
                                "pcode": [
                                    {
                                        "opcode": "COPY",
                                        "output": "(register, 0x10, 8)",
                                        "inputs": ["(const, 0x1, 8)"],
                                    }
                                ],
                            },
                            {
                                "address": "0x1008",
                                "mnemonic": "Y",
                                "pcode": [
                                    {
                                        "opcode": "INT_ADD",
                                        "output": "(register, 0x20, 8)",
                                        "inputs": [
                                            "(register, 0x10, 8)",
                                            "(const, 0x2, 8)",
                                        ],
                                    }
                                ],
                            },
                        ],
                    }
                ],
            }
        ]
    }
    out = build_lsir(raw)
    fn = out["functions"][0]
    dfg = fn["dfg"]
    try:
        import networkx as nx

        assert isinstance(dfg, nx.DiGraph)
        def_a = "0x1000:(register, 0x10, 8)"
        def_b = "0x1008:(register, 0x20, 8)"
        assert dfg.has_edge(def_a, def_b)
    except ImportError:
        edges = dfg.get("edges", [])
        assert ("0x1000:(register, 0x10, 8)", "0x1008:(register, 0x20, 8)") in edges


def test_build_lsir_include_dfg_false_still_has_dfg_key():
    out = build_lsir({"functions": [{"name": "g", "entry": "0x0", "basic_blocks": []}]}, include_dfg=False)
    fn = out["functions"][0]
    assert "dfg" in fn
    try:
        import networkx as nx

        assert isinstance(fn["dfg"], nx.DiGraph)
        assert fn["dfg"].number_of_nodes() == 0
    except ImportError:
        assert fn["dfg"] == {"edges": []}
