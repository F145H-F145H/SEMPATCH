"""从 CFG/DFG 提取图特征。"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore


def extract_graph_features(
    lsir_func: Dict[str, Any],
    *,
    cfg_weight: float = 1.0,
    dfg_weight: float = 1.0,
) -> Dict[str, Any]:
    """
    从 LSIR 函数提取图特征（CFG/DFG）。
    返回可序列化结构，供模型或后续处理使用。
    """
    out: Dict[str, Any] = {"cfg": {}, "dfg": {}}

    cfg = lsir_func.get("cfg")
    if cfg is not None:
        if nx and isinstance(cfg, nx.DiGraph):
            out["cfg"] = {
                "num_nodes": cfg.number_of_nodes(),
                "num_edges": cfg.number_of_edges(),
                "adjacency": [
                    list(cfg.successors(n)) for n in sorted(cfg.nodes())
                ],
                "node_list": list(cfg.nodes()),
            }
        elif isinstance(cfg, dict):
            out["cfg"] = {
                "num_nodes": len(cfg.get("nodes", [])),
                "num_edges": len(cfg.get("edges", [])),
                "adjacency": [],
                "node_list": cfg.get("nodes", []),
            }

    dfg = lsir_func.get("dfg")
    if dfg is not None:
        if nx and isinstance(dfg, nx.DiGraph):
            out["dfg"] = {
                "num_nodes": dfg.number_of_nodes(),
                "num_edges": dfg.number_of_edges(),
                "adjacency": [
                    list(dfg.successors(n)) for n in sorted(dfg.nodes())
                ],
                "node_list": list(dfg.nodes()),
            }
        elif isinstance(dfg, dict):
            out["dfg"] = {
                "num_nodes": len(set(a for a, _ in dfg.get("edges", [])) | set(b for _, b in dfg.get("edges", []))),
                "num_edges": len(dfg.get("edges", [])),
                "adjacency": [],
                "node_list": list(set(a for a, _ in dfg.get("edges", [])) | set(b for _, b in dfg.get("edges", []))),
            }

    out["cfg_weight"] = cfg_weight
    out["dfg_weight"] = dfg_weight

    # 部分提取告警：CFG 为空但 DFG 非空
    if not out["cfg"] and out.get("dfg") and out["dfg"].get("num_nodes", 0) > 0:
        logger.warning("CFG 为空但 DFG 非空（%d 节点），图特征为部分提取", out["dfg"]["num_nodes"])

    return out


def extract_acfg_features(lsir_func: Dict[str, Any]) -> Dict[str, Any]:
    """
    ACFG 特征：在 CFG 基础上增加基本块级属性（Genius/Gemini）。
    返回 {num_nodes, num_edges, node_features: [{inst_count, pcode_opcodes: [...]}, ...]}。
    """
    out: Dict[str, Any] = {"num_nodes": 0, "num_edges": 0, "node_features": []}

    cfg = lsir_func.get("cfg")
    bbs = lsir_func.get("basic_blocks", []) or []

    node_list: List[str] = []
    if cfg is not None:
        try:
            import networkx as nx
            if isinstance(cfg, nx.DiGraph):
                out["num_nodes"] = cfg.number_of_nodes()
                out["num_edges"] = cfg.number_of_edges()
                node_list = list(cfg.nodes())
        except ImportError:
            pass
        if isinstance(cfg, dict):
            edges = cfg.get("edges", [])
            nodes = cfg.get("nodes", [])
            out["num_nodes"] = len(nodes)
            out["num_edges"] = len(edges)
            node_list = nodes or list(set(a for a, _ in edges) | set(b for _, b in edges))

    for i, bb in enumerate(bbs):
        insts = bb.get("instructions", []) or []
        inst_count = len(insts)
        pcode_opcodes: List[str] = []
        for inst in insts:
            for pco in inst.get("pcode", []) or []:
                opcode = (pco.get("opcode") or "").strip()
                if opcode:
                    pcode_opcodes.append(opcode)
        out["node_features"].append({
            "inst_count": inst_count,
            "pcode_opcodes": pcode_opcodes,
            "pcode_len": len(pcode_opcodes),
        })

    if not out["node_features"] and node_list:
        for _ in node_list:
            out["node_features"].append({"inst_count": 0, "pcode_opcodes": [], "pcode_len": 0})

    return out
