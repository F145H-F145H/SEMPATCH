"""
多模态融合（survey 5.1）：图特征 + 序列特征 -> 规范化多模态输入。
输出可被 MultiModalFusionModel 消费的格式：图邻接、节点特征、序列 token、跳转 mask；
可选 DFG 子图（阶段 H，与 graph 同 schema）。
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

# 与 MultiModalFusionModel 节点嵌入表一致；0=pad，1=unk，DFG 节点用 [2, size-1]
_DFG_NODE_EMBED_TABLE_SIZE = 512


def _stable_dfg_node_feature_id(node_key: str) -> int:
    """DFG 节点字符串 -> 稳定整数 id，落入 node_embed 表可用范围。"""
    h = int(hashlib.md5(node_key.encode("utf-8")).hexdigest(), 16)
    return 2 + (h % (_DFG_NODE_EMBED_TABLE_SIZE - 2))


def fuse_features(
    graph_feats: Dict[str, Any],
    seq_feats: Dict[str, Any],
    *,
    graph_weight: float = 0.5,
    seq_weight: float = 0.5,
    acfg_feats: Optional[Dict[str, Any]] = None,
    include_dfg: bool = True,
    max_dfg_nodes: int = 128,
) -> Dict[str, Any]:
    """
    融合图特征与序列特征，输出统一表示。
    扩展为规范化多模态格式，供模型消费。
    acfg_feats: 可选，用于图分支的 ACFG（块级节点+特征）；无则用 graph_feats 的 CFG。
    include_dfg: 为 True 时写入 multimodal.dfg（无数据则为空图）；False 时不含 dfg 键。
    """
    graph_for_model = _build_graph_for_model(graph_feats, acfg_feats)
    seq_for_model = _build_sequence_for_model(seq_feats)

    multimodal: Dict[str, Any] = {
        "graph": graph_for_model,
        "sequence": seq_for_model,
    }
    if include_dfg:
        multimodal["dfg"] = _build_dfg_for_model(graph_feats, max_dfg_nodes=max_dfg_nodes)

    return {
        "graph": graph_feats,
        "sequence": seq_feats,
        "graph_weight": graph_weight,
        "seq_weight": seq_weight,
        "fused": True,
        "multimodal": multimodal,
    }


def _build_graph_for_model(
    graph_feats: Dict[str, Any],
    acfg_feats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    构建图分支输入：ACFG 风格（块级节点+邻接+节点特征）。
    节点特征来自 ACFG 的 pcode_opcodes；无 ACFG 时用 CFG 结构。
    """
    cfg = graph_feats.get("cfg") or {}
    cfg_node_list = cfg.get("node_list", [])
    cfg_adjacency = cfg.get("adjacency", [])

    if acfg_feats:
        node_features_raw = acfg_feats.get("node_features") or []
        node_list = cfg_node_list or [
            f"bb_{i}" for i in range(len(node_features_raw))
        ]
        # 节点特征：pcode_opcodes 列表（模型将用 vocab 转为 id）
        node_features = [
            nf.get("pcode_opcodes", []) or []
            for nf in node_features_raw
        ]
    else:
        node_list = cfg_node_list
        node_features = [[] for _ in node_list]

    node_to_idx = {n: i for i, n in enumerate(node_list)}
    edges_src: List[int] = []
    edges_dst: List[int] = []
    # adjacency[i] 对应 graph_features 中 sorted(cfg.nodes())[i]
    sorted_cfg = sorted(node_list) if node_list else []
    for i, succs in enumerate(cfg_adjacency):
        if i >= len(sorted_cfg):
            continue
        src_node = sorted_cfg[i]
        src = node_to_idx.get(src_node)
        if src is None:
            continue
        for s in succs:
            dst = node_to_idx.get(s)
            if dst is not None:
                edges_src.append(src)
                edges_dst.append(dst)

    return {
        "num_nodes": len(node_list),
        "edge_index": [edges_src, edges_dst],
        "node_list": node_list,
        "node_features": node_features,
    }


def _build_dfg_for_model(
    graph_feats: Dict[str, Any],
    *,
    max_dfg_nodes: int = 128,
) -> Dict[str, Any]:
    """
    构建 DFG 分支输入；与 multimodal.graph 同形。
    graph_feats.dfg 来自 extract_graph_features；adjacency 与 sorted(node_list) 对齐。
    """
    dfg = graph_feats.get("dfg") or {}
    raw_nodes = list(dfg.get("node_list") or [])
    if not raw_nodes:
        return {
            "num_nodes": 0,
            "edge_index": [[], []],
            "node_list": [],
            "node_features": [],
        }

    sorted_all = sorted(raw_nodes)
    node_list = sorted_all[:max_dfg_nodes]
    allowed = set(node_list)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    edges_src: List[int] = []
    edges_dst: List[int] = []

    adj = dfg.get("adjacency") or []
    if adj:
        for i, succs in enumerate(adj):
            if i >= len(sorted_all):
                break
            src_node = sorted_all[i]
            if src_node not in allowed:
                continue
            src = node_to_idx[src_node]
            for s in succs:
                if s in allowed:
                    dst = node_to_idx.get(s)
                    if dst is not None:
                        edges_src.append(src)
                        edges_dst.append(dst)
    else:
        for a, b in dfg.get("edges", []) or []:
            if a in allowed and b in allowed:
                sa, sb = node_to_idx.get(a), node_to_idx.get(b)
                if sa is not None and sb is not None:
                    edges_src.append(sa)
                    edges_dst.append(sb)

    node_features = [_stable_dfg_node_feature_id(n) for n in node_list]
    return {
        "num_nodes": len(node_list),
        "edge_index": [edges_src, edges_dst],
        "node_list": node_list,
        "node_features": node_features,
    }


def _build_sequence_for_model(seq_feats: Dict[str, Any]) -> Dict[str, Any]:
    """
    构建序列分支输入：P-code token 序列、跳转 mask（jTrans 风格）。
    jump_mask 与 pcode_seq 对齐：pcode op 本身为控制流相关则为 1。
    """
    pcode_seq = seq_feats.get("pcode_seq") or []
    control_ops = {"BRANCH", "CBRANCH", "BRANCHIND", "CALL", "CALLIND", "RETURN"}
    jump_mask = [1 if (op or "").upper() in control_ops else 0 for op in pcode_seq]
    return {
        "pcode_tokens": pcode_seq,
        "jump_mask": jump_mask,
        "seq_len": len(pcode_seq),
    }
