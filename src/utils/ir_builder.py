"""
从 Ghidra lsir_raw 结构构建 LSIR（CFG/DFG）。
输入：in-memory dict（等同 lsir_raw.json 解析结果）。
输出：LSIR 结构，含 CFG、DFG，兼容后续扩展字段。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore


def _get_block_id(bb: Dict[str, Any], idx: int) -> str:
    """为基本块生成唯一 ID。"""
    start = bb.get("start", "")
    return f"bb_{idx}_{start}"


def _extract_cfg_edges(
    basic_blocks: List[Dict[str, Any]], func_name: str
) -> List[tuple]:
    """
    从基本块序列推断 CFG 边。
    按地址顺序，添加 fall-through 边；跳转边需从指令解析（后续扩展）。
    """
    edges = []
    for i, bb in enumerate(basic_blocks):
        bid = _get_block_id(bb, i)
        # fall-through：下一块
        if i + 1 < len(basic_blocks):
            next_bb = basic_blocks[i + 1]
            next_bid = _get_block_id(next_bb, i + 1)
            edges.append((bid, next_bid))
        # 从最后一条指令的跳转目标推断（简单启发式）
        insts = bb.get("instructions", [])
        for inst in reversed(insts):
            mnemonic = (inst.get("mnemonic") or "").upper()
            operands = inst.get("operands", "") or ""
            if mnemonic in ("BRANCH", "CBRANCH", "CALL", "BRANCHIND", "CALLIND"):
                # 可解析 operands 中的地址，此处暂跳过，兼容扩展
                break
    return edges


def _extract_dfg_edges(instructions: List[Dict[str, Any]]) -> List[tuple]:
    """
    从 P-code 提取 DFG 边，方向 dataflow：src → dst（use/defs 沿执行顺序传播）。

    在**基本块拼接顺序**下按指令、再按单条指令内 pcode 顺序扫描；对每个带 output 的
    pcode，将每个 input varnode 连到当前 def 节点：若该 varnode 在扫描路径上已有最近定值，
    则边从「最近定值节点」出发，否则从「当前地址上的使用该 varnode 的占位节点」出发
    （与历史「addr:varnode」节点 ID 空间一致）。不建模 PHI/汇合，非线性 CFG 上的精度为启发式。
    """
    edges: List[tuple] = []
    seen: Set[tuple] = set()
    last_def: Dict[str, str] = {}

    for inst in instructions:
        addr = str(inst.get("address", ""))
        for pco in inst.get("pcode", []) or []:
            out_var = pco.get("output")
            in_vars = pco.get("inputs", []) or []
            if not out_var:
                continue
            out_node = f"{addr}:{out_var}"
            for inv in in_vars:
                if not inv or not isinstance(inv, str):
                    continue
                src = last_def.get(inv)
                if src is None:
                    src = f"{addr}:{inv}"
                key = (src, out_node)
                if key not in seen:
                    seen.add(key)
                    edges.append(key)
            last_def[str(out_var)] = out_node
    return edges


def build_lsir(
    lsir_raw: Dict[str, Any],
    *,
    include_cfg: bool = True,
    include_dfg: bool = True,
) -> Dict[str, Any]:
    """
    从 lsir_raw（Ghidra 输出的 in-memory 结构）构建 LSIR。

    兼容扩展：未识别的顶层键和嵌套字段会保留到 output 中。

    Returns:
        {
            "functions": [
                {
                    "name": str,
                    "entry": str,
                    "basic_blocks": [...],  # 原始结构保留
                    "cfg": nx.DiGraph or dict,  # 可选（include_cfg=False 时不写入）
                    "dfg": nx.DiGraph or dict,  # 必有；include_dfg=False 时为空图
                    **extra,  # 扩展字段
                }
            ],
            **extra,
        }
    """
    funcs_raw = lsir_raw.get("functions", [])
    if not isinstance(funcs_raw, list):
        funcs_raw = []

    result: Dict[str, Any] = {}
    for k, v in lsir_raw.items():
        if k != "functions":
            result[k] = v

    result["functions"] = []
    for fi, f in enumerate(funcs_raw):
        if not isinstance(f, dict):
            continue
        fn_out: Dict[str, Any] = {}
        for k, v in f.items():
            fn_out[k] = v

        bbs = f.get("basic_blocks", [])
        if not isinstance(bbs, list):
            bbs = []

        if include_cfg and bbs:
            cfg_edges = _extract_cfg_edges(bbs, f.get("name", ""))
            if nx:
                g = nx.DiGraph()
                for bid, _ in enumerate(bbs):
                    g.add_node(_get_block_id(bbs[bid], bid))
                g.add_edges_from(cfg_edges)
                fn_out["cfg"] = g
            else:
                fn_out["cfg"] = {"edges": cfg_edges, "nodes": [_get_block_id(bbs[i], i) for i in range(len(bbs))]}

        # LSIR 契约：dfg 始终存在；include_dfg=False 时为空图（不计算跨指令边）
        if include_dfg:
            all_insts: List[Dict[str, Any]] = []
            for bb in bbs:
                for inst in bb.get("instructions", []) or []:
                    all_insts.append(inst)
            dfg_edges = _extract_dfg_edges(all_insts)
            if nx:
                g = nx.DiGraph()
                nodes = set()
                for a, b in dfg_edges:
                    nodes.add(a)
                    nodes.add(b)
                g.add_nodes_from(nodes)
                g.add_edges_from(dfg_edges)
                fn_out["dfg"] = g
            else:
                fn_out["dfg"] = {"edges": dfg_edges}
        else:
            if nx:
                fn_out["dfg"] = nx.DiGraph()
            else:
                fn_out["dfg"] = {"edges": []}

        result["functions"].append(fn_out)

    return result
