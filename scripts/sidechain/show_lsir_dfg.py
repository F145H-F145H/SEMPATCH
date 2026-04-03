#!/usr/bin/env python3
"""
从 lsir_raw.json 构建 LSIR 并在 stdout 打印指定函数的 cfg/dfg 摘要（dfg 为 JSON 可序列化边列表）。

用法:
  PYTHONPATH=src python scripts/sidechain/show_lsir_dfg.py path/to/lsir_raw.json
  PYTHONPATH=src python scripts/sidechain/show_lsir_dfg.py path/to/lsir_raw.json --name main
  PYTHONPATH=src python scripts/sidechain/show_lsir_dfg.py path/to/lsir_raw.json --max-edges 50
  # --max-edges 同时限制 DFG 与 CFG 的 edges_preview 条数；仅限制一侧可用 --max-dfg-edges / --max-cfg-edges
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def _graph_to_dict(g) -> dict:
    if g is None:
        return {"nodes": [], "edges": []}
    try:
        import networkx as nx

        if isinstance(g, nx.DiGraph):
            return {
                "nodes": list(g.nodes()),
                "edges": [[str(a), str(b)] for a, b in g.edges()],
            }
    except ImportError:
        pass
    if isinstance(g, dict):
        edges = g.get("edges", [])
        if edges and isinstance(edges[0], (list, tuple)) and len(edges[0]) == 2:
            return {"nodes": g.get("nodes", []), "edges": [[str(a), str(b)] for a, b in edges]}
        return g
    return {"nodes": [], "edges": []}


def main() -> None:
    parser = argparse.ArgumentParser(description="lsir_raw → LSIR，打印 dfg/cfg 摘要")
    parser.add_argument("lsir_raw", help="lsir_raw.json 路径")
    parser.add_argument("--name", default=None, help="按函数名子串匹配（不区分大小写）；默认第一个函数")
    parser.add_argument(
        "--max-edges",
        type=int,
        default=80,
        help="DFG/CFG 边预览共用上限（若未单独指定 --max-dfg-edges / --max-cfg-edges）",
    )
    parser.add_argument(
        "--max-dfg-edges",
        type=int,
        default=None,
        help="仅 DFG edges_preview 上限（默认与 --max-edges 相同）",
    )
    parser.add_argument(
        "--max-cfg-edges",
        type=int,
        default=None,
        help="仅 CFG edges_preview 上限（默认与 --max-edges 相同）",
    )
    parser.add_argument(
        "--max-nodes-preview",
        type=int,
        default=32,
        help="nodes_preview 最多节点名数量（DFG 与 CFG 共用）",
    )
    args = parser.parse_args()

    path = os.path.abspath(args.lsir_raw)
    if not os.path.isfile(path):
        print(f"错误: 文件不存在 {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    from utils.ir_builder import build_lsir

    lsir = build_lsir(raw)
    funcs = lsir.get("functions") or []
    if not funcs:
        print("无函数", file=sys.stderr)
        sys.exit(1)

    fn = None
    if args.name:
        needle = args.name.lower()
        for f in funcs:
            if needle in (f.get("name") or "").lower():
                fn = f
                break
        if fn is None:
            print(f"未找到名称包含 {args.name!r} 的函数", file=sys.stderr)
            sys.exit(1)
    else:
        fn = funcs[0]

    name = fn.get("name", "")
    entry = fn.get("entry", "")
    dfg = fn.get("dfg")
    cfg = fn.get("cfg")
    dfg_d = _graph_to_dict(dfg)
    cfg_d = _graph_to_dict(cfg)

    me = max(0, int(args.max_edges))
    max_dfg = me if args.max_dfg_edges is None else max(0, int(args.max_dfg_edges))
    max_cfg = me if args.max_cfg_edges is None else max(0, int(args.max_cfg_edges))
    npv = max(0, int(args.max_nodes_preview))

    dfg_edges = dfg_d.get("edges") or []
    dfg_preview = dfg_edges[:max_dfg]
    dfg_trunc = len(dfg_edges) - len(dfg_preview)

    cfg_nodes = list(cfg_d.get("nodes") or [])
    cfg_edges = cfg_d.get("edges") or []
    cfg_preview = cfg_edges[:max_cfg]
    cfg_trunc = len(cfg_edges) - len(cfg_preview)

    out = {
        "name": name,
        "entry": entry,
        "dfg": {
            "num_nodes": len(dfg_d.get("nodes") or []),
            "num_edges": len(dfg_edges),
            "nodes_preview": list(dfg_d.get("nodes") or [])[:npv],
            "edges_preview": dfg_preview,
            "edges_truncated": max(0, dfg_trunc),
        },
        "cfg": {
            "num_nodes": len(cfg_nodes),
            "num_edges": len(cfg_edges),
            "nodes_preview": cfg_nodes[:npv],
            "edges_preview": cfg_preview,
            "edges_truncated": max(0, cfg_trunc),
        },
        "assert_dfg_key_present": "dfg" in fn,
        "assert_cfg_key_present": "cfg" in fn,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
