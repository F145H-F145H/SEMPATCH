#!/usr/bin/env python3
"""
从 Ghidra lsir_raw.json 构建漏洞库 (LSIR 格式)，供 SemPatch compare 使用。

用法:
  python scripts/build_vuln_db.py output/vuln_httpd/lsir_raw.json -o data/vulnerability_db/cve-2018-10822.json
  python scripts/build_vuln_db.py lsir_raw.json -o vuln_db.json --filter "ssiCommand"  # 仅保留匹配函数名
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="从 lsir_raw 构建漏洞库")
    parser.add_argument("lsir_raw", help="lsir_raw.json 路径")
    parser.add_argument("-o", "--output", required=True, help="输出 vuln_db JSON")
    parser.add_argument("--filter", default=None, help="函数名过滤（子串匹配）")
    args = parser.parse_args()

    path = os.path.abspath(args.lsir_raw)
    if not os.path.isfile(path):
        print(f"错误: 文件不存在 {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    funcs = raw.get("functions", [])
    if args.filter:
        funcs = [f for f in funcs if args.filter.lower() in (f.get("name") or "").lower()]
        print(f"过滤后保留 {len(funcs)} 个函数")

    from utils.ir_builder import build_lsir

    def _graph_to_dict(g) -> dict:
        """将 nx.DiGraph 转为 JSON 可序列化的 dict。"""
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
            return g
        return {"nodes": [], "edges": []}

    lsir_list = []
    for fn in funcs:
        try:
            lsir = build_lsir({"functions": [fn]}, include_cfg=True, include_dfg=True)
            if lsir.get("functions"):
                f = lsir["functions"][0]
                # 转为可 JSON 序列化
                if "cfg" in f and not isinstance(f.get("cfg"), dict):
                    f["cfg"] = _graph_to_dict(f["cfg"])
                if "dfg" in f and not isinstance(f.get("dfg"), dict):
                    f["dfg"] = _graph_to_dict(f["dfg"])
                lsir_list.append(f)
        except Exception:
            pass

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"functions": lsir_list}, f, indent=2, ensure_ascii=False)

    print(f"已写入 {out_path} ({len(lsir_list)} 个函数)")


if __name__ == "__main__":
    main()
