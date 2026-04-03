#!/usr/bin/env python3
"""
按路径子串过滤 binkit 索引 JSON（用于分阶段训练：例如仅 x86_64 子集预训练）。

示例:
  python scripts/filter_binkit_index_subset.py -i full.json -o stage1.json --path-contains x86_64
  python scripts/filter_binkit_index_subset.py -i full.json -o stage1.json --path-contains aarch64 --path-contains arm
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> None:
    p = argparse.ArgumentParser(description="按 binary 路径子串过滤索引")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument(
        "--path-contains",
        action="append",
        default=[],
        help="binary 相对路径需包含的子串（可重复；任一匹配即保留该项）",
    )
    args = p.parse_args()
    needles = [x for x in (args.path_contains or []) if x]
    if not needles:
        print("错误: 至少指定一个 --path-contains", file=sys.stderr)
        sys.exit(1)

    with open(os.path.abspath(args.input), encoding="utf-8") as f:
        raw = json.load(f)
    items = raw if isinstance(raw, list) else ([raw] if isinstance(raw, dict) else [])

    out: list = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rel = (item.get("binary") or "").replace("\\", "/")
        if any(n in rel for n in needles):
            out.append(item)

    op = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(op) or ".", exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    nfn = sum(len(x.get("functions") or []) for x in out)
    print(f"已写入 {op}: {len(out)} 个二进制, {nfn} 个函数")


if __name__ == "__main__":
    main()
