#!/usr/bin/env python3
"""
从两阶段 ground_truth.json 生成高置信子集：同源 project_id、CFG 节点比例、可选排除 libc 符号。

依赖:
  - ground_truth.json
  - library_features.json / query_features.json（function_id -> multimodal）
  - query_index.json（从索引解析 function_id -> 符号名，用于 libc 过滤）

示例:
  python scripts/filter_ground_truth_high_conf.py \\
    --data-dir data/two_stage \\
    --output data/two_stage/ground_truth_high_conf.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def _graph_nodes(mm: dict) -> int:
    g = mm.get("graph") or {}
    try:
        return int(g.get("num_nodes") or 0)
    except (TypeError, ValueError):
        return 0


def _load_features_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _norm_entry(entry: str) -> str:
    s = (entry or "").strip().lower()
    if s.startswith("0x"):
        return s
    return f"0x{s}" if s else "0x0"


def _fid_to_name_from_index(index_path: str, _project_root: str) -> dict:
    out: dict = {}
    with open(index_path, encoding="utf-8") as f:
        raw = json.load(f)
    items = raw if isinstance(raw, list) else ([raw] if isinstance(raw, dict) else [])
    for item in items:
        if not isinstance(item, dict):
            continue
        binary_rel = (item.get("binary") or "").replace("\\", "/")
        for fn in item.get("functions") or []:
            entry = (fn.get("entry") or "").strip()
            name = (fn.get("name") or "").strip()
            if not entry or not name:
                continue
            fid = f"{binary_rel}|{_norm_entry(entry)}"
            out[fid] = name
    return out


def main() -> None:
    from utils.binkit_provenance import parse_binary_provenance
    from utils.training_function_filter import TrainingSymbolFilter, strip_linker_suffix

    p = argparse.ArgumentParser(description="过滤 ground_truth 为高置信子集")
    p.add_argument("--data-dir", required=True, help="含 ground_truth / *features / query_index")
    p.add_argument(
        "--output",
        default=None,
        help="默认 <data-dir>/ground_truth_high_conf.json",
    )
    p.add_argument("--project-root", default=PROJECT_ROOT)
    p.add_argument(
        "--max-cfg-node-ratio",
        type=float,
        default=5.0,
        help="查询与正例 graph.num_nodes 比例上限（≤0 表示不启用）",
    )
    p.add_argument(
        "--exclude-libc-common",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="按符号名排除 libc_common（默认开）",
    )
    args = p.parse_args()

    ddir = os.path.abspath(args.data_dir)
    gt_path = os.path.join(ddir, "ground_truth.json")
    lib_path = os.path.join(ddir, "library_features.json")
    qfeat_path = os.path.join(ddir, "query_features.json")
    qidx_path = os.path.join(ddir, "query_index.json")
    out_path = os.path.abspath(args.output or os.path.join(ddir, "ground_truth_high_conf.json"))

    for path in (gt_path, lib_path, qfeat_path, qidx_path):
        if not os.path.isfile(path):
            print(f"错误: 缺少文件 {path}", file=sys.stderr)
            sys.exit(1)

    sym_f = TrainingSymbolFilter(
        exclude_runtime=False,
        include_libc_common=bool(args.exclude_libc_common),
    )

    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    if not isinstance(gt, dict):
        print("错误: ground_truth 应为 dict", file=sys.stderr)
        sys.exit(1)

    lib_mm = _load_features_json(lib_path)
    q_mm = _load_features_json(qfeat_path)
    fid_names = _fid_to_name_from_index(qidx_path, args.project_root)

    ratio_cap = float(args.max_cfg_node_ratio)
    new_gt: dict = {}
    dropped = 0

    for qid, positives in gt.items():
        if not isinstance(positives, list) or qid not in q_mm:
            dropped += 1
            continue
        q_bin = qid.split("|", 1)[0]
        q_pid, _ = parse_binary_provenance(q_bin)
        qn = fid_names.get(qid, "")
        if qn and sym_f.is_excluded(strip_linker_suffix(qn)):
            dropped += 1
            continue
        q_nodes = _graph_nodes(q_mm[qid])
        kept_pos: list = []
        for pid in positives:
            if not isinstance(pid, str):
                continue
            p_bin = pid.split("|", 1)[0]
            p_pid, _ = parse_binary_provenance(p_bin)
            if p_pid != q_pid:
                continue
            if pid not in lib_mm:
                continue
            pn = _graph_nodes(lib_mm[pid])
            if ratio_cap > 0 and q_nodes > 0 and pn > 0:
                a, b = max(q_nodes, pn), max(min(q_nodes, pn), 1)
                if (a / b) > ratio_cap:
                    continue
            kept_pos.append(pid)
        if kept_pos:
            new_gt[qid] = kept_pos
        else:
            dropped += 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_gt, f, indent=2, ensure_ascii=False)
    print(
        f"已写入 {out_path}: {len(new_gt)} 条查询（丢弃约 {dropped} 条无合格正例或过滤掉）",
        flush=True,
    )


if __name__ == "__main__":
    main()
