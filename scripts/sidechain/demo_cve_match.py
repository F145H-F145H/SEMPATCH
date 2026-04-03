#!/usr/bin/env python3
"""
CVE 导向二进制匹配 Demo：固定走 TwoStagePipeline（SAFE 粗筛 + 多模态精排）。

实现已迁至 src/cli/two_stage_demo.py；本脚本为侧链兼容入口。

详见 docs/DEMO.md。
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from cli.cve_match import MATCH_COMMAND_EPILOG  # noqa: E402
from cli.two_stage_demo import (  # noqa: E402
    build_query_features_from_binary,
    git_short_hash,
    run_demo,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CVE 匹配 Demo：TwoStagePipeline + 报告（matches.json / report.md）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--query-binary",
        help="查询 ELF 路径（将调用 Ghidra / binary_cache 生成 query_features）",
    )
    src.add_argument(
        "--query-features",
        help="预计算查询特征 JSON，跳过 Ghidra",
    )
    parser.add_argument("--library-emb", required=True, help="库 SAFE 嵌入 JSON")
    parser.add_argument("--library-features", required=True, help="库 multimodal 特征 JSON")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument(
        "--model-path",
        default=None,
        help="精排模型权重，默认 output/best_model.pth",
    )
    parser.add_argument(
        "--safe-model-path",
        default=None,
        help="SAFE 粗筛权重（须与库嵌入一致）",
    )
    parser.add_argument(
        "--coarse-k",
        type=int,
        default=argparse.SUPPRESS,
        help="粗筛 K（默认 100；出现在命令行时视为显式指定）",
    )
    parser.add_argument("--top-k", type=int, default=10, help="match-filter=top_k 时截断 Top-K")
    parser.add_argument(
        "--match-filter",
        choices=["top_k", "unique", "all_above"],
        default="top_k",
        help="结果策略（与 sempatch match 一致）",
    )
    parser.add_argument("--min-similarity", type=float, default=0.95, metavar="S")
    parser.add_argument("--tie-margin", type=float, default=1e-5, metavar="EPS")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="最多处理查询函数数量（function_id 排序后截断）；默认不限制",
    )
    parser.add_argument(
        "--query-entry",
        default=None,
        metavar="ADDR",
        help="仅处理入口地址等于该十六进制的查询 function_id（如 0x401176）",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="强制 CPU，不优先 CUDA",
    )
    parser.add_argument(
        "--use-dfg-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="精排是否使用 DFG 分支：默认按权重/meta 推断；--no-use-dfg-model 强制旧版无 DFG 结构",
    )
    parser.epilog = MATCH_COMMAND_EPILOG
    args = parser.parse_args()

    lib_emb = os.path.abspath(args.library_emb)
    lib_feat = os.path.abspath(args.library_features)
    out_dir = os.path.abspath(args.output_dir)

    for label, p in (
        ("--library-emb", lib_emb),
        ("--library-features", lib_feat),
    ):
        if not os.path.isfile(p):
            print(f"错误: {label} 文件不存在: {p}", file=sys.stderr)
            sys.exit(1)

    query_binary_arg = None
    if args.query_binary:
        query_binary_arg = args.query_binary
        try:
            qpath = build_query_features_from_binary(
                args.query_binary, PROJECT_ROOT, out_dir
            )
        except Exception as e:
            print(f"错误: 查询特征提取失败: {e}", file=sys.stderr)
            sys.exit(1)
        query_mode = "binary"
    else:
        qpath = os.path.abspath(args.query_features)
        if not os.path.isfile(qpath):
            print(f"错误: --query-features 文件不存在: {qpath}", file=sys.stderr)
            sys.exit(1)
        query_mode = "file"

    max_q = args.max_queries
    if max_q is not None and max_q <= 0:
        max_q = None
    q_entry = (args.query_entry or "").strip() or None

    print("=== SemPatch CVE Demo 复现信息 ===", flush=True)
    print(f"git_rev={git_short_hash(PROJECT_ROOT)}", flush=True)
    print(f"project_root={PROJECT_ROOT}", flush=True)
    print(f"query_mode={query_mode}", flush=True)
    if query_binary_arg:
        print(f"query_binary={query_binary_arg}", flush=True)
    print(f"query_features={qpath}", flush=True)
    print(f"library_emb={lib_emb}", flush=True)
    print(f"library_features={lib_feat}", flush=True)
    print(
        f"rerank_model={args.model_path or os.path.join(PROJECT_ROOT, 'output', 'best_model.pth')}",
        flush=True,
    )
    print(f"safe_model={args.safe_model_path}", flush=True)
    coarse_k = int(getattr(args, "coarse_k", 100))
    print(
        f"coarse_k={coarse_k} top_k={args.top_k} match_filter={args.match_filter} "
        f"use_dfg_model={args.use_dfg_model}",
        flush=True,
    )
    print(f"output_dir={out_dir}", flush=True)

    try:
        run_demo(
            query_features_path=qpath,
            library_emb=lib_emb,
            library_features=lib_feat,
            output_dir=out_dir,
            rerank_model_path=args.model_path,
            safe_model_path=args.safe_model_path,
            coarse_k=coarse_k,
            top_k=args.top_k,
            max_queries=max_q,
            prefer_cuda=not args.cpu,
            query_mode=query_mode,
            query_binary=query_binary_arg,
            rerank_use_dfg=args.use_dfg_model,
            query_entry=q_entry,
            match_filter=args.match_filter,
            min_similarity=float(args.min_similarity),
            tie_margin=float(args.tie_margin),
        )
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"已写入 {os.path.join(out_dir, 'matches.json')}", flush=True)
    print(f"已写入 {os.path.join(out_dir, 'report.md')}", flush=True)


if __name__ == "__main__":
    main()
