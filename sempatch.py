#!/usr/bin/env python3
"""
SemPatch 唯一推荐产品入口。

- match：二进制/查询特征 vs 漏洞库（TwoStage + CVE 报告，默认全函数）。
- compare：legacy DAG 路径（实验/对比用）。
- unpack / extract：辅助或弃用能力。
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def run_unpack(firmware_path: str, output_dir: str) -> dict:
    """
    固件解包模式：构建单节点 DAG（unpack），执行并返回 ctx。
    """
    from dag import JobDAG, run_dag
    from dag.builders import build_unpack_node

    firmware_path = os.path.abspath(firmware_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    dag = JobDAG()
    build_unpack_node(
        dag, "unpack_fw", firmware_path, output_dir=output_dir, deps=[]
    )

    ctx = {}
    run_dag(dag, ctx)
    # 回退：若 ctx 未更新则从节点 output 合并（多线程可见性）
    node = dag.nodes.get("unpack_fw")
    if node and node.output and not ctx.get("unpack_dir"):
        ctx["unpack_dir"] = node.output.get("unpack_dir")
        ctx["unpack_binaries"] = node.output.get("unpack_binaries", [])
    return ctx


def _build_compare_dag(
    dag, binary_path: str, ghidra_out: str, db_path: str, strategy: str, force: bool = False
) -> None:
    """按策略构建 compare 模式 DAG。"""
    from dag.builders import (
        build_acfg_extract_node,
        build_cfg_match_node,
        build_diff_bipartite_node,
        build_diff_faiss_node,
        build_diff_fuzzy_node,
        build_diff_node,
        build_embed_node,
        build_feature_extract_node,
        build_fuzzy_hash_node,
        build_ghidra_node,
        build_load_db_node,
        build_lsir_build_node,
    )

    build_ghidra_node(
        dag, "ghidra_fw", binary_path, ghidra_out,
        deps=[], force=force
    )
    build_lsir_build_node(dag, "lsir_fw", deps=["ghidra_fw"])

    if strategy == "traditional_fuzzy":
        build_fuzzy_hash_node(dag, "fuzzy_hash", deps=["lsir_fw"])
        build_load_db_node(dag, "load_db", db_path=db_path, deps=[], db_format="fuzzy_hashes")
        build_diff_fuzzy_node(
            dag, "diff",
            deps=["fuzzy_hash", "load_db"],
            fuzzy_hashes_key="fuzzy_hashes",
            db_fuzzy_hashes_key="db_fuzzy_hashes",
        )
    elif strategy == "traditional_cfg":
        build_load_db_node(dag, "load_db", db_path=db_path, deps=[], db_format="lsir")
        build_cfg_match_node(
            dag, "diff",
            deps=["lsir_fw", "load_db"],
            lsir_key="lsir",
            db_lsir_key="db_lsir",
        )
    elif strategy == "graph_embed":
        build_acfg_extract_node(dag, "acfg_fw", deps=["lsir_fw"])
        build_embed_node(dag, "embed_fw", deps=["acfg_fw"], input_key="acfg_features", output_key="embeddings")
        build_load_db_node(dag, "load_db", db_path=db_path, deps=[])
        build_diff_faiss_node(
            dag, "diff",
            deps=["embed_fw", "load_db"],
            firmware_embeddings_key="embeddings",
            db_embeddings_key="db_embeddings",
        )
    elif strategy == "fusion":
        build_feature_extract_node(dag, "feat_fw", deps=["lsir_fw"])
        build_embed_node(dag, "embed_fw", deps=["feat_fw"])
        build_load_db_node(dag, "load_db", db_path=db_path, deps=[])
        build_diff_bipartite_node(dag, "diff", deps=["embed_fw", "load_db"])
    else:
        strategy = "semantic_embed"
        build_feature_extract_node(dag, "feat_fw", deps=["lsir_fw"])
        build_embed_node(dag, "embed_fw", deps=["feat_fw"])
        build_load_db_node(dag, "load_db", db_path=db_path, deps=[])
        build_diff_bipartite_node(
            dag, "diff",
            deps=["embed_fw", "load_db"],
            firmware_embeddings_key="embeddings",
            db_embeddings_key="db_embeddings",
        )


def run_firmware_vs_db(
    binary_path: str, db_path: str, output_dir: str, **kwargs
) -> dict:
    """
    固件 vs 漏洞库 模式：构建 DAG、执行、返回 ctx。
    按 pipeline_strategy 选择流水线。
    """
    import json

    from dag import JobDAG, run_dag
    from dag.export import export_dot, export_html, export_mermaid

    binary_path = os.path.abspath(binary_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ghidra_out = os.path.join(output_dir, "ghidra_out")
    os.makedirs(ghidra_out, exist_ok=True)

    try:
        from config import PIPELINE_STRATEGY
        strategy = kwargs.get("strategy") or PIPELINE_STRATEGY
    except ImportError:
        strategy = kwargs.get("strategy", "semantic_embed")

    dag = JobDAG()
    _build_compare_dag(
        dag, binary_path, ghidra_out, db_path,
        strategy=strategy,
        force=kwargs.get("force", False),
    )

    ctx = {}
    run_dag(dag, ctx)

    diff_result = ctx.get("diff_result", {})
    result_path = os.path.join(output_dir, "diff_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(diff_result, f, indent=2, ensure_ascii=False)

    export_dag = kwargs.get("export_dag")
    if export_dag:
        if export_dag == "mermaid":
            export_mermaid(dag, os.path.join(output_dir, "dag.mmd"))
        elif export_dag == "dot":
            export_dot(dag, os.path.join(output_dir, "dag.dot"))
        elif export_dag == "html":
            export_html(dag, os.path.join(output_dir, "dag.html"))

    return ctx


def main():
    parser = argparse.ArgumentParser(
        description="SemPatch：固件漏洞分析（产品路径请使用 match 子命令）"
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # match: TwoStage CVE 匹配（推荐）
    match_parser = subparsers.add_parser(
        "match",
        help="查询二进制/特征 vs 漏洞库（TwoStage，默认处理全部函数）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    from cli.cve_match import add_match_arguments, namespace_to_options, run_cve_match_pipeline

    add_match_arguments(match_parser, require_source=True)

    # compare: legacy DAG
    compare_parser = subparsers.add_parser(
        "compare",
        help="[legacy] DAG：固件 vs 漏洞库嵌入/差分（非 TwoStage 产品路径）",
    )
    compare_parser.add_argument("firmware", help="固件二进制路径")
    compare_parser.add_argument("db_path", help="漏洞库路径")
    compare_parser.add_argument("-o", "--output", default="output", help="输出目录")
    compare_parser.add_argument("--force", action="store_true", help="强制重新分析")
    compare_parser.add_argument(
        "--unpack-first",
        action="store_true",
        help="输入为固件镜像时，先运行 unpack 节点解包再分析",
    )
    compare_parser.add_argument(
        "--export-dag", choices=["mermaid", "dot", "html"], help="导出 DAG 可视化"
    )
    compare_parser.add_argument(
        "--strategy",
        choices=["traditional_fuzzy", "traditional_cfg", "graph_embed", "semantic_embed", "fusion"],
        help="流水线策略（覆盖配置）",
    )

    # unpack: binwalk 解包固件
    unpack_parser = subparsers.add_parser(
        "unpack", help="使用 binwalk 解包固件镜像（DAG 节点）"
    )
    unpack_parser.add_argument("firmware", help="固件镜像路径")
    unpack_parser.add_argument(
        "-o", "--output", default="output/unpacked", help="解包输出目录"
    )

    # legacy: Ghidra 仅导出 lsir_raw
    legacy_parser = subparsers.add_parser(
        "extract",
        help="[legacy] Ghidra 提取 lsir_raw.json（CVE 匹配请用 match）",
    )
    legacy_parser.add_argument("binary", help="二进制文件路径")
    legacy_parser.add_argument("-o", "--output", default="output/v1", help="输出目录")
    legacy_parser.add_argument("--force", action="store_true", help="强制重新提取")
    legacy_parser.add_argument("--timeout", type=int, default=None, help="超时秒数")

    args = parser.parse_args()

    if args.command == "match":
        try:
            sys.exit(run_cve_match_pipeline(namespace_to_options(args)))
        except Exception as e:
            print("错误:", e)
            sys.exit(1)

    elif args.command == "compare":
        print(
            "提示: compare 为 legacy DAG；产品 CVE 匹配请使用: python sempatch.py match ...",
            file=sys.stderr,
        )
        if not os.path.isfile(args.firmware):
            print("错误: 固件不存在:", args.firmware)
            sys.exit(1)
        try:
            from utils.ghidra_runner import GhidraEnvironmentError, require_ghidra_environment

            require_ghidra_environment()
        except GhidraEnvironmentError as e:
            print("错误: compare 需要可用的 Ghidra 环境:", e, file=sys.stderr)
            sys.exit(1)
        os.makedirs(args.output, exist_ok=True)
        binary_path = args.firmware
        try:
            if getattr(args, "unpack_first", False):
                ctx_unpack = run_unpack(args.firmware, args.output)
                bins = ctx_unpack.get("unpack_binaries", [])
                if not bins:
                    print(
                        "错误: --unpack-first 解包后未发现 ELF 二进制，"
                        "请检查固件或直接指定二进制路径"
                    )
                    sys.exit(1)
                binary_path = bins[0]
                print("使用解包后的二进制:", binary_path)
            ctx = run_firmware_vs_db(
                binary_path,
                args.db_path,
                args.output,
                force=args.force,
                export_dag=args.export_dag,
                strategy=getattr(args, "strategy", None),
            )
            print("完成: 结果已写入", args.output)
        except Exception as e:
            print("错误:", e)
            sys.exit(1)

    elif args.command == "unpack":
        if not os.path.isfile(args.firmware):
            print("错误: 固件不存在:", args.firmware)
            sys.exit(1)
        try:
            ctx = run_unpack(args.firmware, args.output)
            unpack_dir = ctx.get("unpack_dir", "")
            bins = ctx.get("unpack_binaries", [])
            print("解包完成:", unpack_dir)
            if bins:
                print("发现的 ELF 二进制数量:", len(bins))

        except Exception as e:
            print("错误:", e)
            sys.exit(1)

    elif args.command == "extract":
        print(
            "提示: extract 为 legacy；CVE 匹配请使用: python sempatch.py match --query-binary ...",
            file=sys.stderr,
        )
        if not os.path.isfile(args.binary):
            print("错误: 二进制不存在:", args.binary)
            sys.exit(1)
        try:
            from utils.ghidra_runner import GhidraEnvironmentError, require_ghidra_environment

            require_ghidra_environment()
        except GhidraEnvironmentError as e:
            print("错误: extract 需要可用的 Ghidra 环境:", e, file=sys.stderr)
            sys.exit(1)
        from utils.ghidra_runner import run_ghidra_analysis
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        run_ghidra_analysis(
            binary_path=args.binary,
            output_dir=output_dir,
            force=args.force,
            timeout=args.timeout,
        )
        print("完成:", os.path.join(output_dir, "lsir_raw.json"))

    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
