#!/usr/bin/env python3
"""
基线对比：运行 SemPatch 与 SAFE / jtrans_style 等基线，统一输出 Recall@K、MRR。

用法:
  python scripts/run_baseline_comparison.py --firmware-emb q.json --db-emb d.json
  python scripts/run_baseline_comparison.py --index-file data/binkit_functions.json --model safe -k 1 5 10
  python scripts/run_baseline_comparison.py --index-file data/binkit_functions.json --model jtrans_style -k 1 5

详见 docs/BASELINE_AND_EVAL.md。
"""
import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_OUT_DIR = os.path.join(PROJECT_ROOT, "output", "baseline_comparison")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="基线对比：SemPatch vs SAFE / jtrans_style，输出 Recall@K、MRR"
    )
    parser.add_argument("--firmware-emb", default=None, help="查询嵌入路径")
    parser.add_argument("--db-emb", default=None, help="数据库嵌入路径")
    parser.add_argument(
        "--index-file",
        default=None,
        help="索引路径（与 binkit_functions.json 格式一致）；指定时将构建嵌入",
    )
    parser.add_argument(
        "--model",
        choices=["sempatch", "safe", "jtrans_style"],
        default="sempatch",
        help="嵌入模型（仅当从 --index-file 构建时生效）",
    )
    parser.add_argument("-k", nargs="+", type=int, default=[1, 5, 10], help="K 值")
    parser.add_argument("--model-path", default=None, help="SemPatch 训练模型路径")
    args = parser.parse_args()

    if args.index_file:
        os.makedirs(_OUT_DIR, exist_ok=True)
        emb_path = os.path.join(_OUT_DIR, f"embeddings_{args.model}.json")
        build_cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "build_embeddings_db.py"),
            "--index-file", args.index_file,
            "-o", emb_path,
            "--model", args.model,
        ]
        if args.model_path:
            build_cmd.extend(["--model-path", args.model_path])
        subprocess.run(build_cmd, check=True, cwd=PROJECT_ROOT)
        args.firmware_emb = emb_path
        args.db_emb = emb_path

    if not args.firmware_emb or not args.db_emb:
        print("错误: 必须指定 --firmware-emb 与 --db-emb，或 --index-file", file=sys.stderr)
        sys.exit(1)

    eval_cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "scripts", "eval_bcsd.py"),
        "--firmware-emb", args.firmware_emb,
        "--db-emb", args.db_emb,
        "-k", *map(str, args.k),
    ]
    subprocess.run(eval_cmd, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
