#!/usr/bin/env python3
"""
跨编译器、跨优化、跨架构的挑战场景评估。

使用 --query-index 与 --db-index 指定两组索引（如 O0 vs O3、gcc vs clang），
分别构建嵌入后运行 Recall@K、MRR 评估。若 BinKit 子集仅含单一架构，将输出说明。

用法:
  python scripts/eval_challenge.py --query-index data/query_index.json --db-index data/db_index.json
  python scripts/eval_challenge.py --query-emb q.json --db-emb d.json -k 1 5 10
  python scripts/eval_challenge.py --query-index data/binkit_functions.json --db-index data/binkit_functions.json
"""
import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_TEMP_DIR = os.path.join(PROJECT_ROOT, "output", "challenge_temp")


def _run_build_embeddings(
    index_path: str,
    output_path: str,
    model_path: str | None,
    temp_dir: str,
    model: str = "sempatch",
) -> bool:
    """调用 build_embeddings_db 从索引构建嵌入。"""
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "scripts", "build_embeddings_db.py"),
        "--index-file",
        index_path,
        "-o",
        output_path,
        "--temp-dir",
        temp_dir,
        "--model",
        model,
    ]
    if model_path:
        cmd.extend(["--model-path", model_path])
    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: 构建嵌入失败 (exit {e.returncode})", file=sys.stderr)
        return False


def _run_eval_bcsd(
    firmware_emb: str,
    db_emb: str,
    k_list: list[int],
    output_path: str | None,
) -> bool:
    """调用 eval_bcsd 计算 Recall@K、MRR。"""
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "scripts", "eval_bcsd.py"),
        "--firmware-emb",
        firmware_emb,
        "--db-emb",
        db_emb,
        "-k",
        *map(str, k_list),
    ]
    if output_path:
        cmd.extend(["--output", output_path])
    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: 评估失败 (exit {e.returncode})", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="跨编译器/优化/架构挑战场景评估（O0 vs O3、gcc vs clang 等）"
    )
    parser.add_argument(
        "--query-index",
        default=None,
        help="查询端索引路径（与 binkit_functions.json 格式一致）",
    )
    parser.add_argument(
        "--db-index",
        default=None,
        help="数据库端索引路径",
    )
    parser.add_argument(
        "--query-emb",
        "--firmware-emb",
        dest="query_emb",
        default=None,
        help="预构建的查询嵌入路径（若指定则跳过 query 构建）",
    )
    parser.add_argument(
        "--db-emb",
        default=None,
        help="预构建的数据库嵌入路径（若指定则跳过 db 构建）",
    )
    parser.add_argument(
        "-k",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="K 值列表",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="模型权重路径：sempatch 为 MultiModalFusion；safe/jtrans_style 为对应 .pt",
    )
    parser.add_argument(
        "--model",
        choices=["sempatch", "safe", "jtrans_style"],
        default="sempatch",
        help="构建嵌入时使用的模型（与 build_embeddings_db --model 一致）",
    )
    parser.add_argument(
        "--temp-dir",
        default=_DEFAULT_TEMP_DIR,
        help="Ghidra 临时目录",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="评估结果输出路径",
    )
    args = parser.parse_args()

    query_emb = args.query_emb
    db_emb = args.db_emb

    if not query_emb and not args.query_index:
        print("错误: 必须指定 --query-emb 或 --query-index", file=sys.stderr)
        sys.exit(1)
    if not db_emb and not args.db_index:
        print("错误: 必须指定 --db-emb 或 --db-index", file=sys.stderr)
        sys.exit(1)

    temp_dir = os.path.abspath(args.temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    if not query_emb:
        if not os.path.isfile(args.query_index):
            print(f"错误: 索引不存在 {args.query_index}", file=sys.stderr)
            sys.exit(1)
        query_emb = os.path.join(temp_dir, "query_embeddings.json")
        print(f"构建查询嵌入（{args.query_index}）...")
        if not _run_build_embeddings(
            args.query_index,
            query_emb,
            args.model_path,
            temp_dir,
            model=args.model,
        ):
            sys.exit(1)

    if not db_emb:
        if not os.path.isfile(args.db_index):
            print(f"错误: 索引不存在 {args.db_index}", file=sys.stderr)
            sys.exit(1)
        db_emb = os.path.join(temp_dir, "db_embeddings.json")
        print(f"构建数据库嵌入（{args.db_index}）...")
        if not _run_build_embeddings(
            args.db_index,
            db_emb,
            args.model_path,
            temp_dir,
            model=args.model,
        ):
            sys.exit(1)

    if not os.path.isfile(query_emb):
        print(f"错误: 查询嵌入文件不存在 {query_emb}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(db_emb):
        print(f"错误: 数据库嵌入文件不存在 {db_emb}", file=sys.stderr)
        sys.exit(1)

    print("运行评估...", flush=True)
    if not _run_eval_bcsd(query_emb, db_emb, args.k, args.output):
        sys.exit(1)


if __name__ == "__main__":
    main()
