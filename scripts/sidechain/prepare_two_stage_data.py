#!/usr/bin/env python3
"""
两阶段数据划分：将 BinKit 索引拆分为函数库与查询集。

按二进制随机划分（80% 库 / 20% 查询），仅保留「正样本充足」的查询
（在库中至少有 1 个同名函数）。若有效查询数 < min-queries，逐步提高查询侧比例。

输出：
  - data/two_stage/library_index.json
  - data/two_stage/query_index.json
  - data/two_stage/ground_truth.json

用法:
  python scripts/prepare_two_stage_data.py
  python scripts/prepare_two_stage_data.py --seed 42 --min-queries 1000
"""
import argparse
import json
import os
import random
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _log(msg: str) -> None:
    """立即输出到 stderr，避免管道/重定向时进度滞后。"""
    print(msg, file=sys.stderr, flush=True)


def _file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return -1.0


def _norm_entry(entry: str) -> str:
    """Ensure entry is 0x-prefixed hex."""
    if not entry:
        return "0x0"
    s = str(entry).strip().lower()
    if s.startswith("0x"):
        return s
    while len(s) > 1 and s[0] == "0":
        s = s[1:]
    return "0x" + s


def _function_id(binary_path: str, entry: str) -> str:
    """Format: binary_path|entry_address (relative path)."""
    return f"{binary_path}|{_norm_entry(entry)}"


def _build_library_function_names(
    library_items: list,
    *,
    progress_every: int = 0,
    label: str = "扫描库侧函数名",
) -> set:
    """库侧出现过的函数名集合（试探划分比例时用，避免构建 name→站点大表）。"""
    names: set = set()
    n_bin = len(library_items)
    t0 = time.perf_counter()
    for bi, item in enumerate(library_items, start=1):
        for fn in item.get("functions", []) or []:
            name = fn.get("name", "")
            if name:
                names.add(name)
        if progress_every > 0 and bi % progress_every == 0:
            elapsed = time.perf_counter() - t0
            _log(f"  [{label}] 二进制 {bi}/{n_bin} ({elapsed:.1f}s)")
    return names


def _build_name_to_positive_function_ids(
    library_items: list,
    *,
    progress_every: int = 0,
    label: str = "构建 name→正样本 function_id",
) -> dict:
    """
    每个函数名在库侧的全部 function_id（只构建一次）。
    值为 tuple，便于多查询同名时共享引用且避免被下游误改。
    """
    name_to_ids: dict = {}
    n_bin = len(library_items)
    t0 = time.perf_counter()
    for bi, item in enumerate(library_items, start=1):
        binary = item.get("binary", "")
        for fn in item.get("functions", []) or []:
            name = fn.get("name", "")
            entry = fn.get("entry", "")
            if not name or not entry:
                continue
            fid = _function_id(binary, entry)
            name_to_ids.setdefault(name, []).append(fid)
        if progress_every > 0 and bi % progress_every == 0:
            elapsed = time.perf_counter() - t0
            _log(f"  [{label}] 二进制 {bi}/{n_bin} ({elapsed:.1f}s)")
    # 冻结为 tuple，供 ground_truth 多查询复用同一份正样本列表
    return {k: tuple(v) for k, v in name_to_ids.items()}


def _count_positive_sufficient_queries(
    query_items: list, library_names: set
) -> int:
    """统计查询侧中「在库侧至少有一个同名函数」的函数个数。"""
    count = 0
    for item in query_items:
        for fn in item.get("functions", []) or []:
            name = fn.get("name", "")
            if name and name in library_names:
                count += 1
    return count


def _build_ground_truth(
    query_items: list,
    name_to_positives: dict,
    *,
    progress_every: int = 0,
    label: str = "ground_truth 查询侧",
) -> dict:
    """
    ground_truth: query_function_id -> [positive_function_id, ...]
    同名查询共享同一份正样本 tuple（语义与原先逐条 list 构造一致，json 写出仍为数组）。
    """
    gt = {}
    n_bin = len(query_items)
    t0 = time.perf_counter()
    for bi, item in enumerate(query_items, start=1):
        binary = item.get("binary", "")
        for fn in item.get("functions", []) or []:
            name = fn.get("name", "")
            entry = fn.get("entry", "")
            if not name or not entry:
                continue
            positives = name_to_positives.get(name)
            if not positives:
                continue
            qid = _function_id(binary, entry)
            gt[qid] = positives
        if progress_every > 0 and bi % progress_every == 0:
            elapsed = time.perf_counter() - t0
            _log(f"  [{label}] 二进制 {bi}/{n_bin}，已写入 {len(gt)} 条查询 ({elapsed:.1f}s)")
    return gt


def main() -> None:
    parser = argparse.ArgumentParser(description="两阶段数据划分：库/查询索引与 ground_truth")
    parser.add_argument(
        "--index-file",
        default=os.path.join(PROJECT_ROOT, "data", "binkit_functions.json"),
        help="BinKit 索引路径",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "data", "two_stage"),
        help="输出目录",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--min-queries",
        type=int,
        default=1000,
        help="目标最小有效查询数（正样本充足）",
    )
    parser.add_argument(
        "--progress-every-binaries",
        type=int,
        default=5,
        help="构建 name→sites 时每处理多少个二进制打印一行进度（0=关闭）",
    )
    args = parser.parse_args()

    index_path = os.path.abspath(args.index_file)
    if not os.path.isfile(index_path):
        _log(f"错误: 索引文件不存在: {index_path}")
        sys.exit(1)
    if os.path.basename(index_path).startswith("-"):
        _log(
            "错误: --index-file 的值看起来像被 shell 吃掉了。"
            " 检查续行：反斜杠后不能有空格（应写成 \\ 然后立即换行）。"
        )
        sys.exit(1)

    sz_mb = _file_size_mb(index_path)
    _log(f"[1/5] 读取索引: {index_path}（约 {sz_mb:.1f} MiB）…")
    t_load = time.perf_counter()
    with open(index_path, encoding="utf-8") as f:
        index_items = json.load(f)
    _log(f"      json.load 完成，用时 {time.perf_counter() - t_load:.1f}s")

    if not isinstance(index_items, list):
        index_items = [index_items] if isinstance(index_items, dict) else []

    n_items = len(index_items)
    n_funcs = sum(len(x.get("functions", []) or []) for x in index_items)
    _log(f"[2/5] 索引条目: {n_items} 个二进制, 共约 {n_funcs} 个函数")

    # 原地打乱，避免再复制一整份列表（降低峰值内存）
    rng = random.Random(args.seed)
    _log("[3/5] 按二进制随机打乱（原地 shuffle）…")
    rng.shuffle(index_items)
    shuffled = index_items

    n_total = len(shuffled)
    query_ratios = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    library_items = []
    query_items = []
    used_ratio = 0.20

    pe = max(0, int(args.progress_every_binaries))
    _log("[4/5] 尝试查询侧比例 " + "/".join(f"{int(r * 100)}%" for r in query_ratios) + " …")
    for ratio in query_ratios:
        n_lib = int(n_total * (1 - ratio) + 0.5)
        n_query = n_total - n_lib
        if n_lib <= 0 or n_query <= 0:
            continue
        lib = shuffled[:n_lib]
        qry = shuffled[n_lib:]
        _log(f"  比例 {ratio:.0%}: 库 {n_lib} 个二进制 / 查询 {n_query} 个二进制，统计有效查询…")
        lib_names = _build_library_function_names(
            lib,
            progress_every=pe,
            label=f"库名集合 ratio={ratio:.2f}",
        )
        n_sufficient = _count_positive_sufficient_queries(qry, lib_names)
        del lib_names
        _log(f"       → 正样本充足查询数: {n_sufficient}")
        if n_sufficient >= args.min_queries:
            library_items = lib
            query_items = qry
            used_ratio = ratio
            break
        # Use the best we can get if we've reached 50%
        if ratio == 0.50:
            library_items = lib
            query_items = qry
            used_ratio = ratio
            if n_sufficient < args.min_queries:
                print(
                    f"提示: 查询侧 50% 时有效查询数仍为 {n_sufficient}，"
                    f"不足目标 {args.min_queries}，接受实际数量。",
                    file=sys.stderr,
                )
            break

    _log("[5/5] 生成 ground_truth（按函数名共享正样本列表）…")
    name_to_positives = _build_name_to_positive_function_ids(
        library_items,
        progress_every=pe,
        label="最终库侧 name→正样本",
    )
    ground_truth = _build_ground_truth(
        query_items,
        name_to_positives,
        progress_every=pe,
        label="查询侧映射",
    )
    del name_to_positives
    n_lib_bin = len(library_items)
    n_query_bin = len(query_items)
    n_lib_func = sum(len(x.get("functions", [])) for x in library_items)
    n_query_func = sum(len(x.get("functions", [])) for x in query_items)

    print(f"划分比例: 库 {1-used_ratio:.0%} / 查询 {used_ratio:.0%}")
    print(f"库: {n_lib_bin} 个二进制, {n_lib_func} 个函数")
    print(f"查询: {n_query_bin} 个二进制, {n_query_func} 个函数（原始）")
    print(f"正样本充足查询数: {len(ground_truth)}")

    os.makedirs(args.output_dir, exist_ok=True)

    lib_path = os.path.join(args.output_dir, "library_index.json")
    query_path = os.path.join(args.output_dir, "query_index.json")
    gt_path = os.path.join(args.output_dir, "ground_truth.json")

    _log("写入 JSON（库索引 / 查询索引 / ground_truth）…")
    with open(lib_path, "w", encoding="utf-8") as f:
        json.dump(library_items, f, indent=2, ensure_ascii=False)
    with open(query_path, "w", encoding="utf-8") as f:
        json.dump(query_items, f, indent=2, ensure_ascii=False)
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"已写入: {lib_path}, {query_path}, {gt_path}")


if __name__ == "__main__":
    main()
