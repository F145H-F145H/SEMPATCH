#!/usr/bin/env python3
"""
从 data/binkit_subset/ 下的二进制构建函数索引 data/binkit_functions.json。

遍历每个二进制，使用 Ghidra extract_lsir_raw.java 一次性导出全函数 lsir_raw，
从 lsir_raw 推导 {name, entry} 并写入 BINARY_CACHE_DIR，供后续 build_library_features、
PairwiseFunctionDataset 复用，实现「一二进制一 Ghidra」。

缓存策略（Plan B）：每个二进制优先从 binary_cache 直接读取，缓存命中时不创建任何临时目录。
缓存未命中时，在 session temp_dir 下创建编号子目录运行 Ghidra，子目录用毕立即删除。
session temp_dir 在脚本结束时统一删除。

输出格式：[{"binary": "data/binkit_subset/xxx.elf", "functions": [{"name": "...", "entry": "0x1234"}]}]。

用法:
  python scripts/build_binkit_index.py
  python scripts/build_binkit_index.py --input-dir data/binkit_subset --output data/binkit_functions.json
  python scripts/build_binkit_index.py --from-index-file data/vuln_library_binary_index.json -o data/vuln_functions.json
  python scripts/build_binkit_index.py --force   # 强制重新提取，忽略缓存
  python scripts/build_binkit_index.py --workers 4   # 多线程并行
  python scripts/build_binkit_index.py --exclude-runtime-symbols  # 索引阶段即去掉 main/CRT（可选；训练默认在 filter_index_by_pcode_len 排除）
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Callable, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


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


def _lsir_to_funcs(
    lsir_raw: dict,
    name_filter: Optional[Callable[[str], bool]] = None,
) -> list:
    """从 lsir_raw dict 推导 {name, entry} 列表；可选按函数名排除（与 training_function_filter 一致）。"""
    out: list = []
    for f in (lsir_raw or {}).get("functions", []):
        nm = (f.get("name") or "").strip()
        if name_filter is not None and name_filter(nm):
            continue
        out.append(
            {"name": f.get("name", ""), "entry": _norm_entry(f.get("entry", "0x0"))}
        )
    return out


def _output_binary_field(binary_abs: str) -> str:
    """写入索引的 binary 字段：在项目根内用相对路径，否则用绝对路径（与 build_library_binary_index 一致）。"""
    rel = os.path.relpath(os.path.abspath(binary_abs), PROJECT_ROOT)
    if rel.startswith(".."):
        return os.path.abspath(binary_abs)
    return rel


def _collect_binaries_from_index_file(index_path: str) -> list[str]:
    """
    读取 build_library_binary_index 等输出的兼容索引 JSON：数组项含 "binary" 键。
    返回去重后的绝对路径列表（顺序稳定）。
    """
    with open(os.path.abspath(index_path), encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("索引文件应为 JSON 数组")
    seen: set[str] = set()
    binaries: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        b = (item.get("binary") or "").strip()
        if not b:
            continue
        if os.path.isabs(b):
            abs_p = os.path.abspath(b)
        else:
            abs_p = os.path.abspath(os.path.join(PROJECT_ROOT, b))
        if not os.path.isfile(abs_p):
            print(f"警告: 索引中文件不存在，跳过: {b}", file=sys.stderr)
            continue
        if abs_p in seen:
            continue
        seen.add(abs_p)
        binaries.append(abs_p)
    return binaries


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 BinKit 函数索引 binkit_functions.json")
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument(
        "--input-dir",
        default=None,
        help="二进制目录（与 --from-index-file 二选一；均未指定时默认 data/binkit_subset）",
    )
    src.add_argument(
        "--from-index-file",
        default=None,
        metavar="PATH",
        help="索引 JSON（如 data/vuln_library_binary_index.json），每项含 binary 字段；输出全函数列表供 build_library_features 使用",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default=os.path.join(PROJECT_ROOT, "data", "binkit_functions.json"),
        help="输出索引 JSON 路径",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新提取，忽略 BINARY_CACHE_DIR 缓存",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Ghidra 临时目录（调试用，默认使用系统临时目录；脚本结束后均会删除）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Ghidra 并行数（默认使用 PARALLEL_WORKERS）",
    )
    parser.add_argument(
        "--exclude-runtime-symbols",
        action="store_true",
        help="索引中排除 main/CRT 等（默认关；常规训练在 filter_index_by_pcode_len 默认排除即可，无需此处重复）",
    )
    parser.add_argument(
        "--exclude-names",
        default=None,
        help="与 --exclude-runtime-symbols 联用：逗号分隔额外精确符号",
    )
    parser.add_argument(
        "--exclude-names-file",
        default=None,
        metavar="PATH",
        help="与 --exclude-runtime-symbols 联用：每行一个符号",
    )
    parser.add_argument(
        "--extra-exclude-prefix",
        action="append",
        default=None,
        help="与 --exclude-runtime-symbols 联用：可重复传入额外前缀",
    )
    args = parser.parse_args()

    name_filter: Optional[Callable[[str], bool]] = None
    if args.exclude_runtime_symbols:
        names_file_abs = ""
        if args.exclude_names_file:
            names_file_abs = os.path.abspath(args.exclude_names_file)
            if not os.path.isfile(names_file_abs):
                print(f"错误: --exclude-names-file 不存在 {names_file_abs}", file=sys.stderr)
                sys.exit(1)
        extra_exact_cli: set[str] = set()
        if args.exclude_names:
            extra_exact_cli.update(
                x.strip() for x in args.exclude_names.split(",") if x.strip()
            )
        prefix_list = [p.strip() for p in (args.extra_exclude_prefix or []) if p and p.strip()]
        from utils.training_function_filter import TrainingSymbolFilter

        sym_filter = TrainingSymbolFilter(
            exclude_runtime=True,
            extra_exact=extra_exact_cli,
            extra_prefixes=tuple(prefix_list),
            names_from_file=names_file_abs if names_file_abs else None,
        )
        name_filter = sym_filter.is_excluded

    binaries: list[str] = []
    if args.from_index_file:
        idx_path = os.path.abspath(args.from_index_file)
        if not os.path.isfile(idx_path):
            print(f"错误: 索引文件不存在 {idx_path}", file=sys.stderr)
            sys.exit(1)
        try:
            binaries = _collect_binaries_from_index_file(idx_path)
        except (json.JSONDecodeError, OSError, ValueError) as e:
            print(f"错误: 读取索引失败: {e}", file=sys.stderr)
            sys.exit(1)
        if not binaries:
            print("错误: 索引中无有效二进制路径", file=sys.stderr)
            sys.exit(1)
        print(f"从索引 {args.from_index_file} 得到 {len(binaries)} 个二进制（已去重）", flush=True)
    else:
        input_dir = os.path.abspath(
            args.input_dir
            if args.input_dir is not None
            else os.path.join(PROJECT_ROOT, "data", "binkit_subset")
        )
        if not os.path.isdir(input_dir):
            print(f"错误: 输入目录不存在 {input_dir}", file=sys.stderr)
            sys.exit(1)

        for f in sorted(os.listdir(input_dir)):
            path = os.path.join(input_dir, f)
            if os.path.isfile(path):
                binaries.append(os.path.abspath(path))

        if not binaries:
            print(f"错误: 目录下无文件 {input_dir}", file=sys.stderr)
            sys.exit(1)

    # 创建 session temp_dir；脚本结束时统一删除
    if args.temp_dir:
        session_temp_dir = os.path.abspath(args.temp_dir)
        os.makedirs(session_temp_dir, exist_ok=True)
    else:
        session_temp_dir = tempfile.mkdtemp(prefix="sempatch_binkit_")

    print(f"找到 {len(binaries)} 个二进制，开始提取函数列表...", flush=True)

    try:
        from utils.ghidra_runner import GhidraEnvironmentError, require_ghidra_environment

        require_ghidra_environment()
    except GhidraEnvironmentError as e:
        print(f"错误: 本脚本需要可用的 Ghidra 环境: {e}", file=sys.stderr)
        sys.exit(1)

    from utils.ghidra_runner import peek_binary_cache, run_ghidra_analysis
    from utils.concurrency import get_parallel_workers, get_global_semaphore, bounded_task
    from concurrent.futures import ThreadPoolExecutor, as_completed

    workers = args.workers if args.workers is not None else get_parallel_workers()
    use_parallel = len(binaries) > 1 and workers > 0
    force = args.force

    def _process_binary_fast(idx_and_path):
        """快速路径：仅尝试缓存命中，不拿信号量。返回 None 表示需要走慢速路径。"""
        idx, bin_path = idx_and_path
        binary_abs = os.path.abspath(bin_path)
        rel_path = _output_binary_field(binary_abs)
        if not force:
            lsir_raw = peek_binary_cache(binary_abs)
            if lsir_raw is not None:
                return (idx, rel_path, _lsir_to_funcs(lsir_raw, name_filter=name_filter))
        return None  # 需要走慢速路径

    def _process_binary_slow(idx_and_path):
        """慢速路径：缓存未命中或 force，创建临时目录调用 Ghidra。需要在 bounded_task 内运行。"""
        idx, bin_path = idx_and_path
        binary_abs = os.path.abspath(bin_path)
        rel_path = _output_binary_field(binary_abs)

        output_subdir = os.path.join(session_temp_dir, f"v{idx}")
        os.makedirs(output_subdir, exist_ok=True)
        lsir_raw = None
        try:
            lsir_raw = run_ghidra_analysis(
                binary_path=bin_path,
                output_dir=output_subdir,
                project_name=f"BinkitIndex_{idx}",
                script_name="extract_lsir_raw.java",
                script_output_name="lsir_raw.json",
                force=force,
                return_dict=True,
            )
        except Exception as e:
            print(f"警告: 处理失败 {rel_path}: {e}", file=sys.stderr, flush=True)
        finally:
            shutil.rmtree(output_subdir, ignore_errors=True)

        if lsir_raw is None:
            return (idx, rel_path, None)
        return (idx, rel_path, _lsir_to_funcs(lsir_raw, name_filter=name_filter))

    index_slots = [None] * len(binaries)
    tasks = list(enumerate(binaries, start=1))
    sem = get_global_semaphore() if use_parallel else None

    try:
        if use_parallel:
            max_workers = min(len(binaries), workers)
            print(f"使用 {max_workers} 线程并行提取", flush=True)

            # === 第一轮：快速路径，并发检查缓存，不拿信号量 ===
            pending = []  # 缓存未命中的任务
            hit_count = 0
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                fast_futures = {
                    ex.submit(_process_binary_fast, (idx, path)): idx
                    for idx, path in tasks
                }
                for fut in as_completed(fast_futures):
                    idx = fast_futures[fut]
                    result = fut.result()
                    if result is not None:
                        _, rel_path, funcs = result
                        if funcs is not None:
                            index_slots[idx - 1] = {"binary": rel_path, "functions": funcs}
                            hit_count += 1
                            print(f"  [{idx}/{len(binaries)}] {rel_path}: {len(funcs)} 函数", flush=True)
                    else:
                        pending.append((idx, binaries[idx - 1]))

            if not force:
                print(f"缓存命中: {hit_count}/{len(binaries)}", flush=True)

            # === 第二轮：缓存未命中的，拿信号量跑 Ghidra ===
            if pending:
                print(f"缓存未命中: {len(pending)} 个二进制，启动 Ghidra 分析...", flush=True)
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    slow_futures = {
                        ex.submit(bounded_task, sem, _process_binary_slow, (idx, path)): idx
                        for idx, path in pending
                    }
                    for fut in as_completed(slow_futures):
                        idx = slow_futures[fut]
                        try:
                            _, rel_path, funcs = fut.result()
                            if funcs is not None:
                                index_slots[idx - 1] = {"binary": rel_path, "functions": funcs}
                                print(f"  [{idx}/{len(binaries)}] {rel_path}: {len(funcs)} 函数", flush=True)
                        except Exception as e:
                            print(f"  [{idx}/{len(binaries)}] 处理失败: {e}", file=sys.stderr, flush=True)
        else:
            for idx, bin_path in tasks:
                result = _process_binary_fast((idx, bin_path))
                if result is not None:
                    _, rel_path, funcs = result
                    if funcs is not None:
                        index_slots[idx - 1] = {"binary": rel_path, "functions": funcs}
                        print(f"  [{idx}/{len(binaries)}] {rel_path}: {len(funcs)} 函数", flush=True)
                else:
                    _, rel_path, funcs = _process_binary_slow((idx, bin_path))
                    if funcs is not None:
                        index_slots[idx - 1] = {"binary": rel_path, "functions": funcs}
                        print(f"  [{idx}/{len(binaries)}] {rel_path}: {len(funcs)} 函数", flush=True)
    finally:
        # session temp_dir 及所有残留子目录统一清理
        shutil.rmtree(session_temp_dir, ignore_errors=True)

    index = [e for e in index_slots if e is not None]
    total_funcs = sum(len(e["functions"]) for e in index)
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"已写入 {out_path}: {len(index)} 个二进制, {total_funcs} 个函数", flush=True)


if __name__ == "__main__":
    main()