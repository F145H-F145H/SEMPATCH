#!/usr/bin/env python3
"""
按 pcode_tokens 长度预过滤索引。

遍历索引中所有函数，提取 multimodal 特征，仅保留 len(pcode_tokens) >= min_pcode_len 的函数。
默认排除 main、__libc_start_main 等 CRT/启动胶水符号（见 utils.training_function_filter），不进行 multimodal 提取。
输出格式与 binkit_functions.json 一致，供 prepare_two_stage_data、build_library_features、train_safe 使用。

缓存策略（Plan B）：每个二进制优先从 binary_cache 直接读取（peek_binary_cache），
命中时不创建任何临时目录。未命中时在 session temp_dir 下创建子目录运行 Ghidra，
子目录用毕立即删除，session temp_dir 脚本结束时统一删除。

并行模型：
  - 二进制维度串行：同一时间仅驻留一份 lsir_raw，避免 seen_binaries 式无界缓存导致 OOM。
  - 函数维度多进程：extract_multimodal 为 CPU 密集且受 GIL 限制，使用 ProcessPoolExecutor（Unix 上优选 fork 上下文）
    真并行；每个任务只序列化单函数的 LSIR 片段，不把整份二进制传入子进程。

用法:
  python scripts/filter_index_by_pcode_len.py --input data/binkit_functions.json --output data/binkit_functions_filtered.json
  python scripts/filter_index_by_pcode_len.py --input index.json -o out.json --min-pcode-len 16
  python scripts/filter_index_by_pcode_len.py -i index.json -o out.json --filtered-features-output data/two_stage/filtered_features.jsonl
  python scripts/filter_index_by_pcode_len.py -i index.json -o out.json --workers 4   # 函数级多进程
  python scripts/filter_index_by_pcode_len.py -i index.json -o out.json --resume      # 从 checkpoint 续跑
  python scripts/filter_index_by_pcode_len.py -i index.json -o out.json --no-exclude-runtime-symbols  # 保留 main/CRT
"""
import argparse
import json
import logging
import multiprocessing
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

logger = logging.getLogger(__name__)


def _norm_entry(entry: str) -> str:
    """统一 entry 格式。"""
    s = (entry or "").strip().lower()
    if s.startswith("0x"):
        return s
    return f"0x{s}" if s else "0x0"


def _function_id(binary_path: str, entry: str) -> str:
    return f"{binary_path}|{_norm_entry(entry)}"


def _build_lsir_entry_index(lsir_funcs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    将 lsir_raw.functions 建成 entry -> 函数 dict 的映射（与 extract 中首次匹配一致，重复 entry 保留先出现的）。
    """
    idx: Dict[str, Dict[str, Any]] = {}
    for f in lsir_funcs:
        key = _norm_entry(f.get("entry", ""))
        if key and key != "0x0" and key not in idx:
            idx[key] = f
    return idx


def _is_trivial_getset_symbol(fn_name: str, mm: Dict[str, Any]) -> bool:
    """保守启发式：get*/set*/is* + 极短序列 + 极少 CFG 节点 → 琐碎存取子。"""
    import re

    from utils.training_function_filter import strip_linker_suffix

    n = strip_linker_suffix((fn_name or "").strip())
    if not n or not re.match(r"^(get|set|is)([_A-Z]|$)", n, re.IGNORECASE):
        return False
    seq = mm.get("sequence") or {}
    sl = int(seq.get("seq_len") or len(seq.get("pcode_tokens") or []))
    nn = int((mm.get("graph") or {}).get("num_nodes") or 0)
    return sl <= 16 and nn <= 3


def _pcode_filter_worker(payload: Tuple[Dict[str, Any], int, int, bool, str]) -> Tuple[str, Any]:
    """
    子进程任务：仅携带单函数 LSIR 片段，不把整份 lsir_raw 传入子进程。

    payload: (target, min_pcode_len, min_basic_blocks, exclude_getter_setter, fn_name)

    Returns:
        ("keep", multimodal:dict) | ("short", token_len:int) | ("short_bb", nn:int)
        | ("trivial_getset", 0) | ("exc", err:str)
    """
    target, min_pcode_len, min_basic_blocks, exclude_getter_setter, fn_name = payload
    try:
        from utils.feature_extractors import extract_multimodal_from_lsir_raw

        entry = target.get("entry", "")
        mm = extract_multimodal_from_lsir_raw([target], entry)
        tokens = mm.get("sequence", {}).get("pcode_tokens") or []
        ntok = len(tokens)
        if ntok < min_pcode_len:
            return ("short", ntok)
        nn = int((mm.get("graph") or {}).get("num_nodes") or 0)
        if min_basic_blocks > 0 and nn < min_basic_blocks:
            return ("short_bb", nn)
        if exclude_getter_setter and _is_trivial_getset_symbol(fn_name, mm):
            return ("trivial_getset", 0)
        return ("keep", mm)
    except Exception as e:
        return ("exc", str(e))


def _process_pool_executor(
    max_workers: int,
    *,
    pool_max_tasks_per_child: int = 0,
) -> Optional[ProcessPoolExecutor]:
    """在支持 fork 的平台上使用 fork 上下文。max_tasks_per_child 与 fork 不兼容，由 memory_mitigation 自动忽略。"""
    if max_workers <= 0:
        return None
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:
        return None
    from utils.memory_mitigation import build_process_pool_executor_kwargs

    kwargs = build_process_pool_executor_kwargs(
        max_workers=max_workers,
        mp_context=ctx,
        max_tasks_per_child=pool_max_tasks_per_child,
    )
    return ProcessPoolExecutor(**kwargs)


def _fork_process_pool_available() -> bool:
    """是否可用 fork 多进程（无需实际创建 Pool，避免泄露句柄）。"""
    try:
        multiprocessing.get_context("fork")
        return True
    except ValueError:
        return False


def _lookup_lsir_target(
    entry_index: Dict[str, Dict[str, Any]],
    lsir_funcs: List[Dict[str, Any]],
    entry: str,
) -> Optional[Dict[str, Any]]:
    """在 lsir 函数列表中解析 entry：先 O(1) 查表，再回退到地址等价匹配（与 multimodal_extraction 一致）。"""
    from utils.feature_extractors.multimodal_extraction import _entry_matches

    key = _norm_entry(entry)
    hit = entry_index.get(key)
    if hit is not None:
        return hit
    for f in lsir_funcs:
        if _entry_matches(f.get("entry", ""), entry):
            return f
    return None


def _filter_index(
    index_items: list,
    project_root: str,
    temp_dir: str,
    min_pcode_len: int,
    min_basic_blocks: int = 0,
    exclude_getter_setter: bool = False,
    workers: Optional[int] = None,
    kept_features: Optional[Dict[str, Dict[str, Any]]] = None,
    feature_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    gc_after_each_binary: bool = False,
    pool_max_tasks_per_child: int = 0,
    resume_state: Optional[Dict[str, Any]] = None,
    on_binary_done: Optional[Callable[[Dict[str, Any]], None]] = None,
    name_filter: Optional[Callable[[str], bool]] = None,
) -> list:
    """
    遍历索引，提取每个函数的 multimodal，仅保留 len(pcode_tokens) >= min_pcode_len 的函数。
    返回过滤后的索引列表，格式与输入一致。

    name_filter：若提供，对索引项中的函数名调用 name_filter(name)；返回 True 则跳过（不提取 multimodal）。

    缓存策略（Plan B）：先 peek binary_cache，命中时不创建临时目录；
    未命中时创建子目录运行 Ghidra，子目录用毕立即删除（成功或异常均删除）。

    内存：按 binary_abs 分组并串行处理，处理完即释放该二进制的 lsir_raw，避免跨二进制累积。
    CPU：同一索引行内对多函数使用进程池并行 extract（--workers）；Ghidra 仍受全局信号量限制。

    内存：若提供 feature_sink（JSONL 流式写出），不在内存中累积全量 map；进程池使用 map(chunksize=1)
    并逐条消费，避免 list(map(...)) 一次性缓存全部 multimodal。
    """
    from utils.concurrency import get_parallel_workers, get_global_semaphore, bounded_task
    from utils.ghidra_runner import peek_binary_cache, run_ghidra_analysis
    from utils.memory_mitigation import maybe_gc_after_binary, warn_if_large_lsir

    n = len(index_items)
    sem = get_global_semaphore()
    w = workers if workers is not None else get_parallel_workers()
    cpu_n = multiprocessing.cpu_count() or 1
    # 函数级进程池规模：不超过 CPU、用户 workers 与单批任务数
    proc_cap = max(0, min(cpu_n, w)) if w > 0 else 0

    groups: Dict[str, List[Tuple[int, dict]]] = {}
    for idx0, item in enumerate(index_items):
        idx = idx0 + 1
        binary_rel = item.get("binary", "")
        if not binary_rel:
            raise ValueError(f"索引项 {idx} 缺少 binary 字段")
        binary_abs = (
            os.path.join(project_root, binary_rel)
            if not os.path.isabs(binary_rel)
            else binary_rel
        )
        groups.setdefault(binary_abs, []).append((idx, item))

    def _group_order_key(bpath: str) -> int:
        return min(i for i, _ in groups[bpath])

    sorted_binaries = sorted(groups.keys(), key=_group_order_key)

    if proc_cap > 1 and _fork_process_pool_available():
        print(f"函数级并行：最多 {proc_cap} 个进程（每二进制内）")
    elif proc_cap > 1:
        logger.warning(
            "当前环境无法使用 fork 进程池，函数提取将退回主进程串行（可避免 OOM，但 CPU 占用较低）"
        )

    if resume_state is not None:
        raw_slots = resume_state.get("slots")
        if not isinstance(raw_slots, list) or len(raw_slots) != n:
            raise ValueError(
                f"checkpoint slots 形状不匹配: got={type(raw_slots).__name__}/len="
                f"{len(raw_slots) if isinstance(raw_slots, list) else 'NA'}, expect_len={n}"
            )
        slots: List[Optional[dict]] = raw_slots
        raw_counters = resume_state.get("counters")
        if not isinstance(raw_counters, dict):
            raw_counters = {}
        total_original = int(raw_counters.get("total_original", 0))
        total_kept = int(raw_counters.get("total_kept", 0))
        binaries_dropped = int(raw_counters.get("binaries_dropped", 0))
        completed_binaries: Set[str] = {
            os.path.abspath(x)
            for x in (resume_state.get("completed_binaries") or [])
            if isinstance(x, str) and x
        }
    else:
        total_original = 0
        total_kept = 0
        binaries_dropped = 0
        slots = [None] * n
        completed_binaries = set()

    for b_i, binary_abs in enumerate(sorted_binaries, start=1):
        if binary_abs in completed_binaries:
            logger.info(
                "[%d/%d] 跳过已完成二进制: %s",
                b_i,
                len(sorted_binaries),
                binary_abs,
            )
            continue
        entries = groups[binary_abs]
        first_idx = min(i for i, _ in entries)
        binary_feature_buffer: List[Tuple[str, Dict[str, Any]]] = []

        lsir_raw: Any = peek_binary_cache(binary_abs)
        ghidra_failed = False

        if lsir_raw is None:
            tmp_dir = tempfile.mkdtemp(dir=temp_dir, prefix=f"filter_{first_idx}_")
            try:

                def _run_ghidra() -> Any:
                    return run_ghidra_analysis(
                        binary_path=binary_abs,
                        output_dir=tmp_dir,
                        project_name=f"Filter_{first_idx}",
                        script_name="extract_lsir_raw.java",
                        script_output_name="lsir_raw.json",
                        return_dict=True,
                    )

                try:
                    lsir_raw = bounded_task(sem, _run_ghidra)
                except Exception as e:
                    logger.warning(
                        "Ghidra 处理失败 %s: %s", binary_abs, e, exc_info=True
                    )
                    ghidra_failed = True
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        if ghidra_failed:
            for idx, _item in entries:
                slots[idx - 1] = None
                binaries_dropped += 1
            completed_binaries.add(binary_abs)
            if on_binary_done is not None:
                on_binary_done(
                    {
                        "completed_binary": binary_abs,
                        "completed_binaries": sorted(completed_binaries),
                        "slots": slots,
                        "counters": {
                            "total_original": total_original,
                            "total_kept": total_kept,
                            "binaries_dropped": binaries_dropped,
                        },
                    }
                )
            continue

        lsir_blob = lsir_raw or {}
        lsir_funcs = list(lsir_blob.get("functions") or [])
        warn_if_large_lsir(binary_label=binary_abs, num_functions=len(lsir_funcs))
        entry_index = _build_lsir_entry_index(lsir_funcs)

        # 每个二进制只建一次进程池，避免每条索引行反复 fork/shutdown。
        ex_bin = (
            _process_pool_executor(proc_cap, pool_max_tasks_per_child=pool_max_tasks_per_child)
            if proc_cap > 1
            else None
        )

        def _one_index_row(idx: int, item: dict) -> None:
            nonlocal total_original, total_kept, binaries_dropped

            binary_rel = item.get("binary", "")
            funcs = item.get("functions", [])
            kept_funcs: list = []
            row_total_orig = 0
            row_total_kept = 0

            work: List[Tuple[dict, Dict[str, Any]]] = []
            for fn in funcs:
                entry = fn.get("entry", "")
                if not entry:
                    logger.warning("索引项 %d 中函数缺少 entry，跳过", idx)
                    continue

                fn_name = (fn.get("name") or "").strip()
                if name_filter is not None and name_filter(fn_name):
                    logger.debug(
                        "跳过 %s: 训练符号排除",
                        _function_id(binary_rel, entry),
                    )
                    continue

                row_total_orig += 1
                target = _lookup_lsir_target(entry_index, lsir_funcs, entry)
                if target is None:
                    logger.warning(
                        "跳过 %s: entry 在 lsir_raw 中未找到",
                        _function_id(binary_rel, entry),
                    )
                    continue
                work.append((fn, target))

            payloads = [
                (
                    t,
                    min_pcode_len,
                    int(min_basic_blocks),
                    bool(exclude_getter_setter),
                    (fn.get("name") or ""),
                )
                for fn, t in work
            ]
            n_work = len(payloads)

            if ex_bin is not None and n_work > 1:
                # chunksize=1 + 迭代消费，降低进程结果队列中同时驻留的大对象数量
                chunksize = 1 if feature_sink is not None else max(1, n_work // (4 * proc_cap + 1))
                result_iter = ex_bin.map(_pcode_filter_worker, payloads, chunksize=chunksize)
                pairs = zip(work, result_iter)
            else:
                pairs = ((wrow, _pcode_filter_worker(p)) for wrow, p in zip(work, payloads))

            for (fn, _t), (status, detail) in pairs:
                fid = _function_id(binary_rel, fn.get("entry", ""))
                if status == "keep":
                    kept_funcs.append(fn)
                    row_total_kept += 1
                    if isinstance(detail, dict):
                        if feature_sink is not None:
                            binary_feature_buffer.append((fid, detail))
                        if kept_features is not None:
                            kept_features[fid] = detail
                elif status == "short":
                    logger.debug(
                        "跳过 %s: len(pcode_tokens)=%s < %d",
                        fid,
                        detail,
                        min_pcode_len,
                    )
                elif status == "short_bb":
                    logger.debug(
                        "跳过 %s: graph.num_nodes=%s < min_basic_blocks=%d",
                        fid,
                        detail,
                        min_basic_blocks,
                    )
                elif status == "trivial_getset":
                    logger.debug("跳过 %s: trivial getter/setter 启发式", fid)
                else:
                    logger.warning("跳过 %s: %s", fid, detail)

            if kept_funcs:
                slots[idx - 1] = {"binary": binary_rel, "functions": kept_funcs}
                bdrop = 0
            else:
                slots[idx - 1] = None
                bdrop = 1

            total_original += row_total_orig
            total_kept += row_total_kept
            binaries_dropped += bdrop

            logger.info(
                "[%d/%d] [%d/%d] %s: %d/%d 函数保留",
                idx,
                n,
                b_i,
                len(sorted_binaries),
                binary_rel,
                row_total_kept,
                len(funcs),
            )

        ordered = sorted(entries, key=lambda x: x[0])
        if ex_bin is not None:
            with ex_bin:
                for idx, item in ordered:
                    _one_index_row(idx, item)
        else:
            for idx, item in ordered:
                _one_index_row(idx, item)

        if feature_sink is not None and binary_feature_buffer:
            for fid, mm in binary_feature_buffer:
                feature_sink(fid, mm)

        completed_binaries.add(binary_abs)
        if on_binary_done is not None:
            on_binary_done(
                {
                    "completed_binary": binary_abs,
                    "completed_binaries": sorted(completed_binaries),
                    "slots": slots,
                    "counters": {
                        "total_original": total_original,
                        "total_kept": total_kept,
                        "binaries_dropped": binaries_dropped,
                    },
                }
            )

        del lsir_blob, lsir_funcs, entry_index, lsir_raw
        maybe_gc_after_binary(gc_after_each_binary)

    output_items = [row for row in slots if row is not None]

    print(f"原始函数总数: {total_original}, 保留: {total_kept}, 丢弃的二进制数: {binaries_dropped}")
    return output_items


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="按 pcode_tokens 长度预过滤索引")
    parser.add_argument("--input", "-i", required=True, help="输入索引路径（如 binkit_functions.json）")
    parser.add_argument("--output", "-o", required=True, help="输出索引路径（如 binkit_functions_filtered.json）")
    parser.add_argument("--min-pcode-len", type=int, default=16, help="最小 pcode_tokens 长度阈值")
    parser.add_argument(
        "--min-basic-blocks",
        type=int,
        default=0,
        metavar="N",
        help="最小 CFG 节点数（multimodal.graph.num_nodes，0 表示不限制）",
    )
    parser.add_argument(
        "--exclude-getter-setter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="排除名称形如 get*/set*/is* 且序列/CFG 极短的琐碎存取子（默认关）",
    )
    parser.add_argument(
        "--exclude-libc-common",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="合并排除 libc_common_exact.txt 中的标准库符号（默认开；--no-exclude-libc-common 关闭）",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Ghidra 临时目录（调试用，默认使用系统临时目录；脚本结束后均会删除）",
    )
    parser.add_argument("--project-root", default=PROJECT_ROOT, help="项目根目录")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="函数级并行进程数（默认 PARALLEL_WORKERS，上限为 CPU 核数；--workers 0 表示主进程串行提取）",
    )
    parser.add_argument(
        "--filtered-features-output",
        default=None,
        help="可选：输出过滤后保留函数的 multimodal 侧车文件路径",
    )
    parser.add_argument(
        "--filtered-features-format",
        choices=("jsonl", "json"),
        default="jsonl",
        help="侧车格式：jsonl 为逐行流式（低内存，推荐）；json 为单文件大对象（仅适合小库，易 OOM）",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint 路径（默认: <output>.filter_checkpoint.json）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 checkpoint 断点续跑（仅 JSONL 侧车支持）",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="忽略并清理已有 checkpoint，从头开始（JSONL 侧车会重新覆盖）",
    )
    parser.add_argument(
        "--max-memory-mb",
        type=int,
        default=None,
        help="类 Unix 上尝试设置进程虚拟地址空间上限（RLIMIT_AS，MiB）；也可用环境变量 SEMPATCH_MAX_MEMORY_MB",
    )
    parser.add_argument(
        "--gc-after-each-binary",
        action="store_true",
        help="每处理完一个二进制后执行 gc.collect()，略降速、可能缓解峰值",
    )
    parser.add_argument(
        "--process-pool-recycle-after-tasks",
        type=int,
        default=0,
        metavar="N",
        help="保留参数：与 fork 进程池不兼容（CPython 限制），当前会被忽略；请用较小 --workers 或 --gc-after-each-binary",
    )
    parser.add_argument(
        "--exclude-runtime-symbols",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="排除 main、__libc_start_main 等 CRT/启动符号（默认开；--no-exclude-runtime-symbols 关闭）",
    )
    parser.add_argument(
        "--exclude-names",
        default=None,
        help="逗号分隔的额外精确符号名（与内置表合并）",
    )
    parser.add_argument(
        "--exclude-names-file",
        default=None,
        metavar="PATH",
        help="每行一个符号（# 行为注释）；与 --exclude-names 合并",
    )
    parser.add_argument(
        "--extra-exclude-prefix",
        action="append",
        default=None,
        help="额外前缀排除，可重复传入（例: --extra-exclude-prefix _dl_）",
    )
    args = parser.parse_args()

    if args.resume and args.fresh:
        print("错误: --resume 与 --fresh 不能同时使用", file=sys.stderr)
        sys.exit(1)
    if args.resume and args.filtered_features_format == "json":
        print("错误: --resume 仅支持 --filtered-features-format jsonl", file=sys.stderr)
        sys.exit(1)

    names_file_abs = ""
    if args.exclude_names_file:
        names_file_abs = os.path.abspath(args.exclude_names_file)
        if not os.path.isfile(names_file_abs):
            print(f"错误: --exclude-names-file 不存在 {names_file_abs}", file=sys.stderr)
            sys.exit(1)

    extra_exact_cli: Set[str] = set()
    if args.exclude_names:
        extra_exact_cli.update(
            x.strip() for x in args.exclude_names.split(",") if x.strip()
        )
    prefix_list = [p.strip() for p in (args.extra_exclude_prefix or []) if p and p.strip()]

    try:
        from utils.ghidra_runner import GhidraEnvironmentError, require_ghidra_environment

        require_ghidra_environment()
    except GhidraEnvironmentError as e:
        print(f"错误: filter_index_by_pcode_len 需要可用的 Ghidra 环境: {e}", file=sys.stderr)
        sys.exit(1)

    from utils.filter_checkpoint import (
        CHECKPOINT_VERSION,
        build_default_checkpoint_path,
        compute_file_sha256,
        load_checkpoint,
        save_checkpoint_atomic,
        validate_checkpoint_meta,
    )
    from utils.memory_mitigation import configure_address_space_limit, resolve_max_memory_mb
    from utils.training_function_filter import TrainingSymbolFilter

    exclude_names_file_sha256 = (
        compute_file_sha256(names_file_abs) if names_file_abs else ""
    )
    sym_filter = TrainingSymbolFilter(
        exclude_runtime=bool(args.exclude_runtime_symbols),
        extra_exact=extra_exact_cli,
        extra_prefixes=tuple(prefix_list),
        names_from_file=names_file_abs if names_file_abs else None,
        include_libc_common=bool(args.exclude_libc_common),
    )

    _lim_mb = resolve_max_memory_mb(args.max_memory_mb)
    _ok, _msg = configure_address_space_limit(_lim_mb)
    if _lim_mb:
        if _ok:
            logger.info("%s", _msg)
        else:
            logger.warning("%s", _msg)

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"错误: 输入索引不存在 {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        index_items = [raw] if isinstance(raw, dict) else []
    else:
        index_items = raw

    if not index_items:
        print("错误: 输入索引为空", file=sys.stderr)
        sys.exit(1)

    out_path = os.path.abspath(args.output)
    checkpoint_path = (
        os.path.abspath(args.checkpoint)
        if args.checkpoint
        else build_default_checkpoint_path(out_path)
    )

    if args.fresh and os.path.isfile(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except OSError as e:
            print(f"错误: 无法删除旧 checkpoint {checkpoint_path}: {e}", file=sys.stderr)
            sys.exit(1)

    # 创建 session temp_dir；脚本结束时统一删除
    if args.temp_dir:
        session_temp_dir = os.path.abspath(args.temp_dir)
        os.makedirs(session_temp_dir, exist_ok=True)
    else:
        session_temp_dir = tempfile.mkdtemp(prefix="sempatch_filter_")

    project_root = os.path.abspath(args.project_root)
    kept_features: Optional[Dict[str, Dict[str, Any]]] = (
        {} if args.filtered_features_output and args.filtered_features_format == "json" else None
    )
    feature_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None
    on_binary_done: Optional[Callable[[Dict[str, Any]], None]] = None
    sidecar_fp = None
    sidecar_count = 0
    resume_state: Optional[Dict[str, Any]] = None

    feat_out = os.path.abspath(args.filtered_features_output) if args.filtered_features_output else ""
    input_sha256 = compute_file_sha256(input_path)
    expected_meta: Dict[str, Any] = {
        "input_path": input_path,
        "input_sha256": input_sha256,
        "project_root": project_root,
        "min_pcode_len": int(args.min_pcode_len),
        "min_basic_blocks": int(args.min_basic_blocks),
        "exclude_getter_setter": bool(args.exclude_getter_setter),
        "exclude_libc_common": bool(args.exclude_libc_common),
        "filtered_features_output": feat_out,
        "filtered_features_format": args.filtered_features_format,
        "exclude_runtime_symbols": bool(args.exclude_runtime_symbols),
        "extra_exclude_names": sorted(extra_exact_cli),
        "extra_exclude_prefixes": sorted(prefix_list),
        "exclude_names_file_sha256": exclude_names_file_sha256,
    }

    if args.resume:
        if not os.path.isfile(checkpoint_path):
            print(f"错误: 指定了 --resume 但 checkpoint 不存在: {checkpoint_path}", file=sys.stderr)
            sys.exit(1)
        try:
            loaded = load_checkpoint(checkpoint_path)
        except Exception as e:
            print(f"错误: 无法读取 checkpoint {checkpoint_path}: {e}", file=sys.stderr)
            sys.exit(1)
        ok, msg = validate_checkpoint_meta(loaded, expected_meta)
        if not ok:
            print(f"错误: checkpoint 校验失败: {msg}", file=sys.stderr)
            print("提示: 可改用 --fresh 从头开始", file=sys.stderr)
            sys.exit(1)
        resume_state = {
            "slots": loaded.get("slots"),
            "completed_binaries": loaded.get("completed_binaries"),
            "counters": loaded.get("counters"),
        }
        counters = loaded.get("counters") if isinstance(loaded.get("counters"), dict) else {}
        sidecar_count = int(counters.get("sidecar_count", 0))

    if args.filtered_features_output:
        feat_out_dir = os.path.dirname(feat_out)
        if feat_out_dir:
            os.makedirs(feat_out_dir, exist_ok=True)
        if args.filtered_features_format == "jsonl":
            from utils.precomputed_multimodal_io import write_jsonl_sidecar_line

            if args.resume and not os.path.isfile(feat_out):
                print(f"错误: --resume 需要已有 JSONL 侧车文件: {feat_out}", file=sys.stderr)
                sys.exit(1)
            mode = "a" if args.resume else "w"
            sidecar_fp = open(feat_out, mode, encoding="utf-8")

            def feature_sink(fid: str, mm: Dict[str, Any]) -> None:
                nonlocal sidecar_count
                write_jsonl_sidecar_line(sidecar_fp, fid, mm)
                sidecar_count += 1

    def _on_binary_done(state: Dict[str, Any]) -> None:
        if sidecar_fp is not None:
            sidecar_fp.flush()
            os.fsync(sidecar_fp.fileno())
        payload = {
            "version": CHECKPOINT_VERSION,
            "meta": expected_meta,
            "completed_binaries": state.get("completed_binaries") or [],
            "slots": state.get("slots") or [],
            "counters": {
                **(state.get("counters") or {}),
                "sidecar_count": sidecar_count,
            },
        }
        save_checkpoint_atomic(checkpoint_path, payload)

    on_binary_done = _on_binary_done

    try:
        output_items = _filter_index(
            index_items,
            project_root,
            session_temp_dir,
            args.min_pcode_len,
            min_basic_blocks=int(args.min_basic_blocks),
            exclude_getter_setter=bool(args.exclude_getter_setter),
            workers=args.workers,
            kept_features=kept_features,
            feature_sink=feature_sink,
            gc_after_each_binary=args.gc_after_each_binary,
            pool_max_tasks_per_child=max(0, args.process_pool_recycle_after_tasks),
            resume_state=resume_state,
            on_binary_done=on_binary_done,
            name_filter=sym_filter.is_excluded,
        )
    finally:
        if sidecar_fp is not None:
            try:
                sidecar_fp.close()
            except Exception:
                pass
        shutil.rmtree(session_temp_dir, ignore_errors=True)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_items, f, indent=2, ensure_ascii=False)

    print(f"已写入 {out_path} ({len(output_items)} 个二进制, {sum(len(x['functions']) for x in output_items)} 个函数)")
    if args.filtered_features_output:
        feat_out = os.path.abspath(args.filtered_features_output)
        if args.filtered_features_format == "jsonl":
            print(f"已写入 {feat_out} ({sidecar_count} 条保留特征, JSONL)")
        else:
            with open(feat_out, "w", encoding="utf-8") as f:
                json.dump(kept_features or {}, f, indent=2, ensure_ascii=False)
            print(f"已写入 {feat_out} ({len(kept_features or {})} 条保留特征, JSON)")

    if os.path.isfile(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except OSError:
            logger.warning("清理 checkpoint 失败: %s", checkpoint_path)


if __name__ == "__main__":
    main()
