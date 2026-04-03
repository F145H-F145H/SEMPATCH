#!/usr/bin/env python3
"""
预计算库函数的 multimodal 特征，供两阶段流水线粗筛（SAFE）与精排复用。

遍历 library_index.json 中全部库函数，对每个二进制调用一次 Ghidra 获取 lsir_raw
（优先命中 BINARY_CACHE_DIR，build_binkit_index 已 populate 则零 Ghidra 调用），
再逐函数提取 multimodal 特征，输出 library_features.json。
格式：{function_id: multimodal_dict}，function_id 为 binary_path|entry_address。

缓存策略（Plan B）：每个二进制优先从 binary_cache 直接读取（peek_binary_cache），
命中时不创建任何临时目录。未命中时在 session temp_dir 下创建子目录运行 Ghidra，
子目录用毕立即删除，session temp_dir 脚本结束时统一删除。

用法:
  python scripts/build_library_features.py
  python scripts/build_library_features.py --library-index data/two_stage/library_index.json
  python scripts/build_library_features.py --library-index lib.json --query-index query.json  # 同时输出查询特征
  python scripts/build_library_features.py --precomputed-multimodal data/two_stage/filtered_features.jsonl
  python scripts/build_library_features.py --workers 4   # 多线程并行（侧车全命中时几乎不占 CPU）
  python scripts/build_library_features.py --precomputed-eager-load  # 小 JSONL 可整表载入
  python scripts/build_library_features.py --max-lsir-in-flight 2    # 有回退提取时限制同时加载 lsir 数，防 OOM
  python scripts/build_library_features.py --serialize-sidecar-reads   # 大侧车+多 worker 时串行读盘，降峰值内存/抖动
"""
import argparse
import gc
import json
import logging
import os
import queue
import shutil
import tempfile
import sys
import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional, Set, TextIO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from utils.precomputed_multimodal_io import JsonlSidecarLazyIndex  # noqa: E402

logger = logging.getLogger(__name__)
_STREAM_QUEUE_STOP = object()


def _norm_entry(entry: str) -> str:
    """统一 entry 格式（脚本层辅助）。"""
    s = (entry or "").strip().lower()
    if s.startswith("0x"):
        return s
    return f"0x{s}" if s else "0x0"


def _function_id(binary_path: str, entry: str) -> str:
    return f"{binary_path}|{_norm_entry(entry)}"


def _collect_function_ids_from_index(items: list) -> Set[str]:
    s: Set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        binary_rel = item.get("binary", "")
        for fn in item.get("functions", []) or []:
            entry = fn.get("entry", "") if isinstance(fn, dict) else ""
            if entry:
                s.add(_function_id(binary_rel, entry))
    return s


@contextmanager
def _lsir_concurrency_slot(lsir_sem: Optional[threading.Semaphore]):
    """限制同时 json.load 整份 lsir_raw 的线程数，降低多 worker 时峰值内存。"""
    if lsir_sem is None:
        yield
        return
    lsir_sem.acquire()
    try:
        yield
    finally:
        lsir_sem.release()


def _process_single_binary(
    args_tuple: tuple,
) -> tuple:
    """
    处理单个二进制：Plan B — 先 peek binary_cache，命中则直接用；
    未命中时创建子目录，调用 Ghidra，子目录用毕立即删除。
    若侧车已覆盖本二进制全部 function_id，则完全不读 lsir_raw（省内存、省 CPU）。
    返回 (idx, binary_rel, {function_id: multimodal})。
    """
    if len(args_tuple) == 9:
        (
            idx,
            item,
            project_root,
            temp_dir,
            prefix,
            total,
            precomputed_multimodal,
            gc_after_binary,
            lsir_sem,
        ) = args_tuple
        sidecar_stream_queue = None
    else:
        (
            idx,
            item,
            project_root,
            temp_dir,
            prefix,
            total,
            precomputed_multimodal,
            gc_after_binary,
            lsir_sem,
            sidecar_stream_queue,
        ) = args_tuple
    import shutil as _shutil
    from utils.ghidra_runner import peek_binary_cache, run_ghidra_analysis
    from utils.feature_extractors import extract_multimodal_from_lsir_raw
    from utils.memory_mitigation import maybe_gc_after_binary, warn_if_large_lsir

    binary_rel = item.get("binary", "")
    if not binary_rel:
        raise ValueError(f"索引项 {idx} 缺少 binary 字段")
    binary_abs = (
        os.path.join(project_root, binary_rel)
        if not os.path.isabs(binary_rel)
        else binary_rel
    )
    funcs = item.get("functions", [])

    sidecar = precomputed_multimodal
    features: dict = {}
    precomputed_hits = 0
    pending: list = []  # (entry, fid)

    if isinstance(sidecar, JsonlSidecarLazyIndex):
        sidecar_fids: list[str] = []
        for fn in funcs:
            entry = fn.get("entry", "")
            if not entry:
                raise ValueError(f"索引项 {idx} 中函数缺少 entry")
            fid = _function_id(binary_rel, entry)
            sidecar_fids.append(fid)
            pending.append((entry, fid))
        if sidecar_stream_queue is not None:
            hit_fids: set[str] = set()
            for fid, mm in sidecar.bulk_get_iter(sidecar_fids):
                sidecar_stream_queue.put((fid, mm))
                hit_fids.add(fid)
                precomputed_hits += 1
            if precomputed_hits:
                pending = [(entry, fid) for entry, fid in pending if fid not in hit_fids]
        else:
            for fid, mm in sidecar.bulk_get_iter(sidecar_fids):
                features[fid] = mm
                precomputed_hits += 1
            if precomputed_hits:
                pending = [(entry, fid) for entry, fid in pending if fid not in features]
    elif sidecar is not None:
        for fn in funcs:
            entry = fn.get("entry", "")
            if not entry:
                raise ValueError(f"索引项 {idx} 中函数缺少 entry")
            fid = _function_id(binary_rel, entry)
            mm = sidecar.get(fid)
            if mm is not None:
                features[fid] = mm
                precomputed_hits += 1
            else:
                pending.append((entry, fid))
    else:
        for fn in funcs:
            entry = fn.get("entry", "")
            if not entry:
                raise ValueError(f"索引项 {idx} 中函数缺少 entry")
            fid = _function_id(binary_rel, entry)
            pending.append((entry, fid))

    if not pending:
        maybe_gc_after_binary(gc_after_binary)
        return (idx, binary_rel, features, precomputed_hits, 0)

    extracted_fallbacks = 0
    with _lsir_concurrency_slot(lsir_sem):
        lsir_raw = peek_binary_cache(binary_abs)
        if lsir_raw is None:
            output_dir = os.path.join(temp_dir, f"{prefix}_{idx}")
            os.makedirs(output_dir, exist_ok=True)
            try:
                lsir_raw = run_ghidra_analysis(
                    binary_path=binary_abs,
                    output_dir=output_dir,
                    project_name=f"LibFeat_{prefix}_{idx}",
                    script_name="extract_lsir_raw.java",
                    script_output_name="lsir_raw.json",
                    return_dict=True,
                )
            finally:
                _shutil.rmtree(output_dir, ignore_errors=True)

        lsir_funcs = (lsir_raw or {}).get("functions", [])
        warn_if_large_lsir(binary_label=binary_rel, num_functions=len(lsir_funcs))

        for entry, fid in pending:
            mm = extract_multimodal_from_lsir_raw(lsir_funcs, entry)
            features[fid] = mm
            extracted_fallbacks += 1
        del lsir_raw
        del lsir_funcs

    maybe_gc_after_binary(gc_after_binary)
    return (idx, binary_rel, features, precomputed_hits, extracted_fallbacks)


class _JsonObjectStreamWriter:
    """以流式方式写入 JSON 对象，避免全量聚合到内存。"""

    def __init__(self, fp: TextIO) -> None:
        self._fp = fp
        self._started = False
        self._closed = False

    def __enter__(self) -> "_JsonObjectStreamWriter":
        self._fp.write("{\n")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._closed:
            self._fp.write("\n}\n")
            self._closed = True

    def write_item(self, key: str, value: Dict[str, Any]) -> None:
        if self._started:
            self._fp.write(",\n")
        self._fp.write(json.dumps(key, ensure_ascii=False))
        self._fp.write(": ")
        self._fp.write(json.dumps(value, ensure_ascii=False))
        self._started = True

    def write_dict(self, data: Dict[str, Any]) -> int:
        if not data:
            return 0
        count = 0
        for k, v in data.items():
            self.write_item(k, v)
            count += 1
        return count


def _process_index(
    index_items: list,
    project_root: str,
    temp_dir: str,
    prefix: str = "lib",
    workers: Optional[int] = None,
    precomputed_multimodal: Optional[Any] = None,
    gc_after_each_binary: bool = False,
    lsir_sem: Optional[threading.Semaphore] = None,
    stream_writer: Optional[_JsonObjectStreamWriter] = None,
) -> tuple:
    """处理索引，返回 (函数总数, 命中总数, 回退总数, 可选features_dict)。支持多线程并行。"""
    from utils.concurrency import get_parallel_workers, get_global_semaphore, bounded_task
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n = len(index_items)
    use_parallel = n > 1 and (workers or get_parallel_workers()) > 0
    w = workers if workers is not None else get_parallel_workers()
    max_workers = min(n, w) if use_parallel else 1

    tasks = [
        (
            idx,
            item,
            project_root,
            temp_dir,
            prefix,
            n,
            precomputed_multimodal,
            gc_after_each_binary,
            lsir_sem,
            None,
        )
        for idx, item in enumerate(index_items, start=1)
    ]
    sem = get_global_semaphore() if use_parallel else None

    features: Optional[dict] = {} if stream_writer is None else None
    n_written = 0
    total_hits = 0
    total_fallbacks = 0
    sidecar_queue_mode = (
        use_parallel
        and stream_writer is not None
        and isinstance(precomputed_multimodal, JsonlSidecarLazyIndex)
    )
    sidecar_stream_queue: Optional["queue.Queue[Any]"] = None
    writer_thread: Optional[threading.Thread] = None
    queue_written = 0
    queue_written_lock = threading.Lock()

    if sidecar_queue_mode:
        sidecar_stream_queue = queue.Queue(maxsize=max(256, max_workers * 32))
        tasks = [(*t[:-1], sidecar_stream_queue) for t in tasks]

        def _drain_sidecar_queue() -> None:
            nonlocal queue_written
            assert sidecar_stream_queue is not None
            assert stream_writer is not None
            while True:
                item = sidecar_stream_queue.get()
                if item is _STREAM_QUEUE_STOP:
                    break
                fid, mm = item
                stream_writer.write_item(fid, mm)
                with queue_written_lock:
                    queue_written += 1

        writer_thread = threading.Thread(
            target=_drain_sidecar_queue,
            name="libfeat-sidecar-writer",
            daemon=True,
        )
        writer_thread.start()

    if use_parallel:
        print(f"使用 {max_workers} 线程并行处理")
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(bounded_task, sem, _process_single_binary, t): t[0]
                    for t in tasks
                }
                for fut in as_completed(futures):
                    idx = futures[fut]
                    _, binary_rel, sub_features, hits, fallbacks = fut.result()
                    n_sub = len(sub_features) + (hits if sidecar_queue_mode else 0)
                    if stream_writer is not None:
                        if sidecar_queue_mode and sidecar_stream_queue is not None:
                            for kv in sub_features.items():
                                sidecar_stream_queue.put(kv)
                        else:
                            n_written += stream_writer.write_dict(sub_features)
                    else:
                        assert features is not None
                        features.update(sub_features)
                    del sub_features
                    total_hits += hits
                    total_fallbacks += fallbacks
                    print(
                        f"  [{idx}/{n}] {binary_rel}: {n_sub} 函数 "
                        f"(预计算命中 {hits}, 回退提取 {fallbacks})"
                    )
                    if gc_after_each_binary:
                        gc.collect()
        finally:
            if writer_thread is not None and sidecar_stream_queue is not None:
                sidecar_stream_queue.put(_STREAM_QUEUE_STOP)
                writer_thread.join()
    else:
        for t in tasks:
            idx, binary_rel, sub_features, hits, fallbacks = _process_single_binary(t)
            n_sub = len(sub_features)
            if stream_writer is not None:
                n_written += stream_writer.write_dict(sub_features)
            else:
                assert features is not None
                features.update(sub_features)
            del sub_features
            total_hits += hits
            total_fallbacks += fallbacks
            print(
                f"  [{idx}/{n}] {binary_rel}: {n_sub} 函数 "
                f"(预计算命中 {hits}, 回退提取 {fallbacks})"
            )
            if gc_after_each_binary:
                gc.collect()
    if sidecar_queue_mode:
        n_written += queue_written
    n_features = n_written if stream_writer is not None else len(features or {})
    print(
        f"{prefix} 汇总: 共 {n_features} 函数, 预计算命中 {total_hits}, 回退提取 {total_fallbacks}"
    )
    return (n_features, total_hits, total_fallbacks, features)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="预计算库/查询函数的 multimodal 特征"
    )
    parser.add_argument(
        "--library-index",
        default=os.path.join(PROJECT_ROOT, "data", "two_stage", "library_index.json"),
        help="库函数索引",
    )
    parser.add_argument(
        "--query-index",
        default=None,
        help="查询集索引（可选，指定时同时输出 query_features.json）",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "data", "two_stage"),
        help="输出目录",
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
        help="并行数（默认 PARALLEL_WORKERS）",
    )
    parser.add_argument(
        "--precomputed-multimodal",
        default=None,
        help="可选：预计算侧车（.jsonl 默认仅建行偏移索引+懒加载，防 OOM；.json 整文件解析）",
    )
    parser.add_argument(
        "--precomputed-eager-load",
        action="store_true",
        help="JSONL 侧车仍整表载入内存（仅适合小文件；大侧车易 OOM）",
    )
    parser.add_argument(
        "--max-lsir-in-flight",
        type=int,
        default=1,
        help="同时持有解析后 lsir_raw 的线程上限（回退提取时有效；默认 1 降内存，0 表示不限制）",
    )
    parser.add_argument(
        "--serialize-sidecar-reads",
        action="store_true",
        help="JSONL 懒加载侧车全局串行读盘（多 worker 时避免同时解析多行；略慢，利于控内存与页缓存）",
    )
    parser.add_argument(
        "--max-memory-mb",
        type=int,
        default=None,
        help="类 Unix 上尝试 RLIMIT_AS（MiB）；或环境变量 SEMPATCH_MAX_MEMORY_MB",
    )
    parser.add_argument(
        "--gc-after-each-binary",
        action="store_true",
        help="每处理完一个二进制后 gc.collect()（多线程模式下在每个任务完成后触发）",
    )
    parser.add_argument(
        "--buffer-json",
        action="store_true",
        help="整表聚合到内存后再 json.dump（默认关闭；大数据建议保持流式写出以防 OOM）",
    )
    args = parser.parse_args()

    try:
        from utils.ghidra_runner import GhidraEnvironmentError, require_ghidra_environment

        require_ghidra_environment()
    except GhidraEnvironmentError as e:
        print(f"错误: build_library_features 需要可用的 Ghidra 环境: {e}", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from utils.memory_mitigation import configure_address_space_limit, resolve_max_memory_mb

    _lim_mb = resolve_max_memory_mb(args.max_memory_mb)
    _ok, _msg = configure_address_space_limit(_lim_mb)
    if _lim_mb:
        if _ok:
            logger.info("%s", _msg)
        else:
            logger.warning("%s", _msg)

    lib_path = os.path.abspath(args.library_index)
    if not os.path.isfile(lib_path):
        print(f"错误: 库索引不存在 {lib_path}", file=sys.stderr)
        sys.exit(1)

    with open(lib_path, encoding="utf-8") as f:
        lib_items = json.load(f)
    if not isinstance(lib_items, list):
        lib_items = [lib_items] if isinstance(lib_items, dict) else []

    from utils.precomputed_multimodal_io import (
        build_jsonl_sidecar_lazy_index,
        is_jsonl_sidecar_path,
        load_precomputed_multimodal_map,
    )

    needed_precomputed_ids = _collect_function_ids_from_index(lib_items)
    query_items: Optional[list] = None
    if args.query_index:
        query_path = os.path.abspath(args.query_index)
        if os.path.isfile(query_path):
            with open(query_path, encoding="utf-8") as f:
                query_items = json.load(f)
            if not isinstance(query_items, list):
                query_items = [query_items] if isinstance(query_items, dict) else []
            needed_precomputed_ids |= _collect_function_ids_from_index(query_items)

    precomputed_multimodal: Any = None
    if args.precomputed_multimodal:
        pc_path = os.path.abspath(args.precomputed_multimodal)
        if args.precomputed_eager_load or not is_jsonl_sidecar_path(pc_path):
            print("载入预计算侧车（整表解析到内存）…", flush=True)
            precomputed_multimodal = load_precomputed_multimodal_map(
                pc_path,
                needed_precomputed_ids,
            )
        else:
            print(
                f"扫描 JSONL 侧车行偏移索引（需命中 {len(needed_precomputed_ids)} 个 function_id）…",
                flush=True,
            )
            _sidecar_lock = threading.Lock() if args.serialize_sidecar_reads else None
            precomputed_multimodal = build_jsonl_sidecar_lazy_index(
                pc_path,
                needed_precomputed_ids,
                read_lock=_sidecar_lock,
            )
            if args.serialize_sidecar_reads:
                print("侧车读盘: 已启用全局串行（--serialize-sidecar-reads）", flush=True)
            hit = len(precomputed_multimodal)
            miss = len(needed_precomputed_ids) - hit
            print(
                f"侧车索引: 命中 {hit}, 未在侧车中找到 {miss}"
                + ("（将回退 lsir 提取）" if miss else "（本阶段不加载整份 lsir 到多线程）"),
                flush=True,
            )

    lsir_sem: Optional[threading.Semaphore] = None
    if int(args.max_lsir_in_flight) > 0:
        lsir_sem = threading.Semaphore(int(args.max_lsir_in_flight))

    # 创建 session temp_dir；脚本结束时统一删除
    if args.temp_dir:
        session_temp_dir = os.path.abspath(args.temp_dir)
        os.makedirs(session_temp_dir, exist_ok=True)
    else:
        session_temp_dir = tempfile.mkdtemp(prefix="sempatch_libfeat_")

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    try:
        print("处理库函数...")
        lib_out = os.path.join(out_dir, "library_features.json")
        if args.buffer_json:
            lib_features_count, _, _, lib_features = _process_index(
                lib_items,
                PROJECT_ROOT,
                session_temp_dir,
                prefix="lib",
                workers=args.workers,
                precomputed_multimodal=precomputed_multimodal,
                gc_after_each_binary=args.gc_after_each_binary,
                lsir_sem=lsir_sem,
                stream_writer=None,
            )
            if lib_features is None:
                raise RuntimeError("buffer-json 模式下未获得库特征结果")
            with open(lib_out, "w", encoding="utf-8") as f:
                json.dump(lib_features, f, indent=2, ensure_ascii=False)
        else:
            lib_tmp = f"{lib_out}.tmp"
            try:
                with open(lib_tmp, "w", encoding="utf-8") as f:
                    with _JsonObjectStreamWriter(f) as writer:
                        lib_features_count, _, _, _ = _process_index(
                            lib_items,
                            PROJECT_ROOT,
                            session_temp_dir,
                            prefix="lib",
                            workers=args.workers,
                            precomputed_multimodal=precomputed_multimodal,
                            gc_after_each_binary=args.gc_after_each_binary,
                            lsir_sem=lsir_sem,
                            stream_writer=writer,
                        )
                os.replace(lib_tmp, lib_out)
            except Exception:
                if os.path.exists(lib_tmp):
                    os.remove(lib_tmp)
                raise
        print(f"已写入 {lib_out} ({lib_features_count} 个函数)")

        if args.query_index:
            query_path = os.path.abspath(args.query_index)
            if not os.path.isfile(query_path):
                print(f"错误: 查询索引不存在 {query_path}", file=sys.stderr)
                sys.exit(1)
            if query_items is None:
                with open(query_path, encoding="utf-8") as f:
                    query_items = json.load(f)
                if not isinstance(query_items, list):
                    query_items = [query_items] if isinstance(query_items, dict) else []

            print("处理查询函数...")
            query_out = os.path.join(out_dir, "query_features.json")
            if args.buffer_json:
                query_features_count, _, _, query_features = _process_index(
                    query_items,
                    PROJECT_ROOT,
                    session_temp_dir,
                    prefix="query",
                    workers=args.workers,
                    precomputed_multimodal=precomputed_multimodal,
                    gc_after_each_binary=args.gc_after_each_binary,
                    lsir_sem=lsir_sem,
                    stream_writer=None,
                )
                if query_features is None:
                    raise RuntimeError("buffer-json 模式下未获得查询特征结果")
                with open(query_out, "w", encoding="utf-8") as f:
                    json.dump(query_features, f, indent=2, ensure_ascii=False)
            else:
                query_tmp = f"{query_out}.tmp"
                try:
                    with open(query_tmp, "w", encoding="utf-8") as f:
                        with _JsonObjectStreamWriter(f) as writer:
                            query_features_count, _, _, _ = _process_index(
                                query_items,
                                PROJECT_ROOT,
                                session_temp_dir,
                                prefix="query",
                                workers=args.workers,
                                precomputed_multimodal=precomputed_multimodal,
                                gc_after_each_binary=args.gc_after_each_binary,
                                lsir_sem=lsir_sem,
                                stream_writer=writer,
                            )
                    os.replace(query_tmp, query_out)
                except Exception:
                    if os.path.exists(query_tmp):
                        os.remove(query_tmp)
                    raise
            print(f"已写入 {query_out} ({query_features_count} 个函数)")
    finally:
        shutil.rmtree(session_temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
