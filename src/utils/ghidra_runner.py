"""Ghidra headless 分析封装。支持返回文件路径或 in-memory dict（供 DAG 使用）。

公开导出：
  run_ghidra_analysis  — 单二进制分析入口
  batch_run_ghidra     — 批量分析入口
  peek_binary_cache    — 直接从 binary_cache 读取 lsir_raw dict（不触发 Ghidra）
"""

import json
import os
from typing import List, Optional

from config import LOG_DIR, PROJECT_ROOT
from utils.logger import get_logger

from ._ghidra_helpers import (
    GhidraEnvironmentError,
    build_ghidra_command,
    can_skip_ghidra,
    execute_ghidra_process,
    peek_binary_cache,
    read_from_binary_cache,
    validate_ghidra_environment,
    write_to_binary_cache,
)


def require_ghidra_environment() -> None:
    """
    对外入口：在 extract / match --query-binary 等路径上尽早调用。
    未配置 GHIDRA_HOME 或 analyzeHeadless 不可执行时抛出 GhidraEnvironmentError。
    """
    validate_ghidra_environment()

logger = get_logger(
    name="SemPatch.GhidraRunner",
    level=None,
    format_type="colored",
)

_DEFAULT_SCRIPT_DIR = os.path.join(PROJECT_ROOT, "src", "utils", "ghidra_scripts")


def run_ghidra_analysis(
    binary_path: str,
    output_dir: str,
    project_name: str = "SemPatchProject",
    script_dir: Optional[str] = None,
    script_name: Optional[str] = "extract_lsir_raw.java",
    script_output_name: Optional[str] = None,
    timeout: Optional[int] = None,
    force: bool = False,
    return_dict: bool = False,
):
    """
    处理单个二进制文件的 Ghidra headless 分析。
    return_dict=True 时返回 in-memory dict（供 DAG ctx），否则返回输出文件路径。
    script_output_name: 输出文件名，未指定时根据 script_name 推断
      （extract_function_list -> function_list.json，否则 lsir_raw.json）。
    """
    validate_ghidra_environment()

    if script_dir is None:
        script_dir = _DEFAULT_SCRIPT_DIR

    if script_output_name is None:
        script_output_name = (
            "function_list.json" if script_name and "function_list" in script_name else "lsir_raw.json"
        )

    binary_path = os.path.abspath(binary_path)

    # 当 return_dict=True 且非强制模式时，优先从 binary_cache 直接读取 dict，
    # 无需创建 output_dir 或调用 Ghidra。
    if return_dict and not force and script_output_name == "lsir_raw.json":
        cached = peek_binary_cache(binary_path)
        if cached is not None:
            logger.info("Binary cache hit (peek): %s", binary_path)
            return cached

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    script_output_path = os.path.join(output_dir, script_output_name)
    if not force and can_skip_ghidra(binary_path, script_output_path):
        logger.info("Reusing existing output (mtime ok): %s", script_output_path)
        if return_dict:
            with open(script_output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return script_output_path

    if not force and script_output_name == "lsir_raw.json" and read_from_binary_cache(binary_path, script_output_path):
        if return_dict:
            with open(script_output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return script_output_path

    ghidra_project_dir = os.path.join(output_dir, "ghidra_project")
    os.makedirs(ghidra_project_dir, exist_ok=True)

    cmd = build_ghidra_command(
        ghidra_project_dir=ghidra_project_dir,
        project_name=project_name,
        binary_path=binary_path,
        script_dir=script_dir,
        script_name=script_name,
        script_output_path=script_output_path,
    )

    logger.structured(
        "Launching Ghidra headless analysis",
        binary=binary_path,
        project_dir=ghidra_project_dir,
        script=script_name,
    )
    logger.debug("Ghidra command: %s", " ".join(cmd))

    ghidra_log_path = os.path.join(LOG_DIR, "ghidra.log")

    try:
        return_code, _ = execute_ghidra_process(cmd, binary_path, timeout)

        if return_code != 0:
            logger.error(
                "Ghidra analysis failed (exit %s), log: %s",
                return_code,
                ghidra_log_path,
            )
            raise RuntimeError(f"Ghidra exited with code {return_code}")

        logger.info("Ghidra headless analysis completed: %s (log: %s)", binary_path, ghidra_log_path)

        # 验证输出文件非空（Ghidra exit 0 但输出为空是已知失败模式）
        if os.path.isfile(script_output_path) and os.path.getsize(script_output_path) == 0:
            raise RuntimeError(
                f"Ghidra exited 0 but output is empty: {script_output_path}"
            )

        write_to_binary_cache(script_output_path, script_output_name, binary_path)

        if return_dict:
            with open(script_output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return script_output_path

    except Exception:
        logger.exception(
            "Unexpected error during Ghidra execution: %s (log: %s)",
            binary_path,
            ghidra_log_path,
        )
        raise


def batch_run_ghidra(
    binary_paths: List[str],
    output_root: str = "output",
    project_name: str = "SemPatchProject",
    script_dir: Optional[str] = None,
    script_name: Optional[str] = "extract_lsir_raw.java",
    timeout: Optional[int] = None,
    force: bool = False,
) -> list:
    """批量处理多个二进制，返回 lsir_raw.json 路径列表。"""
    try:
        from utils.concurrency import get_parallel_workers

        workers = get_parallel_workers()
    except ImportError:
        workers = 0

    def _run_one(args):
        idx, bin_path = args
        output_dir = os.path.join(output_root, f"v{idx}")
        logger.progress(f"Processing {bin_path} -> {output_dir}")
        try:
            json_path = run_ghidra_analysis(
                binary_path=bin_path,
                output_dir=output_dir,
                project_name=f"{project_name}_v{idx}",
                script_dir=script_dir,
                script_name=script_name,
                timeout=timeout,
                force=force,
            )
            logger.success(f"Generated: {json_path}")
            return (idx, json_path)
        except Exception as e:
            logger.fail(f"Failed to process {bin_path}: {e}")
            return (idx, None)

    use_parallel = len(binary_paths) > 1 and workers > 0
    if use_parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from utils.concurrency import bounded_task, get_global_semaphore

        max_workers = min(len(binary_paths), workers)
        sem = get_global_semaphore()
        logger.info(f"Ghidra batch: parallel ({max_workers} workers)")
        results = [None] * len(binary_paths)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(bounded_task, sem, _run_one, (idx, bin_path)): idx
                for idx, bin_path in enumerate(binary_paths, start=1)
            }
            for future in as_completed(futures):
                idx, json_path = future.result()
                results[idx - 1] = json_path
        succeeded = [p for p in results if p is not None]
        failed = len(binary_paths) - len(succeeded)
        if failed > 0:
            logger.warning("Ghidra batch: %d/%d 二进制分析失败", failed, len(binary_paths))
        return succeeded
    result = []
    for idx, bin_path in enumerate(binary_paths, start=1):
        _, json_path = _run_one((idx, bin_path))
        if json_path:
            result.append(json_path)
    failed = len(binary_paths) - len(result)
    if failed > 0:
        logger.warning("Ghidra batch: %d/%d 二进制分析失败", failed, len(binary_paths))
    return result
