"""Ghidra headless 分析辅助函数。仅被 ghidra_runner 调用，不对外导出。"""

import hashlib
import json
import os
import re
import shutil
import threading
import time
from subprocess import PIPE, STDOUT, Popen, TimeoutExpired
from typing import List, Optional, Tuple

from exceptions import SemPatchError
from config import ANALYZE_HEADLESS, BINARY_CACHE_DIR, GHIDRA_HOME, LOG_DIR
from utils.logger import get_logger

logger = get_logger(
    name="SemPatch.GhidraRunner",
    level=None,
    format_type="colored",
)

_LOG_LOCK: Optional[threading.Lock] = None


def _get_log_lock() -> threading.Lock:
    global _LOG_LOCK
    if _LOG_LOCK is None:
        _LOG_LOCK = threading.Lock()
    return _LOG_LOCK


class GhidraEnvironmentError(SemPatchError):
    """Raised when Ghidra environment is invalid."""


def validate_ghidra_environment() -> None:
    """校验 Ghidra 安装与路径。失败时抛出 GhidraEnvironmentError。"""
    if not GHIDRA_HOME or not os.path.isdir(GHIDRA_HOME):
        raise GhidraEnvironmentError(f"Ghidra home not found: {GHIDRA_HOME}")
    if not os.path.isfile(ANALYZE_HEADLESS):
        raise GhidraEnvironmentError(f"analyzeHeadless not found: {ANALYZE_HEADLESS}")
    if not os.access(ANALYZE_HEADLESS, os.X_OK):
        raise GhidraEnvironmentError(f"analyzeHeadless is not executable: {ANALYZE_HEADLESS}")
    logger.progress("Validating Ghidra environment")
    logger.success("Ghidra environment validated")


def can_skip_ghidra(binary_path: str, script_output_path: str) -> bool:
    """若脚本输出已存在且比二进制新，则跳过 Ghidra 分析。"""
    if not os.path.isfile(script_output_path):
        return False
    try:
        if os.path.getsize(script_output_path) == 0:
            return False
        json_mtime = os.path.getmtime(script_output_path)
        bin_mtime = os.path.getmtime(binary_path)
        return json_mtime >= bin_mtime
    except OSError:
        return False


def binary_cache_key(binary_path: str) -> str:
    """根据二进制路径与 mtime/size 生成缓存键。"""
    try:
        st = os.stat(binary_path)
        raw = f"{os.path.abspath(binary_path)}|{st.st_mtime}|{st.st_size}"
    except OSError:
        raw = f"{os.path.abspath(binary_path)}|0|0"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def build_ghidra_command(
    ghidra_project_dir: str,
    project_name: str,
    binary_path: str,
    script_dir: Optional[str],
    script_name: Optional[str],
    script_output_path: str,
) -> List[str]:
    """构造 analyzeHeadless 命令行参数列表。"""
    cmd: List[str] = [
        ANALYZE_HEADLESS,
        ghidra_project_dir,
        project_name,
        "-import",
        binary_path,
        "-overwrite",
        "-analysisTimeoutPerFile",
        "300",
    ]
    if script_dir and script_name:
        cmd.extend(["-scriptPath", script_dir, "-postScript", script_name, script_output_path])
    return cmd


def execute_ghidra_process(
    cmd: List[str],
    binary_path: str,
    timeout: Optional[int],
) -> Tuple[int, str]:
    """
    执行 Ghidra 进程，处理超时与日志写入。
    返回 (return_code, stdout_stderr 合并输出)。
    """
    ghidra_log_path = os.path.join(LOG_DIR, "ghidra.log")
    safe_name = re.sub(r"[^\w\-.]", "_", os.path.basename(binary_path))[:48]
    ts = time.strftime("%Y%m%d_%H%M%S")

    os.makedirs(LOG_DIR, exist_ok=True)

    process = Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1)
    try:
        from utils.shutdown_handler import register_process, unregister_process

        register_process(process)
        try:
            out, _ = process.communicate(timeout=timeout)
        finally:
            unregister_process(process)
    except TimeoutExpired:
        process.kill()
        process.wait()
        logger.error("Ghidra analysis timed out: %s (log: %s)", binary_path, ghidra_log_path)
        raise
    return_code = process.returncode
    out_str = out if out is not None else ""

    with _get_log_lock():
        with open(ghidra_log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] [{safe_name}] # command: {' '.join(cmd)}\n")
            for line in out_str.splitlines(keepends=True):
                f.write(f"[{ts}] [{safe_name}] {line}")
            if out_str and not out_str.endswith("\n"):
                f.write("\n")

    return return_code, out_str


def try_get_binary_cache_dir() -> Optional[str]:
    """安全获取 BINARY_CACHE_DIR，导入异常时返回 None。"""
    try:
        return BINARY_CACHE_DIR
    except (ImportError, AttributeError):
        return None


def write_to_binary_cache(
    script_output_path: str,
    script_output_name: str,
    binary_path: str,
) -> None:
    """若启用缓存且为 lsir_raw.json，则写入二进制缓存。"""
    cache_dir = try_get_binary_cache_dir()
    if not cache_dir or script_output_name != "lsir_raw.json":
        return
    if not os.path.isfile(script_output_path) or os.path.getsize(script_output_path) == 0:
        return
    key = binary_cache_key(binary_path)
    cache_sub = os.path.join(cache_dir, key)
    os.makedirs(cache_sub, exist_ok=True)
    dst = os.path.join(cache_sub, "lsir_raw.json")
    shutil.copy2(script_output_path, dst)
    logger.debug("Cached lsir_raw to %s", dst)


def read_from_binary_cache(
    binary_path: str,
    script_output_path: str,
) -> Optional[str]:
    """
    若缓存中存在有效 lsir_raw.json，复制到 script_output_path 并返回其路径。
    否则返回 None。
    """
    cache_dir = try_get_binary_cache_dir()
    if not cache_dir:
        return None
    key = binary_cache_key(binary_path)
    cached_path = os.path.join(cache_dir, key, "lsir_raw.json")
    if not os.path.isfile(cached_path) or os.path.getsize(cached_path) == 0:
        return None
    shutil.copy2(cached_path, script_output_path)
    logger.info("Reusing binary cache (Ghidra skip): %s -> %s", cached_path, script_output_path)
    return script_output_path


def peek_binary_cache(binary_path: str) -> Optional[dict]:
    """
    若 binary_cache 中存在有效 lsir_raw.json，直接读取并返回解析后的 dict。

    不执行任何文件复制，不创建临时目录，仅做存在性检查与 JSON 解析。
    BINARY_CACHE_DIR 未配置或缓存不存在时返回 None。
    复用 binary_cache_key 与 try_get_binary_cache_dir，与 read_from_binary_cache 键逻辑一致。
    """
    cache_dir = try_get_binary_cache_dir()
    if not cache_dir:
        return None
    key = binary_cache_key(binary_path)
    cached_path = os.path.join(cache_dir, key, "lsir_raw.json")
    if not os.path.isfile(cached_path) or os.path.getsize(cached_path) == 0:
        return None
    try:
        with open(cached_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
