"""
大程序 / 大 lsir_raw 场景下的内存缓解工具。

- 进程级虚拟地址空间上限（Unix RLIMIT_AS，best-effort）
- ProcessPoolExecutor 的 max_tasks_per_child（Python 3.11+，回收子进程释放碎片）
- 可选按二进制 gc.collect()
"""
from __future__ import annotations

import gc
import inspect
import multiprocessing
import os
import sys
from typing import Any, Dict, Optional, Tuple

# 与 CLI --max-memory-mb 等价；CLI 优先于环境变量
ENV_MAX_MEMORY_MB = "SEMPATCH_MAX_MEMORY_MB"


def resolve_max_memory_mb(cli_value: Optional[int]) -> Optional[int]:
    if cli_value is not None and cli_value > 0:
        return cli_value
    raw = os.environ.get(ENV_MAX_MEMORY_MB, "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
        return v if v > 0 else None
    except ValueError:
        return None


def configure_address_space_limit(max_memory_mb: Optional[int]) -> Tuple[bool, str]:
    """
    设置 RLIMIT_AS（虚拟地址空间软/硬上限，单位字节）。

    仅在类 Unix 上通常可用；macOS/部分环境可能失败或行为不同。
    失败时不抛异常，返回 (False, reason)。
    """
    if max_memory_mb is None or max_memory_mb <= 0:
        return False, "未启用地址空间上限"
    try:
        import resource

        limit = int(max_memory_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        return True, f"RLIMIT_AS 已设为约 {max_memory_mb} MiB（虚拟地址空间上限）"
    except (AttributeError, OSError, ValueError) as e:
        return False, f"无法设置 RLIMIT_AS: {e}"


def process_pool_executor_supports_max_tasks_per_child() -> bool:
    return sys.version_info >= (3, 11)


def build_process_pool_executor_kwargs(
    *,
    max_workers: int,
    mp_context: Optional[multiprocessing.context.BaseContext] = None,
    max_tasks_per_child: int = 0,
) -> Dict[str, Any]:
    """
    构造 ProcessPoolExecutor 关键字参数。

    CPython 3.11+：max_tasks_per_child 与 start method「fork」不兼容，会抛 ValueError。
    SemPatch 过滤脚本依赖 fork（避免 spawn 下对 __main__ 内函数的 pickle 问题），
    故在 fork 上下文中自动忽略 max_tasks_per_child 并打日志。
    """
    import logging

    kwargs: Dict[str, Any] = {"max_workers": max_workers}
    if mp_context is not None:
        kwargs["mp_context"] = mp_context
    if max_tasks_per_child and max_tasks_per_child > 0:
        from concurrent.futures import ProcessPoolExecutor

        sig = inspect.signature(ProcessPoolExecutor)
        if "max_tasks_per_child" not in sig.parameters:
            return kwargs
        start_method = (
            mp_context.get_start_method()
            if mp_context is not None
            else multiprocessing.get_start_method()
        )
        if start_method == "fork":
            logging.getLogger(__name__).warning(
                "max_tasks_per_child=%s 与 multiprocessing start method「fork」不兼容，已忽略；"
                "过滤脚本固定使用 fork。可改用较小 --workers 或 --gc-after-each-binary 缓解内存。",
                max_tasks_per_child,
            )
            return kwargs
        kwargs["max_tasks_per_child"] = max_tasks_per_child
    return kwargs


def maybe_gc_after_binary(enabled: bool) -> None:
    if enabled:
        gc.collect()


def warn_if_large_lsir(
    *,
    binary_label: str,
    num_functions: int,
    warn_functions: int = 20000,
) -> None:
    """单二进制函数数过大时提示（整份 lsir_raw 常驻内存）。"""
    if num_functions >= warn_functions:
        import logging

        logging.getLogger(__name__).warning(
            "二进制 %s 含 %d 个函数，lsir_raw 体积可能极大；若 OOM 请降低并行、"
            "启用 --gc-after-each-binary，或使用 systemd/cgroup 限制内存（见 docs/memory_and_oom.md）",
            binary_label,
            num_functions,
        )
