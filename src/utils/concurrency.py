"""
全局并发控制：用于 Ghidra 批量并行。
"""

import multiprocessing
import os
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")

_global_semaphore: Optional[multiprocessing.Semaphore] = None

try:
    from utils.config import PARALLEL_WORKERS
except Exception:
    PARALLEL_WORKERS = 8


def get_parallel_workers() -> int:
    return max(0, min(64, int(PARALLEL_WORKERS)))


def get_global_semaphore() -> multiprocessing.Semaphore:
    global _global_semaphore
    if _global_semaphore is None:
        n = max(1, min(64, get_parallel_workers()))
        _global_semaphore = multiprocessing.Semaphore(n)
    return _global_semaphore


def bounded_task(
    sem: multiprocessing.Semaphore, fn: Callable[..., T], *args: Any, **kwargs: Any
) -> T:
    sem.acquire()
    try:
        return fn(*args, **kwargs)
    finally:
        sem.release()


def bounded_task_slots(
    sem: multiprocessing.Semaphore,
    slots: int,
    fn: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """占用 slots 个许可执行 fn，执行完毕后释放。用于 DAG 节点并发控制。"""
    for _ in range(slots):
        sem.acquire()
    try:
        return fn(*args, **kwargs)
    finally:
        for _ in range(slots):
            sem.release()
