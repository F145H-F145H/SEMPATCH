"""
全局并发控制：用于 Ghidra 批量并行。
"""

import logging
import os
import threading
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)

_global_semaphore: Optional[threading.Semaphore] = None

try:
    from utils.config import PARALLEL_WORKERS
except Exception:
    PARALLEL_WORKERS = 4


def get_parallel_workers() -> int:
    return max(0, min(64, int(PARALLEL_WORKERS)))


def get_global_semaphore() -> threading.Semaphore:
    global _global_semaphore
    if _global_semaphore is None:
        n = max(1, min(64, get_parallel_workers()))
        _global_semaphore = threading.Semaphore(n)
    return _global_semaphore


def bounded_task(sem: threading.Semaphore, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    sem.acquire()
    try:
        return fn(*args, **kwargs)
    finally:
        sem.release()


def bounded_task_slots(
    sem: threading.Semaphore,
    slots: int,
    fn: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """占用 slots 个许可执行 fn，执行完毕后释放。用于 DAG 节点并发控制。

    Raises:
        ValueError: slots 超过信号量初始容量，会产生死锁。
    """
    if slots <= 0:
        raise ValueError(f"slots must be positive, got {slots}")
    # threading.Semaphore 没有公开的初始值属性，用 _initial_value (CPython) 或 _value 兜底
    capacity = getattr(sem, "_initial_value", getattr(sem, "_value", slots))
    if slots > capacity:
        raise ValueError(f"slots ({slots}) exceeds semaphore capacity ({capacity}), would deadlock")
    acquired = 0
    try:
        for _ in range(slots):
            sem.acquire(timeout=300.0)
            acquired += 1
        return fn(*args, **kwargs)
    except Exception:
        # 释放已获取的信号量，避免泄漏
        for _ in range(acquired):
            sem.release()
        raise
    finally:
        if acquired == slots:
            for _ in range(slots):
                sem.release()
