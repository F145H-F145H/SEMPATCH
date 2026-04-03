"""
SIGINT 时立即回收资源：注册活跃子进程，触发时终止它们。
供 ghidra_runner 等注册 Popen，auto_optimize 的 SIGINT handler 调用 trigger_shutdown。
"""

import threading
from typing import Set

_active_processes: Set = set()
_lock = threading.Lock()


def register_process(process) -> None:
    """注册需在 SIGINT 时终止的子进程。"""
    with _lock:
        _active_processes.add(process)


def unregister_process(process) -> None:
    """子进程结束后注销。"""
    with _lock:
        _active_processes.discard(process)


def trigger_shutdown() -> None:
    """SIGINT 时调用：终止所有已注册子进程，并清理注册表。"""
    with _lock:
        to_kill = list(_active_processes)
        _active_processes.clear()
    for p in to_kill:
        try:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=2)
            if p.poll() is None:
                p.kill()
                p.wait(timeout=1)
        except Exception:
            pass
