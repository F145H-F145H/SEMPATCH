"""
简化日志：兼容 ghidra_runner 所需接口。
"""

import logging
import sys
from typing import Any, Dict, Optional

try:
    from config import LOG_LEVEL
except Exception:
    LOG_LEVEL = "INFO"


class _SimpleLogger:
    def __init__(self, name: str, level: Optional[str] = None):
        self._log = logging.getLogger(name)
        self._log.setLevel(getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO))
        if not self._log.handlers:
            h = logging.StreamHandler(sys.stderr)
            h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            self._log.addHandler(h)
            self._log.propagate = False

    def progress(self, msg: str, *args, **kwargs) -> None:
        self._log.info(msg, *args, **kwargs)

    def success(self, msg: str, *args, **kwargs) -> None:
        self._log.info(msg, *args, **kwargs)

    def fail(self, msg: str, *args, **kwargs) -> None:
        self._log.error(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._log.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._log.debug(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._log.error(msg, *args, **kwargs)

    def structured(self, msg: str, **kwargs) -> None:
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self._log.info("%s %s", msg, extra or "")

    def exception(self, msg: str, *args, exc_info=True, **kwargs) -> None:
        self._log.exception(msg, *args, exc_info=exc_info, **kwargs)


def get_logger(
    name: str = None,
    level: Optional[str] = None,
    format_type: Optional[str] = None,
    **kwargs,
) -> _SimpleLogger:
    if name is None:
        import inspect
        frame = inspect.currentframe()
        caller = frame.f_back if frame else None
        name = caller.f_globals.get("__name__", "root") if caller else "root"
    return _SimpleLogger(name, level)
