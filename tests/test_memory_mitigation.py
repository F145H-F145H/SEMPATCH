"""utils.memory_mitigation 单元测试。"""

import multiprocessing
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.memory_mitigation import (  # noqa: E402
    build_process_pool_executor_kwargs,
    configure_address_space_limit,
    process_pool_executor_supports_max_tasks_per_child,
    resolve_max_memory_mb,
)


def test_resolve_max_memory_mb_cli_over_env(monkeypatch):
    monkeypatch.delenv("SEMPATCH_MAX_MEMORY_MB", raising=False)
    assert resolve_max_memory_mb(1024) == 1024
    monkeypatch.setenv("SEMPATCH_MAX_MEMORY_MB", "2048")
    assert resolve_max_memory_mb(1024) == 1024
    assert resolve_max_memory_mb(None) == 2048


def test_configure_address_space_limit_noop():
    ok, msg = configure_address_space_limit(None)
    assert ok is False
    assert "未启用" in msg


@patch("resource.setrlimit")
def test_configure_address_space_limit_sets_rlimit(mock_setrlimit):
    mock_setrlimit.return_value = None
    ok, msg = configure_address_space_limit(512)
    assert ok is True
    mock_setrlimit.assert_called_once()
    assert "512" in msg


def test_build_process_pool_executor_kwargs_skips_recycle_on_fork():
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:
        pytest.skip("fork not available")
    kwargs = build_process_pool_executor_kwargs(
        max_workers=2,
        mp_context=ctx,
        max_tasks_per_child=7,
    )
    assert "max_tasks_per_child" not in kwargs


def test_build_process_pool_executor_kwargs_recycle_only_with_spawn():
    try:
        ctx = multiprocessing.get_context("spawn")
    except ValueError:
        pytest.skip("spawn not available")
    kwargs = build_process_pool_executor_kwargs(
        max_workers=2,
        mp_context=ctx,
        max_tasks_per_child=3,
    )
    if process_pool_executor_supports_max_tasks_per_child():
        assert kwargs.get("max_tasks_per_child") == 3
    else:
        assert "max_tasks_per_child" not in kwargs
