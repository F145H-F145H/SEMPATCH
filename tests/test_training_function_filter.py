"""training_function_filter 单元测试。"""

from __future__ import annotations

import os
import tempfile

import pytest

from utils.training_function_filter import (
    TrainingSymbolFilter,
    is_excluded_training_symbol,
    load_exclude_names_from_file,
    strip_linker_suffix,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("foo", "foo"),
        ("foo@@GLIBC_2.34", "foo"),
        ("__libc_start_main@@GLIBC_2.2.5", "__libc_start_main"),
    ],
)
def test_strip_linker_suffix(raw: str, expected: str) -> None:
    assert strip_linker_suffix(raw) == expected


@pytest.mark.parametrize(
    "name,excluded",
    [
        ("main", True),
        ("MAIN", False),  # 大小写敏感，与 Ghidra 导出一致
        ("__libc_start_main", True),
        ("__libc_start_main@@GLIBC_2.2.5", True),
        ("not_main", False),
        ("my_init_helper", False),
        ("frame_dummy", True),
        ("register_tm_clones", True),
    ],
)
def test_default_is_excluded_training_symbol(name: str, excluded: bool) -> None:
    assert is_excluded_training_symbol(name) is excluded


def test_exclude_runtime_off_keeps_main() -> None:
    f = TrainingSymbolFilter(exclude_runtime=False)
    assert not f.is_excluded("main")
    assert not f.is_excluded("__libc_start_main")


def test_extra_exact() -> None:
    f = TrainingSymbolFilter(exclude_runtime=False, extra_exact={"foo", "bar"})
    assert f.is_excluded("foo")
    assert not f.is_excluded("main")


def test_extra_prefix() -> None:
    f = TrainingSymbolFilter(
        exclude_runtime=False,
        extra_prefixes=("__wrap_",),
    )
    assert f.is_excluded("__wrap_malloc")
    assert not f.is_excluded("malloc")


def test_include_libc_common_excludes_printf() -> None:
    flt = TrainingSymbolFilter(exclude_runtime=False, include_libc_common=True)
    assert flt.is_excluded("printf")
    assert flt.is_excluded("memcpy")
    assert not flt.is_excluded("my_printf_helper")


def test_names_from_file() -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        tf.write("# comment\n")
        tf.write("custom_sym\n")
        tf.write("\n")
        tf.write("another\n")
        path = tf.name
    try:
        names = load_exclude_names_from_file(path)
        assert names == frozenset({"custom_sym", "another"})
        flt = TrainingSymbolFilter(exclude_runtime=False, names_from_file=path)
        assert flt.is_excluded("custom_sym")
        assert not flt.is_excluded("main")
    finally:
        os.unlink(path)
