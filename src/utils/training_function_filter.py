"""
训练数据用的函数名过滤：排除 main、CRT/启动胶水符号等。

用于 filter_index_by_pcode_len、build_binkit_index（可选）等，在昂贵特征提取前丢弃无关符号。
"""

from __future__ import annotations

import os
from typing import AbstractSet, Iterable, Optional, Sequence, Set


def strip_linker_suffix(name: str) -> str:
    """去掉 GNU 链接器版本后缀，如 foo@@GLIBC_2.34 -> foo。"""
    if "@@" in name:
        return name.split("@@", 1)[0]
    return name


# 保守默认：精确匹配（规范化后），避免宽泛前缀误杀业务符号
_DEFAULT_RUNTIME_EXACT: frozenset[str] = frozenset(
    {
        "main",
        "_start",
        "__libc_start_main",
        "__libc_csu_init",
        "__libc_csu_fini",
        "__gmon_start__",
        "_init",
        "_fini",
        "frame_dummy",
        "register_tm_clones",
        "__do_global_dtors_aux",
        "__do_global_ctors_aux",
        "call_weak_fn",
        "_init_term",
        "__libc_init_first",
        "__libc_setup_tls",
        "__pthread_initialize_minimal",
    }
)


def load_exclude_names_from_file(path: str) -> frozenset[str]:
    """
    每行一个符号；空行与 # 开头行忽略；strip 后入库。
    """
    out: Set[str] = set()
    with open(os.path.abspath(path), encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.add(s)
    return frozenset(out)


_LIBC_BUNDLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libc_common_exact.txt")


class TrainingSymbolFilter:
    """
    可配置的训练符号排除器。

    exclude_runtime=False 时不应用内置 CRT 列表，仍应用 extra_exact / file / prefixes。
    include_libc_common=True 时合并 bundled libc_common_exact.txt（printf/memcpy 等），降低库符号正对噪声。
    """

    def __init__(
        self,
        *,
        exclude_runtime: bool = True,
        extra_exact: Optional[AbstractSet[str]] = None,
        extra_prefixes: Optional[Sequence[str]] = None,
        names_from_file: Optional[str] = None,
        include_libc_common: bool = False,
    ) -> None:
        self._exclude_runtime = bool(exclude_runtime)
        exact: Set[str] = set(extra_exact or ())
        if names_from_file:
            exact |= set(load_exclude_names_from_file(names_from_file))
        if include_libc_common and os.path.isfile(_LIBC_BUNDLE_PATH):
            exact |= set(load_exclude_names_from_file(_LIBC_BUNDLE_PATH))
        self._extra_exact = frozenset(exact)
        prefs = tuple(p.strip() for p in (extra_prefixes or ()) if p and p.strip())
        self._extra_prefixes = prefs
        if self._exclude_runtime:
            self._exact = _DEFAULT_RUNTIME_EXACT | self._extra_exact
        else:
            self._exact = frozenset(self._extra_exact)

    @property
    def exact_names(self) -> frozenset[str]:
        return self._exact

    @property
    def extra_prefixes(self) -> tuple[str, ...]:
        return self._extra_prefixes

    def is_excluded(self, raw_name: str) -> bool:
        name = strip_linker_suffix((raw_name or "").strip())
        if not name:
            return False
        if name in self._exact:
            return True
        for p in self._extra_prefixes:
            if name.startswith(p):
                return True
        return False


_default_filter = TrainingSymbolFilter(exclude_runtime=True)


def is_excluded_training_symbol(name: str) -> bool:
    """使用默认内置规则（含 CRT 精确表，无额外前缀/文件）。"""
    return _default_filter.is_excluded(name)
