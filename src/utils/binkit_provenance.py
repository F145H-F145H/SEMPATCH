"""
BinKit / 固件二进制路径启发式同源解析（project_id + 变体 hints）。

不依赖外部元数据表：从相对路径与 basename 抽取 arch / compiler / opt，并给出弱同源键。
无法识别变体时 project_id 退化为「去扩展名的 basename」。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# 路径片段 → 规范 arch 标签（小写）
_ARCH_PATH_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("x86_64", "x86_64"),
    ("x86-64", "x86_64"),
    ("amd64", "x86_64"),
    ("i386", "x86"),
    ("i686", "x86"),
    ("aarch64", "aarch64"),
    ("arm64", "aarch64"),
    ("armhf", "arm"),
    ("armeb", "arm"),
    ("mipsel", "mipsel"),
    ("mips64el", "mips64el"),
    ("mips", "mips"),
    ("riscv64", "riscv64"),
    ("riscv32", "riscv32"),
)

# basename 中的编译器 / 优化标记
_CC_RE = re.compile(r"(?P<cc>gcc|clang)(?:[\W_]|$)", re.IGNORECASE)
_OPT_RE = re.compile(r"(?:^|[._-])(?P<opt>O[0-3]|Os|Ofast)(?:[._-]|$)", re.IGNORECASE)

# 从 project_id 尾部剥编译产物后缀：_O2_gcc / -O3-clang / _Os 等
_PROJECT_STRIP_RE = re.compile(
    r"([._-](?:O[0-3]|Os|Ofast))(?:[._-](?:gcc|clang))?$|([._-](?:gcc|clang))$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VariantHints:
    """从路径启发式得到的变体信息；空字符串表示未知。"""

    arch: str = ""
    compiler: str = ""
    opt: str = ""

    def fingerprint(self) -> str:
        """用于判断两二进制是否「同工具链变体」；全空时返回空串。"""
        parts = [self.arch, self.compiler, self.opt]
        if not any(parts):
            return ""
        return "|".join(parts)


def _basename_stem(binary_rel: str) -> str:
    base = os.path.basename((binary_rel or "").replace("\\", "/"))
    if "." in base:
        return base.rsplit(".", 1)[0]
    return base or "unknown"


def _detect_arch_from_path(norm_path: str) -> str:
    lower = norm_path.lower()
    for needle, canonical in _ARCH_PATH_ALIASES:
        if needle in lower:
            return canonical
    return ""


def _detect_compiler(basename_stem: str) -> str:
    m = _CC_RE.search(basename_stem)
    if m:
        return m.group("cc").lower()
    return ""


def _detect_opt(basename_stem: str) -> str:
    m = _OPT_RE.search(basename_stem)
    if not m:
        return ""
    return (m.group("opt") or "").lower()


def derive_project_id(binary_rel: str) -> str:
    """
    同源弱键：basename 去掉扩展名后再去掉尾部 _O? / _gcc / _clang 等片段。
    """
    stem = _basename_stem(binary_rel)
    s = stem
    prev = None
    while prev != s:
        prev = s
        s = _PROJECT_STRIP_RE.sub("", s)
    return s.strip("._-") or stem


def parse_binary_provenance(binary_rel: str) -> Tuple[str, VariantHints]:
    """
    解析单条索引中的 binary 相对路径。

    Returns:
        (project_id, variant_hints)
    """
    norm = (binary_rel or "").replace("\\", "/")
    stem = _basename_stem(norm)
    arch = _detect_arch_from_path(norm)
    compiler = _detect_compiler(stem)
    opt = _detect_opt(stem)
    hints = VariantHints(arch=arch, compiler=compiler, opt=opt)
    return derive_project_id(norm), hints


def classify_pair_relation(h1: VariantHints, h2: VariantHints) -> str:
    """
    用于 pair_mix：cross_arch | same_arch_cross_compiler | same_arch_same_toolchain | unknown
    """
    a1, a2 = h1.arch, h2.arch
    if a1 and a2 and a1 != a2:
        return "cross_arch"
    if not a1 or not a2:
        return "unknown"
    c1, c2 = h1.compiler, h2.compiler
    if c1 and c2 and c1 != c2:
        return "same_arch_cross_compiler"
    return "same_arch_same_toolchain"


def load_provenance_yaml(path: str) -> Dict[str, Any]:
    """
    可选：从 YAML 加载扩展规则（未来使用）。当前若文件不存在或 yaml 不可用则返回空 dict。
    """
    p = os.path.abspath(path)
    if not os.path.isfile(p):
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return dict(data) if isinstance(data, dict) else {}


def summarize_provenance(index_binaries: list) -> Dict[str, int]:
    """统计 project_id 数量、含 arch 的条目数等（供脚本日志）。"""
    from collections import Counter

    pid_c: Counter[str] = Counter()
    with_arch = 0
    for rel in index_binaries:
        _pid, h = parse_binary_provenance(rel if isinstance(rel, str) else "")
        pid_c[_pid] += 1
        if h.arch:
            with_arch += 1
    return {
        "unique_project_id": len(pid_c),
        "binaries_with_arch_hint": with_arch,
        "total_binaries": len(index_binaries),
    }
