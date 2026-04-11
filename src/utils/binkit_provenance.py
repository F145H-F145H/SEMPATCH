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
    ("arm_64", "aarch64"),
    ("arm_32", "arm"),
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
_OPT_RE = re.compile(r"(?:^|[._\-])(?P<opt>O[0-3]|Os|Ofast)(?:[._\-]|$)", re.IGNORECASE)

# BinKit 命名格式：{project}_{compiler-ver}_{arch}_{opt}_{binary}
# 例：coreutils-9.1_gcc-10.3.0_x86_64_Ofast_fmt
# project 可能含 -X.Y 版本号（如 gawk-5.2.1），compiler-ver 为 gcc-N.N.N 或 clang-N.N
# 用 compiler-ver 的起始位置做切分：匹配 _gcc-\d 或 _clang-\d 后面跟 .或数字
_BINKIT_PROJECT_RE = re.compile(
    r"^(?P<project>.+?)_(?:gcc|clang)-\d[\d.]*(?:_\w+)*(?:_\w+)?(?:\.\S+)?$",
    re.IGNORECASE,
)

# 从 project_id 尾部剥编译产物后缀：_O2_gcc / -O3-clang / _Os 等
# 同时处理 _gcc-10.3.0 / _clang-8.0 等带版本号的编译器标记
_PROJECT_STRIP_RE = re.compile(
    r"([._-](?:O[0-3]|Os|Ofast))(?:[._-](?:gcc|clang)(?:-[\d.]+)?)?$"
    r"|(?:[._-](?:gcc|clang)(?:-[\d.]+)?)$",
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

    BinKit 特殊处理：文件名格式 {project}_{compiler-ver}_{arch}_{opt}_{binary}，
    先用 _BINKIT_PROJECT_RE 提取 project 段，避免把 compiler-version 吃进 project_id。
    例：coreutils-9.1_gcc-10.3.0_x86_64_Ofast_fmt → coreutils-9.1
    """
    stem = _basename_stem(binary_rel)
    # 优先尝试 BinKit 命名格式
    m = _BINKIT_PROJECT_RE.match(stem)
    if m:
        return m.group("project").strip("._-") or stem
    # 回退：逐次剥离尾部 compiler/opt 标记
    s = stem
    prev = None
    while prev != s:
        prev = s
        s = _PROJECT_STRIP_RE.sub("", s)
    return s.strip("._-") or stem


def _full_basename(binary_rel: str) -> str:
    """返回完整 basename（不做 .ext 截断），供 BinKit 全名解析用。"""
    base = os.path.basename((binary_rel or "").replace("\\", "/"))
    return base or "unknown"


def parse_binary_provenance(binary_rel: str) -> Tuple[str, VariantHints]:
    """
    解析单条索引中的 binary 相对路径。

    BinKit 格式 {project}_{compiler-ver}_{arch}_{opt}_{binary} 中 basename 含多个 .，
    _basename_stem 的 rsplit 会截断版本号（如 clang-8.0 → clang-8）。
    因此 arch/opt/compiler 检测使用 _full_basename，project_id 使用 _basename_stem + BinKit 专用正则。

    Returns:
        (project_id, variant_hints)
    """
    norm = (binary_rel or "").replace("\\", "/")
    full_base = _full_basename(norm)
    stem = _basename_stem(norm)
    # arch 从路径段检测（不受 basename 截断影响）
    arch = _detect_arch_from_path(norm)
    # compiler/opt 从完整 basename 检测，避免版本号 . 截断
    compiler = _detect_compiler(full_base)
    opt = _detect_opt(full_base)
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