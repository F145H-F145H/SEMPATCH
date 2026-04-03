"""
P-code 规范化（survey 5.3）：统一 varnode 表示、opcode 别名、抑制编译器优化噪声。
输入：lsir_raw 结构；输出：规范化后的 lsir_raw（原地修改或返回新结构）。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Opcode 别名映射：不同 P-code 方言的等价 op 统一
# ---------------------------------------------------------------------------
OPCODE_ALIASES: Dict[str, str] = {
    "INT_ADD": "INT_ADD",
    "INT_SUB": "INT_SUB",
    "INT_AND": "INT_AND",
    "INT_OR": "INT_OR",
    "INT_XOR": "INT_XOR",
    "INT_MULT": "INT_MULT",
    "INT_DIV": "INT_DIV",
    "INT_SDIV": "INT_SDIV",
    "INT_REM": "INT_REM",
    "INT_SREM": "INT_SREM",
    "INT_NEGATE": "INT_NEGATE",
    "INT_LEFT": "INT_LEFT",
    "INT_RIGHT": "INT_RIGHT",
    "INT_SRIGHT": "INT_SRIGHT",
    "INT_EQUAL": "INT_EQUAL",
    "INT_NOTEQUAL": "INT_NOTEQUAL",
    "INT_LESS": "INT_LESS",
    "INT_SLESS": "INT_SLESS",
    "INT_LESSEQUAL": "INT_LESSEQUAL",
    "INT_SLESSEQUAL": "INT_SLESSEQUAL",
    "INT_CARRY": "INT_CARRY",
    "INT_SCARRY": "INT_SCARRY",
    "INT_SBORROW": "INT_SBORROW",
    "INT_ZEXT": "INT_ZEXT",
    "INT_SEXT": "INT_SEXT",
    "INT_2COMP": "INT_2COMP",
    "POPCOUNT": "POPCOUNT",
    "COPY": "COPY",
    "LOAD": "LOAD",
    "STORE": "STORE",
    "BRANCH": "BRANCH",
    "CBRANCH": "CBRANCH",
    "BRANCHIND": "BRANCHIND",
    "CALL": "CALL",
    "CALLIND": "CALLIND",
    "RETURN": "RETURN",
}

# Varnode 解析：支持 "(space, offset, size)" 格式
_VARNODE_RE = re.compile(r"^\s*\(\s*(\w+)\s*,\s*(0x[0-9a-fA-F]+|\d+)\s*,\s*(\d+)\s*\)\s*$")


def _parse_varnode(s: str) -> Optional[Tuple[str, int, int]]:
    """解析 varnode 字符串 -> (space, offset, size)。"""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    m = _VARNODE_RE.match(s)
    if not m:
        return None
    space, offset_str, size_str = m.groups()
    try:
        offset = int(offset_str, 16) if offset_str.startswith("0x") else int(offset_str)
        size = int(size_str)
        return (space.lower(), offset, size)
    except ValueError:
        return None


def _format_varnode(space: str, offset: int, size: int) -> str:
    """格式化 varnode 为规范字符串。"""
    return f"({space},{hex(offset)},{size})"


def normalize_varnode(v: str, *, abstract_unique: bool = True) -> str:
    """
    规范化单个 varnode 字符串。
    - abstract_unique: 将 unique 空间抽象为 "unique:size"，消除临时变量偏移差异
    - register: 保留 space:size:offset（跨架构时 offset 仍可区分）
    - const: 保留
    - ram: 抽象为 ram:size 以减弱绝对地址影响（可选，当前保留）
    """
    parsed = _parse_varnode(v)
    if parsed is None:
        return v
    space, offset, size = parsed

    if space == "unique" and abstract_unique:
        return f"(unique,0x0,{size})"  # 抽象临时变量，消除不同编译的偏移差异

    return _format_varnode(space, offset, size)


def normalize_opcode(op: str) -> str:
    """规范化 opcode，使用别名映射。"""
    op = (op or "").strip().upper()
    return OPCODE_ALIASES.get(op, op)


def normalize_pcode_op(pco: Dict[str, Any], *, abstract_unique: bool = True) -> Dict[str, Any]:
    """规范化单条 P-code 操作。"""
    opcode = normalize_opcode(pco.get("opcode") or "")
    out_raw = pco.get("output")
    in_raw = pco.get("inputs") or []

    out_norm = None
    if out_raw is not None:
        out_norm = normalize_varnode(str(out_raw), abstract_unique=abstract_unique)

    in_norm = [normalize_varnode(str(x), abstract_unique=abstract_unique) for x in in_raw]

    return {
        "opcode": opcode,
        "output": out_norm,
        "inputs": in_norm,
    }


def normalize_instruction(inst: Dict[str, Any], *, abstract_unique: bool = True) -> Dict[str, Any]:
    """规范化单条指令的 pcode 字段。"""
    out = dict(inst)
    pcode = out.get("pcode") or []
    if not pcode:
        return out
    out["pcode"] = [normalize_pcode_op(p, abstract_unique=abstract_unique) for p in pcode]
    return out


def normalize_lsir_raw(
    lsir_raw: Dict[str, Any],
    *,
    abstract_unique: bool = True,
    in_place: bool = False,
) -> Dict[str, Any]:
    """
    规范化 lsir_raw 中所有 P-code。
    输入/输出格式与原始 lsir_raw 兼容，供 ir_builder 直接使用。

    Args:
        lsir_raw: Ghidra 输出的 lsir_raw 结构
        abstract_unique: 是否抽象 unique 空间（抑制编译器临时变量差异）
        in_place: 是否原地修改；否则返回新 dict

    Returns:
        规范化后的 lsir_raw
    """
    if not in_place:
        import copy

        lsir_raw = copy.deepcopy(lsir_raw)

    funcs = lsir_raw.get("functions")
    if not isinstance(funcs, list):
        return lsir_raw

    for f in funcs:
        if not isinstance(f, dict):
            continue
        bbs = f.get("basic_blocks") or []
        for bb in bbs:
            if not isinstance(bb, dict):
                continue
            insts = bb.get("instructions") or []
            for i, inst in enumerate(insts):
                if isinstance(inst, dict):
                    bb["instructions"][i] = normalize_instruction(
                        inst, abstract_unique=abstract_unique
                    )

    return lsir_raw
