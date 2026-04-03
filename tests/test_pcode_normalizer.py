"""P-code 规范化单元测试。"""
import pytest

from utils.pcode_normalizer import (
    normalize_lsir_raw,
    normalize_instruction,
    normalize_opcode,
    normalize_pcode_op,
    normalize_varnode,
)


def test_normalize_varnode_unique_abstracted():
    """unique 空间在 abstract_unique=True 时抽象为 (unique,0x0,size)。"""
    assert normalize_varnode("(unique, 0x2c180, 8)", abstract_unique=True) == "(unique,0x0,8)"
    assert normalize_varnode("(unique, 0x100, 4)", abstract_unique=True) == "(unique,0x0,4)"


def test_normalize_varnode_register_preserved():
    """register 空间保留 format。"""
    assert normalize_varnode("(register, 0x20, 8)") == "(register,0x20,8)"
    assert normalize_varnode("(register, 0x0, 8)") == "(register,0x0,8)"


def test_normalize_varnode_const_preserved():
    """const 空间保留 format。"""
    assert normalize_varnode("(const, 0x8, 8)") == "(const,0x8,8)"


def test_normalize_varnode_unique_not_abstracted():
    """abstract_unique=False 时 unique 不抽象。"""
    assert normalize_varnode("(unique, 0x2c180, 8)", abstract_unique=False) == "(unique,0x2c180,8)"


def test_normalize_opcode():
    """opcode 别名映射，大小写统一。"""
    assert normalize_opcode("int_sub") == "INT_SUB"
    assert normalize_opcode("  COPY  ") == "COPY"
    assert normalize_opcode("int_add") == "INT_ADD"
    assert normalize_opcode("CBRANCH") == "CBRANCH"


def test_normalize_pcode_op():
    """单条 P-code 操作 output/inputs 统一格式。"""
    pco = {
        "opcode": "int_sub",
        "output": "(unique, 0x2c180, 8)",
        "inputs": ["(register, 0x20, 8)", "(const, 0x8, 8)"],
    }
    out = normalize_pcode_op(pco)
    assert out["opcode"] == "INT_SUB"
    assert out["output"] == "(unique,0x0,8)"
    assert out["inputs"] == ["(register,0x20,8)", "(const,0x8,8)"]


def test_normalize_instruction():
    """指令的 pcode 列表被规范化。"""
    inst = {
        "address": "0x1000",
        "mnemonic": "SUB",
        "pcode": [
            {"opcode": "COPY", "output": "(unique, 0x100, 8)", "inputs": ["(register, 0x20, 8)"]},
        ],
    }
    out = normalize_instruction(inst)
    assert out["address"] == "0x1000"
    assert len(out["pcode"]) == 1
    assert out["pcode"][0]["output"] == "(unique,0x0,8)"


def test_normalize_instruction_empty_pcode():
    """无 pcode 的指令保持不变。"""
    inst = {"address": "0x1000", "mnemonic": "NOP", "pcode": []}
    out = normalize_instruction(inst)
    assert out["pcode"] == []


def test_normalize_lsir_raw():
    """含 1 函数的 lsir_raw，unique 抽象、opcode 映射正确。"""
    lsir_raw = {
        "functions": [
            {
                "name": "main",
                "entry": "0x1000",
                "basic_blocks": [
                    {
                        "start": "0x1000",
                        "end": "0x1010",
                        "instructions": [
                            {
                                "address": "0x1000",
                                "mnemonic": "SUB",
                                "pcode": [
                                    {
                                        "opcode": "int_sub",
                                        "output": "(unique, 0x2c180, 8)",
                                        "inputs": ["(register, 0x20, 8)", "(const, 0x8, 8)"],
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        ],
    }
    out = normalize_lsir_raw(lsir_raw)
    assert len(out["functions"]) == 1
    fn = out["functions"][0]
    inst = fn["basic_blocks"][0]["instructions"][0]
    assert inst["pcode"][0]["opcode"] == "INT_SUB"
    assert inst["pcode"][0]["output"] == "(unique,0x0,8)"
    assert inst["pcode"][0]["inputs"] == ["(register,0x20,8)", "(const,0x8,8)"]


def test_normalize_lsir_raw_in_place_false_preserves_original():
    """in_place=False 不修改原始 dict。"""
    original = {
        "functions": [
            {
                "basic_blocks": [
                    {
                        "instructions": [
                            {"pcode": [{"opcode": "COPY", "output": "(unique, 0x100, 8)", "inputs": []}]},
                        ],
                    },
                ],
            },
        ],
    }
    orig_output = original["functions"][0]["basic_blocks"][0]["instructions"][0]["pcode"][0]["output"]
    out = normalize_lsir_raw(original, in_place=False)
    assert original["functions"][0]["basic_blocks"][0]["instructions"][0]["pcode"][0]["output"] == orig_output
    assert out["functions"][0]["basic_blocks"][0]["instructions"][0]["pcode"][0]["output"] == "(unique,0x0,8)"


def test_normalize_lsir_raw_empty_function():
    """边界：空函数（无 basic_blocks 或空 basic_blocks）不报错。"""
    lsir_raw = {"functions": [{"name": "empty", "basic_blocks": []}]}
    out = normalize_lsir_raw(lsir_raw)
    assert out["functions"][0]["basic_blocks"] == []


def test_normalize_lsir_raw_empty_blocks():
    """边界：basic_blocks 无 instructions 不报错。"""
    lsir_raw = {
        "functions": [
            {
                "name": "foo",
                "basic_blocks": [{"start": "0x0", "end": "0x10", "instructions": []}],
            },
        ],
    }
    out = normalize_lsir_raw(lsir_raw)
    assert out["functions"][0]["basic_blocks"][0]["instructions"] == []
