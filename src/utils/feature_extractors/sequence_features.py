"""P-code 序列、跳转编码等序列特征。"""

from typing import Any, Dict, List


def extract_sequence_features(lsir_func: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 LSIR 函数提取序列特征（P-code opcode 序列、跳转编码等）。
    返回可序列化结构。
    """
    out: Dict[str, Any] = {"pcode_seq": [], "mnemonic_seq": [], "jump_encoding": []}

    bbs = lsir_func.get("basic_blocks", []) or []
    for bb in bbs:
        for inst in bb.get("instructions", []) or []:
            mnemonic = inst.get("mnemonic") or ""
            out["mnemonic_seq"].append(mnemonic)
            for pco in inst.get("pcode", []) or []:
                opcode = pco.get("opcode") or ""
                out["pcode_seq"].append(opcode)

    # 跳转编码：根据 mnemonic 是否为跳转类
    jump_mnemonics = frozenset(
        {"BRANCH", "CBRANCH", "CALL", "BRANCHIND", "CALLIND", "RETURN"}
    )
    for m in out["mnemonic_seq"]:
        out["jump_encoding"].append(1 if m.upper() in jump_mnemonics else 0)

    out["seq_len"] = len(out["pcode_seq"])
    return out
