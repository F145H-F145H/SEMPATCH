"""模糊哈希节点：从 LSIR 按函数生成 ssdeep/tlsh 哈希。Costin et al. 2014。"""

from typing import Any, Dict, List

from ..model import DAGNode
from ..specs import assert_ctx_keys


def _serialize_function_for_hash(func: Dict[str, Any]) -> bytes:
    """将函数指令序列序列化为字节，供模糊哈希使用。"""
    parts: List[str] = []
    for bb in func.get("basic_blocks", []) or []:
        for inst in bb.get("instructions", []) or []:
            mnemonic = (inst.get("mnemonic") or "").strip()
            parts.append(mnemonic)
            for pco in inst.get("pcode", []) or []:
                opcode = (pco.get("opcode") or "").strip()
                parts.append(opcode)
    text = " ".join(parts)
    return text.encode("utf-8", errors="replace")


def _compute_fuzzy_hash(data: bytes, algorithm: str) -> str:
    """计算模糊哈希，优先 ssdeep，不足时用 tlsh。"""
    if algorithm in ("ssdeep", "auto"):
        try:
            import ssdeep
            h = ssdeep.hash(data)
            if h and h != "3::":
                return h
        except ImportError:
            pass
        except Exception:
            pass

    try:
        import tlsh
        if len(data) >= 50:
            h = tlsh.hash(data)
            if h:
                return h
    except ImportError:
        pass
    except Exception:
        pass

    return ""


class FuzzyHashNode(DAGNode):
    """从 LSIR 为每个函数生成模糊哈希。复用 ssdeep/py-tlsh。"""

    NODE_TYPE = "fuzzy_hash"

    def execute(self, ctx: Dict[str, Any]) -> None:
        input_key = self.params.get("input_key", "lsir")
        output_key = self.params.get("output_key", "fuzzy_hashes")
        algorithm = self.params.get("algorithm", "ssdeep")  # ssdeep | tlsh | auto

        assert_ctx_keys(ctx, [input_key], "FuzzyHashNode: ")

        lsir = ctx[input_key]
        funcs = lsir.get("functions", [])
        results: List[Dict[str, Any]] = []

        for fn in funcs:
            if not isinstance(fn, dict):
                continue
            name = fn.get("name", "")
            data = _serialize_function_for_hash(fn)
            if not data:
                results.append({"name": name, "hash": "", "algorithm": algorithm})
                continue

            use_algo = algorithm
            if algorithm == "auto":
                use_algo = "ssdeep" if len(data) >= 4096 else "tlsh"

            h = _compute_fuzzy_hash(data, use_algo)
            if not h and use_algo == "ssdeep":
                h = _compute_fuzzy_hash(data, "tlsh")
                use_algo = "tlsh"

            results.append({"name": name, "hash": h, "algorithm": use_algo})

        result = {"functions": results}
        self.output = result
        ctx[output_key] = result
        self.done = True
