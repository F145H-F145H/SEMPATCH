"""P-code 规范化节点（survey 5.3）：在 LSIR 构建前规范化 lsir_raw 中的 P-code。"""

from typing import Any, Dict

from ..model import DAGNode


class PcodeNormalizeNode(DAGNode):
    """对 ctx 中的 ghidra_output（lsir_raw）执行 P-code 规范化。"""

    NODE_TYPE = "pcode_normalize"

    def execute(self, ctx: Dict[str, Any]) -> None:
        from utils.pcode_normalizer import normalize_lsir_raw

        input_key = self.params.get("input_key", "ghidra_output")
        output_key = self.params.get("output_key", "ghidra_output")  # 默认原地覆盖
        abstract_unique = self.params.get("abstract_unique", True)

        raw = ctx.get(input_key)
        if raw is None:
            raise KeyError(f"ctx[{input_key}] not found, cannot normalize P-code")

        normalized = normalize_lsir_raw(raw, abstract_unique=abstract_unique, in_place=False)
        ctx[output_key] = normalized
        self.output = normalized
        self.done = True
