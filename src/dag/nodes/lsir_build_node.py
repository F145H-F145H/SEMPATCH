"""LSIR 构建节点：从 ghidra_output 构建 LSIR。"""

from typing import Any, Dict

from ..model import DAGNode


class LSIRBuildNode(DAGNode):
    """从 ctx 中的 ghidra_output 构建 LSIR（CFG/DFG）。"""

    NODE_TYPE = "lsir_build"

    def execute(self, ctx: Dict[str, Any]) -> None:
        from utils.ir_builder import build_lsir
        from utils.pcode_normalizer import normalize_lsir_raw

        input_key = self.params.get("input_key", "ghidra_output")
        output_key = self.params.get("output_key", "lsir")
        normalize_pcode = self.params.get("normalize_pcode", True)
        include_cfg = self.params.get("include_cfg", True)
        include_dfg = self.params.get("include_dfg", True)

        raw = ctx.get(input_key)
        if raw is None:
            raise KeyError(f"ctx[{input_key}] not found, cannot build LSIR")

        if normalize_pcode and isinstance(raw, dict) and "functions" in raw:
            raw = normalize_lsir_raw(raw, abstract_unique=True, in_place=False)

        lsir = build_lsir(raw, include_cfg=include_cfg, include_dfg=include_dfg)
        self.output = lsir
        ctx[output_key] = lsir
        self.done = True
