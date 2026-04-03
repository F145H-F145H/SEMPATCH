"""ACFG 特征提取节点：Genius/Gemini 风格属性控制流图。"""

from typing import Any, Dict

from ..model import DAGNode
from ..specs import assert_ctx_keys


class ACFGExtractNode(DAGNode):
    """从 LSIR 提取 ACFG 特征（含基本块级属性）。"""

    NODE_TYPE = "acfg_extract"

    def execute(self, ctx: Dict[str, Any]) -> None:
        from utils.feature_extractors import extract_acfg_features

        input_key = self.params.get("input_key", "lsir")
        output_key = self.params.get("output_key", "acfg_features")

        assert_ctx_keys(ctx, [input_key], "ACFGExtractNode: ")

        lsir = ctx[input_key]
        funcs = lsir.get("functions", [])
        results = []
        for fn in funcs:
            if not isinstance(fn, dict):
                continue
            acfg = extract_acfg_features(fn)
            results.append({"name": fn.get("name", ""), "acfg": acfg})

        result = {"functions": results}
        self.output = result
        ctx[output_key] = result
        self.done = True
