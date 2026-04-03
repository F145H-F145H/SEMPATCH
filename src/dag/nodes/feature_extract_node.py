"""特征提取节点：从 LSIR 提取图/序列特征。"""

from typing import Any, Dict

from ..model import DAGNode


class FeatureExtractNode(DAGNode):
    """从 ctx 中的 lsir 提取图特征与序列特征并融合。"""

    NODE_TYPE = "feature_extract"

    def execute(self, ctx: Dict[str, Any]) -> None:
        from utils.feature_extractors import (
            extract_acfg_features,
            extract_graph_features,
            extract_sequence_features,
            fuse_features,
        )

        input_key = self.params.get("input_key", "lsir")
        output_key = self.params.get("output_key", "features")

        lsir = ctx.get(input_key)
        if lsir is None:
            raise KeyError(f"ctx[{input_key}] not found")

        funcs = lsir.get("functions", [])
        features_list = []
        for fn in funcs:
            gf = extract_graph_features(fn)
            sf = extract_sequence_features(fn)
            acfg = extract_acfg_features(fn)
            fused = fuse_features(gf, sf, acfg_feats=acfg, include_dfg=True)
            features_list.append({"name": fn.get("name"), "features": fused})
        result = {"functions": features_list}
        self.output = result
        ctx[output_key] = result
        self.done = True
