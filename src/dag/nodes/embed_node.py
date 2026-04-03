"""嵌入节点：将特征转为向量（占位，模型留空）。"""

from typing import Any, Dict

from ..model import DAGNode


class EmbedNode(DAGNode):
    """从 ctx 中的 features 生成 embeddings。当前为占位，返回空/恒等。"""

    NODE_TYPE = "embed"

    def execute(self, ctx: Dict[str, Any]) -> None:
        input_key = self.params.get("input_key", "features")
        output_key = self.params.get("output_key", "embeddings")

        feats = ctx.get(input_key)
        if feats is None:
            raise KeyError(f"ctx[{input_key}] not found")

        try:
            from features.inference import embed_batch

            embeddings = embed_batch(feats)
        except ImportError:
            # 无 features 模块时，占位
            funcs = feats.get("functions", [])
            embeddings = [{"name": f.get("name", ""), "vector": []} for f in funcs]
        except NotImplementedError:
            funcs = feats.get("functions", [])
            embeddings = [{"name": f.get("name", ""), "vector": []} for f in funcs]

        result = {"functions": embeddings}
        self.output = result
        ctx[output_key] = result
        self.done = True
