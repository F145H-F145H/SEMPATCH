"""差分/匹配节点：固件嵌入 vs 漏洞库嵌入。"""

from typing import Any, Dict

from ..model import DAGNode


class DiffNode(DAGNode):
    """固件 vs 漏洞库：检索匹配，输出 diff_result。"""

    NODE_TYPE = "diff"

    def execute(self, ctx: Dict[str, Any]) -> None:
        firmware_key = self.params.get("firmware_embeddings_key", "embeddings")
        db_key = self.params.get("db_embeddings_key", "db_embeddings")
        output_key = self.params.get("output_key", "diff_result")
        threshold = float(self.params.get("threshold", 0.0))

        fw_emb = ctx.get(firmware_key)
        db_emb = ctx.get(db_key)
        if fw_emb is None:
            raise KeyError(f"ctx[{firmware_key}] not found")
        if db_emb is None:
            raise KeyError(f"ctx[{db_key}] not found")

        fw_funcs = fw_emb.get("functions", [])
        db_funcs = db_emb.get("functions", [])
        matches = []
        try:
            from matcher.similarity import cosine_similarity
        except ImportError:
            cosine_similarity = None

        for fe in fw_funcs:
            vec = fe.get("vector", [])
            if not vec:
                continue
            for de in db_funcs:
                dv = de.get("vector", [])
                if not dv:
                    continue
                if cosine_similarity:
                    sim = cosine_similarity(vec, dv)
                else:
                    sim = 0.0
                if sim < threshold:
                    continue
                matches.append(
                    {
                        "firmware_func": fe.get("name"),
                        "db_func": de.get("name"),
                        "similarity": sim,
                    }
                )
        result = {"matches": matches}
        self.output = result
        ctx[output_key] = result
        self.done = True
