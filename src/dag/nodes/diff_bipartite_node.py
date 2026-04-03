"""二分图匹配节点：Kuhn-Munkres 最优一一匹配。"""

from typing import Any, Dict, List

from ..model import DAGNode
from ..specs import assert_ctx_keys


class DiffBipartiteNode(DAGNode):
    """使用 Kuhn-Munkres 做固件与漏洞库函数的最优一一匹配。"""

    NODE_TYPE = "diff_bipartite"

    def execute(self, ctx: Dict[str, Any]) -> None:
        firmware_key = self.params.get("firmware_embeddings_key", "embeddings")
        db_key = self.params.get("db_embeddings_key", "db_embeddings")
        output_key = self.params.get("output_key", "diff_result")
        similarity_metric = self.params.get("similarity_metric", "cosine")

        assert_ctx_keys(ctx, [firmware_key, db_key], "DiffBipartiteNode: ")

        fw_emb = ctx[firmware_key]
        db_emb = ctx[db_key]
        fw_funcs = [f for f in fw_emb.get("functions", []) if f.get("vector")]
        db_funcs = [d for d in db_emb.get("functions", []) if d.get("vector")]

        matches: List[Dict[str, Any]] = []

        if not fw_funcs or not db_funcs:
            result = {"matches": matches}
            self.output = result
            ctx[output_key] = result
            self.done = True
            return

        try:
            from matcher.similarity import cosine_similarity, euclidean_distance
        except ImportError:
            cosine_similarity = lambda a, b: 0.0  # type: ignore
            euclidean_distance = lambda a, b: float("inf")  # type: ignore

        def sim(a: List[float], b: List[float]) -> float:
            if similarity_metric == "euclidean":
                d = euclidean_distance(a, b)
                return 1.0 / (1.0 + d) if d != float("inf") else 0.0
            return cosine_similarity(a, b)

        M = len(fw_funcs)
        N = len(db_funcs)
        cost_matrix: List[List[float]] = []
        for fe in fw_funcs:
            row = [sim(fe.get("vector", []), de.get("vector", [])) for de in db_funcs]
            cost_matrix.append(row)

        from matcher.bipartite_matcher import kuhn_munkres
        pairs = kuhn_munkres(cost_matrix)

        for fi, dj in pairs:
            if fi < M and dj < N:
                sim_val = cost_matrix[fi][dj]
                matches.append({
                    "firmware_func": fw_funcs[fi].get("name", ""),
                    "db_func": db_funcs[dj].get("name", ""),
                    "similarity": float(sim_val),
                    "method": "bipartite",
                })

        result = {"matches": matches}
        self.output = result
        ctx[output_key] = result
        self.done = True
