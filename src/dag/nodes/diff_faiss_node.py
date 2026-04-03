"""FAISS k-NN 匹配节点：固件嵌入 vs 漏洞库，最近邻搜索。"""

from typing import Any, Dict, List

from ..model import DAGNode
from ..specs import assert_ctx_keys


def _l2_normalize(vec: List[float]) -> List[float]:
    """L2 归一化，使内积等于余弦相似度。"""
    import math
    n = math.sqrt(sum(x * x for x in vec))
    if n == 0:
        return vec
    return [x / n for x in vec]


class DiffFAISSNode(DAGNode):
    """使用 VectorIndex（FAISS）做 k-NN 匹配。无 FAISS 时回退到 search_neighbors。"""

    NODE_TYPE = "diff_faiss"

    def execute(self, ctx: Dict[str, Any]) -> None:
        firmware_key = self.params.get("firmware_embeddings_key", "embeddings")
        db_key = self.params.get("db_embeddings_key", "db_embeddings")
        output_key = self.params.get("output_key", "diff_result")
        k = int(self.params.get("k", 10))
        index_type = self.params.get("index_type", "flat")

        assert_ctx_keys(ctx, [firmware_key, db_key], "DiffFAISSNode: ")

        fw_emb = ctx[firmware_key]
        db_emb = ctx[db_key]
        fw_funcs = fw_emb.get("functions", [])
        db_funcs = db_emb.get("functions", [])

        db_vectors: List[List[float]] = []
        db_ids: List[str] = []
        for d in db_funcs:
            v = d.get("vector", [])
            if v:
                db_vectors.append(_l2_normalize(v))
                db_ids.append(d.get("name", ""))

        matches: List[Dict[str, Any]] = []
        if not db_vectors:
            result = {"matches": matches}
            self.output = result
            ctx[output_key] = result
            self.done = True
            return

        dim = len(db_vectors[0])
        try:
            from matcher.vector_index import VectorIndex, HAS_FAISS, search_neighbors
        except ImportError:
            HAS_FAISS = False
            try:
                from matcher.vector_index import search_neighbors
            except ImportError:
                search_neighbors = lambda q, v, k: []  # type: ignore

        if HAS_FAISS:
            try:
                idx = VectorIndex(dim=dim, index_type=index_type)
                idx.add(db_vectors, db_ids)
                for fe in fw_funcs:
                    vec = fe.get("vector", [])
                    if not vec:
                        continue
                    q = _l2_normalize(vec)
                    hits = idx.search(q, k=k)
                    for db_id, score in hits:
                        matches.append({
                            "firmware_func": fe.get("name", ""),
                            "db_func": db_id,
                            "similarity": float(score),
                            "method": "faiss_knn",
                        })
            except Exception:
                HAS_FAISS = False

        if not HAS_FAISS:
            for fe in fw_funcs:
                vec = fe.get("vector", [])
                if not vec:
                    continue
                q = _l2_normalize(vec)
                hits = search_neighbors(q, db_vectors, k=k)
                for db_idx, score in hits:
                    try:
                        i = int(db_idx)
                        db_id = db_ids[i] if 0 <= i < len(db_ids) else ""
                    except (ValueError, KeyError):
                        db_id = ""
                    matches.append({
                        "firmware_func": fe.get("name", ""),
                        "db_func": db_id,
                        "similarity": float(score),
                        "method": "faiss_knn",
                    })

        result = {"matches": matches}
        self.output = result
        ctx[output_key] = result
        self.done = True
