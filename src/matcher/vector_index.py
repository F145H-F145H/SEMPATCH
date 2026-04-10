"""FAISS 向量索引封装。"""

from typing import Any, List, Optional, Sequence, Tuple

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None  # type: ignore


class VectorIndex:
    """向量索引，支持批量插入与近似最近邻搜索。"""

    def __init__(self, dim: int, index_type: str = "flat"):
        self.dim = dim
        self.index_type = index_type
        self._index: Any = None
        self._ids: List[str] = []
        if HAS_FAISS and faiss:
            if index_type == "flat":
                self._index = faiss.IndexFlatIP(dim)  # inner product
            else:
                self._index = faiss.IndexFlatL2(dim)
        else:
            self._data: List[List[float]] = []

    def add(self, vectors: List[List[float]], ids: Optional[List[str]] = None) -> None:
        """批量插入向量。"""
        if not vectors:
            return
        if HAS_FAISS and self._index is not None:
            import numpy as np

            m = np.array(vectors, dtype="float32")
            self._index.add(m)
            self._ids.extend(ids or [str(i) for i in range(len(vectors))])
        else:
            self._data.extend(vectors)

    def search(self, query: List[float], k: int = 10) -> List[tuple]:
        """搜索最近邻，返回 (id, score) 列表。"""
        if HAS_FAISS and self._index is not None:
            import numpy as np

            q = np.array([query], dtype="float32")
            scores, indices = self._index.search(q, min(k, self._index.ntotal))
            result = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self._ids):
                    result.append((self._ids[idx], float(scores[0][i])))
            return result
        # 无 FAISS：线性扫描降级
        if hasattr(self, "_data") and self._data:
            from .similarity import cosine_similarity

            scored = [(i, cosine_similarity(query, v)) for i, v in enumerate(self._data)]
            scored.sort(key=lambda x: -x[1])
            result = []
            for idx, score in scored[:k]:
                if idx < len(self._ids):
                    result.append((self._ids[idx], score))
                else:
                    result.append((str(idx), score))
            return result
        return []

    def search_many(
        self, queries: Sequence[Sequence[float]], k: int = 10
    ) -> List[List[Tuple[str, float]]]:
        """
        批量搜索最近邻。

        返回与 queries 等长的列表，每项为 [(id, score), ...]。
        """
        if not queries:
            return []
        if HAS_FAISS and self._index is not None:
            import numpy as np

            q = np.array(queries, dtype="float32")
            k_actual = min(k, self._index.ntotal)
            scores, indices = self._index.search(q, k_actual)
            out: List[List[Tuple[str, float]]] = []
            for row_scores, row_indices in zip(scores, indices):
                row: List[Tuple[str, float]] = []
                for s, idx in zip(row_scores, row_indices):
                    if idx >= 0 and idx < len(self._ids):
                        row.append((self._ids[idx], float(s)))
                out.append(row)
            return out

        # 无 FAISS：回退为逐条 search（保持接口一致）
        return [self.search(list(q), k=k) for q in queries]


def search_neighbors(
    query_vec: List[float],
    db_vectors: List[List[float]],
    k: int = 10,
) -> List[tuple]:
    """简单线性搜索最近邻（无 FAISS 时）。"""
    if not query_vec or not db_vectors:
        return []
    try:
        from .similarity import cosine_similarity

        scores = [cosine_similarity(query_vec, v) for v in db_vectors]
        paired = [(i, s) for i, s in enumerate(scores)]
        paired.sort(key=lambda x: -x[1])
        return [(str(i), s) for i, s in paired[:k]]
    except ImportError:
        return []
