"""两阶段流水线：从库嵌入构建 FAISS 索引，支持 Top-K 粗筛检索。"""

import json
from typing import List, Optional, Sequence, Tuple

from .similarity import l2_normalize, l2_normalize_single
from .vector_index import VectorIndex

_SAFE_DIM = 128


def retrieve_coarse(
    query_features: dict,
    library_faiss_index: "LibraryFaissIndex",
    k: int,
    safe_model_path: Optional[str] = None,
) -> List[str]:
    """
    粗筛检索：输入查询多模态特征，返回 Top-K 候选 function_id 列表。

    query_features: 单函数 multimodal 特征 {graph, sequence} 或 FeaturesDict 兼容格式
    library_faiss_index: B.3 构建的 LibraryFaissIndex
    k: 返回的候选数量
    safe_model_path: 训练后的 SAFE 权重路径，指定时与库嵌入使用同一模型
    """
    from features.baselines.safe import embed_batch_safe

    # 包装为 FeaturesDict 兼容格式
    if "functions" in query_features:
        feats = query_features
    else:
        feats = {
            "functions": [
                {"name": "query", "features": {"multimodal": query_features}}
            ]
        }
    result = embed_batch_safe(feats, model_path=safe_model_path)
    if not result:
        return []
    vec = result[0]["vector"]
    q_norm = l2_normalize_single(vec)
    if not q_norm:
        return []
    search_results = library_faiss_index.search(q_norm, k=k)
    return [fid for fid, _ in search_results]

def retrieve_coarse_many(
    query_multimodals: Sequence[dict],
    library_faiss_index: "LibraryFaissIndex",
    k: int,
    *,
    safe_embedder: "object",
) -> List[List[str]]:
    """
    批量粗筛：输入多个 query 的 multimodal 特征，返回每个 query 的 Top-K 候选 id 列表。

    safe_embedder 需要提供 embed_many(multimodals, batch_size=...) -> List[List[float]]
    """
    # 允许 safe_embedder 自己决定 batch_size；这里仅做最薄封装
    vecs = safe_embedder.embed_many(list(query_multimodals))
    if not vecs:
        return [[] for _ in query_multimodals]
    q_norms = [l2_normalize_single(v) for v in vecs]
    results = library_faiss_index.search_many(q_norms, k=k)
    return [[fid for fid, _ in row] for row in results]


class LibraryFaissIndex:
    """从库嵌入 JSON 构建的 FAISS 索引，支持 search(query_vector, k)。"""

    def __init__(self, embeddings_path: str) -> None:
        """加载 embeddings JSON，L2 归一化后构建 FAISS IndexFlatIP。"""
        with open(embeddings_path, encoding="utf-8") as f:
            data = json.load(f)
        funcs = data.get("functions", [])
        if not funcs:
            self._index: VectorIndex | None = VectorIndex(_SAFE_DIM, index_type="flat")
            self._ids: List[str] = []
            return
        vectors = [f["vector"] for f in funcs]
        ids = [f.get("function_id", f.get("name", str(i))) for i, f in enumerate(funcs)]
        norm_vectors = l2_normalize(vectors)
        self._index = VectorIndex(_SAFE_DIM, index_type="flat")
        self._index.add(norm_vectors, ids=ids)
        self._ids = ids

    def search(
        self, query_vector: List[float], k: int = 10
    ) -> List[Tuple[str, float]]:
        """对查询向量 L2 归一化后检索，返回 Top-K 的 (function_id, score)。"""
        if not self._ids:
            return []
        q_norm = l2_normalize_single(query_vector)
        if not q_norm:
            return []
        k_actual = min(k, len(self._ids))
        return self._index.search(q_norm, k=k_actual)

    def search_many(
        self, query_vectors: Sequence[Sequence[float]], k: int = 10
    ) -> List[List[Tuple[str, float]]]:
        """批量检索：对每个 query 向量返回 Top-K 的 (function_id, score)。"""
        if not self._ids:
            return [[] for _ in query_vectors]
        k_actual = min(k, len(self._ids))
        # 输入向量已在上层做过 normalize；这里保持兼容再 normalize 一次也安全
        q_norms = [l2_normalize_single(list(v)) for v in query_vectors]
        return self._index.search_many(q_norms, k=k_actual)
