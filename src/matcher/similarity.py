"""相似度计算函数。"""

from typing import List, Sequence, Union


def cosine_similarity(a: List[Union[int, float]], b: List[Union[int, float]]) -> float:
    """余弦相似度。向量为空或长度不一致时返回 0。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def cosine_similarity_batch(
    queries: Sequence[Sequence[float]],
    db_vectors: Sequence[Sequence[float]],
) -> "list":
    """批量余弦相似度（numpy 向量化）。返回 shape=(len(queries), len(db_vectors)) 的矩阵。"""
    import numpy as np

    q = np.array(queries, dtype=np.float32)
    d = np.array(db_vectors, dtype=np.float32)
    q_norm = q / np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-10)
    d_norm = d / np.linalg.norm(d, axis=1, keepdims=True).clip(min=1e-10)
    return (q_norm @ d_norm.T).tolist()


def euclidean_distance(a: List[Union[int, float]], b: List[Union[int, float]]) -> float:
    """欧氏距离。"""
    if not a or not b or len(a) != len(b):
        return float("inf")
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def l2_normalize_single(vec: List[Union[int, float]]) -> List[float]:
    """单向量 L2 归一化。零向量返回原向量（避免除零）。"""
    if not vec:
        return []
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0:
        return [float(x) for x in vec]
    return [float(x) / norm for x in vec]


def l2_normalize(vectors: List[List[Union[int, float]]]) -> List[List[float]]:
    """批量 L2 归一化。FAISS IndexFlatIP 用内积，归一化后内积等于余弦相似度。"""
    return [l2_normalize_single(v) for v in vectors]
