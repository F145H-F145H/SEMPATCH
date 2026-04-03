"""测试 matcher.similarity。"""
from matcher.similarity import (
    cosine_similarity,
    euclidean_distance,
    l2_normalize,
    l2_normalize_single,
)


def test_cosine_similarity():
    assert abs(cosine_similarity([1, 0], [1, 0]) - 1.0) < 1e-6
    assert abs(cosine_similarity([1, 0], [0, 1]) - 0.0) < 1e-6
    assert abs(cosine_similarity([1, 1], [1, 1]) - 1.0) < 1e-6
    assert cosine_similarity([], [1, 2]) == 0.0
    assert cosine_similarity([1], [1, 2]) == 0.0


def test_l2_normalize_single():
    """单向量 L2 归一化，模长变为 1。"""
    v = [3.0, 4.0]
    n = l2_normalize_single(v)
    assert abs(sum(x * x for x in n) ** 0.5 - 1.0) < 1e-6
    v2 = [1.0, 0.0, 0.0]
    n2 = l2_normalize_single(v2)
    assert abs(sum(x * x for x in n2) ** 0.5 - 1.0) < 1e-6
    # 同向向量内积为 1
    dot = sum(a * b for a, b in zip(n2, [2.0, 0.0, 0.0]))
    n2b = l2_normalize_single([2.0, 0.0, 0.0])
    assert abs(sum(a * b for a, b in zip(n2, n2b)) - 1.0) < 1e-6


def test_l2_normalize():
    """批量 L2 归一化。"""
    vectors = [[3.0, 4.0], [1.0, 0.0]]
    out = l2_normalize(vectors)
    assert len(out) == 2
    for v in out:
        assert abs(sum(x * x for x in v) ** 0.5 - 1.0) < 1e-6


def test_l2_normalize_zero_vector():
    """零向量不除零，返回原向量。"""
    v = [0.0, 0.0]
    n = l2_normalize_single(v)
    assert n == [0.0, 0.0]
