"""二分图最大权匹配（匈牙利/Kuhn-Munkres）。"""

from typing import List, Tuple


def kuhn_munkres(cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """
    二分图最大权匹配。cost_matrix[i][j] 为左侧 i 与右侧 j 的权重。
    返回 [(左侧索引, 右侧索引), ...]。
    """
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        r_ind, c_ind = linear_sum_assignment(-np.array(cost_matrix))
        return list(zip(r_ind.tolist(), c_ind.tolist()))
    except ImportError:
        # 无 scipy 时返回空
        return []
