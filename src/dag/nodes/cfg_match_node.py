"""CFG 结构匹配节点：DiscovRE 风格 MCS 启发式。"""

from typing import Any, Dict, List, Tuple

from ..model import DAGNode
from ..specs import assert_ctx_keys

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore


def _cfg_to_graph(cfg: Any) -> Any:
    """从 lsir 的 cfg 字段转为 nx.DiGraph（若可用）。"""
    if cfg is None:
        return None
    if nx and isinstance(cfg, nx.DiGraph):
        return cfg
    if isinstance(cfg, dict):
        edges = cfg.get("edges", [])
        if not edges:
            return None
        if not nx:
            return None
        # NetworkX 3.x：空图的 __bool__ 为 False，不能用 `if g:` 判断能否建图
        g = nx.DiGraph()
        g.add_edges_from(edges)
        return g
    return None


def _structural_similarity(g1: Any, g2: Any) -> float:
    """
    计算两 CFG 的结构相似度。
    使用边集 Jaccard 的规范化形式，节点映射为拓扑序。
    """
    if not nx or g1 is None or g2 is None:
        return 0.0
    n1, e1 = g1.number_of_nodes(), g1.number_of_edges()
    n2, e2 = g2.number_of_nodes(), g2.number_of_edges()
    if n1 == 0 and n2 == 0:
        return 1.0
    if n1 == 0 or n2 == 0:
        return 0.0

    # 规范化：用 (in_deg, out_deg) 作为节点签名，建立对应关系
    def degree_signature(gr: Any) -> List[Tuple[int, int]]:
        return sorted((gr.in_degree(n), gr.out_degree(n)) for n in gr.nodes())

    sig1 = degree_signature(g1)
    sig2 = degree_signature(g2)

    # 度序列相似度：逐对比较（较短序列 pad 0）
    len_max = max(len(sig1), len(sig2))
    s1 = sig1 + [(0, 0)] * (len_max - len(sig1))
    s2 = sig2 + [(0, 0)] * (len_max - len(sig2))
    match = sum(1 for a, b in zip(s1, s2) if a == b)
    deg_sim = match / len_max if len_max else 1.0

    # 规模相似度
    size_sim = min(n1, n2, e1, e2) / max(n1, n2, 1) * 0.5 + min(e1, e2) / max(e1, e2, 1) * 0.5
    if e1 == 0 and e2 == 0:
        size_sim = 1.0

    # mcs_ratio 近似：度结构相似度与规模相似度的加权
    mcs_ratio = 0.6 * deg_sim + 0.4 * min(1.0, size_sim)
    return mcs_ratio


class CFGMatchNode(DAGNode):
    """固件 lsir vs 漏洞库 db_lsir：CFG 结构匹配，输出 mcs_ratio。"""

    NODE_TYPE = "cfg_match"

    def execute(self, ctx: Dict[str, Any]) -> None:
        lsir_key = self.params.get("lsir_key", "lsir")
        db_lsir_key = self.params.get("db_lsir_key", "db_lsir")
        output_key = self.params.get("output_key", "diff_result")
        threshold = float(self.params.get("threshold", 0.0))

        assert_ctx_keys(ctx, [lsir_key, db_lsir_key], "CFGMatchNode: ")

        lsir = ctx[lsir_key]
        db_lsir = ctx[db_lsir_key]
        fw_funcs = lsir.get("functions", [])
        db_funcs = db_lsir.get("functions", [])

        matches: List[Dict[str, Any]] = []
        for fw in fw_funcs:
            fw_cfg = _cfg_to_graph(fw.get("cfg"))
            if fw_cfg is None:
                continue
            for db in db_funcs:
                db_cfg = _cfg_to_graph(db.get("cfg"))
                if db_cfg is None:
                    continue
                mcs_ratio = _structural_similarity(fw_cfg, db_cfg)
                if mcs_ratio >= threshold:
                    matches.append({
                        "firmware_func": fw.get("name", ""),
                        "db_func": db.get("name", ""),
                        "similarity": mcs_ratio,
                        "method": "cfg_mcs",
                        "mcs_ratio": mcs_ratio,
                    })

        result = {"matches": matches}
        self.output = result
        ctx[output_key] = result
        self.done = True
