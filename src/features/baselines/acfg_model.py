"""ACFG 基线适配器：extract_acfg_features + 哈希投影到 128 维。"""

from typing import Any, Dict, List, Optional

from features.baselines.base import BaseSimilarityModel


class ACFGModel(BaseSimilarityModel):
    """
    ACFG (Attributed Control Flow Graph) 基线。

    对 extract_acfg_features 提取的 node_features 做哈希投影到 128 维向量。
    不需要预训练模型，仅依赖 CFG 结构与基本块属性。
    """

    _HASH_DIM = 128

    @property
    def name(self) -> str:
        return "acfg"

    @property
    def output_dim(self) -> int:
        return self._HASH_DIM

    def embed_batch(
        self,
        features: Dict[str, Any],
        *,
        model_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        from utils.feature_extractors.graph_features import extract_acfg_features

        results: List[Dict[str, Any]] = []
        for func_id, lsir_func in features.items():
            acfg = extract_acfg_features(lsir_func)
            vec = self._hash_project(acfg)
            results.append({"name": func_id, "vector": vec})
        return results

    def _hash_project(self, acfg: Dict[str, Any]) -> List[float]:
        """将 ACFG 特征通过哈希投影到 _HASH_DIM 维向量。"""
        vec = [0.0] * self._HASH_DIM

        # 结构特征
        n_nodes = acfg.get("num_nodes", 0)
        n_edges = acfg.get("num_edges", 0)
        if n_nodes > 0:
            vec[0] = min(n_nodes / 1024.0, 1.0)
        if n_edges > 0:
            vec[1] = min(n_edges / 1024.0, 1.0)

        # 基本块 opcode 哈希投影
        for nf in acfg.get("node_features", []):
            for opcode in nf.get("pcode_opcodes", []):
                h = hash(opcode) % self._HASH_DIM
                vec[h] += 1.0

        # L2 归一化
        norm_sq = sum(x * x for x in vec)
        if norm_sq > 0:
            norm = norm_sq ** 0.5
            vec = [x / norm for x in vec]

        return vec
