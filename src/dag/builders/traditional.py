"""Traditional 流水线节点构建器（CFG、模糊哈希等）。"""

from typing import Any, Dict, List

from ..model import JobDAG


def build_fuzzy_hash_node(
    dag: JobDAG,
    node_id: str,
    deps: List[str],
    *,
    input_key: str = "lsir",
    output_key: str = "fuzzy_hashes",
    algorithm: str = "auto",
    priority: int = 0,
) -> None:
    """添加模糊哈希节点。"""
    params: Dict[str, Any] = {
        "input_key": input_key,
        "output_key": output_key,
        "algorithm": algorithm,
    }
    dag.add_node(
        node_id=node_id,
        node_type="fuzzy_hash",
        params=params,
        deps=deps,
        priority=priority,
    )


def build_cfg_match_node(
    dag: JobDAG,
    node_id: str,
    deps: List[str],
    *,
    lsir_key: str = "lsir",
    db_lsir_key: str = "db_lsir",
    output_key: str = "diff_result",
    threshold: float = 0.0,
    priority: int = 0,
) -> None:
    """添加 CFG MCS 匹配节点。"""
    params: Dict[str, Any] = {
        "lsir_key": lsir_key,
        "db_lsir_key": db_lsir_key,
        "output_key": output_key,
        "threshold": threshold,
    }
    dag.add_node(
        node_id=node_id,
        node_type="cfg_match",
        params=params,
        deps=deps,
        priority=priority,
    )


def build_acfg_extract_node(
    dag: JobDAG,
    node_id: str,
    deps: List[str],
    *,
    input_key: str = "lsir",
    output_key: str = "acfg_features",
    priority: int = 0,
) -> None:
    """添加 ACFG 特征提取节点。"""
    params: Dict[str, Any] = {"input_key": input_key, "output_key": output_key}
    dag.add_node(
        node_id=node_id,
        node_type="acfg_extract",
        params=params,
        deps=deps,
        priority=priority,
    )


def build_diff_fuzzy_node(
    dag: JobDAG,
    node_id: str,
    deps: List[str],
    *,
    fuzzy_hashes_key: str = "fuzzy_hashes",
    db_fuzzy_hashes_key: str = "db_fuzzy_hashes",
    output_key: str = "diff_result",
    threshold: float = 0.0,
    priority: int = 0,
) -> None:
    """添加模糊哈希匹配节点。"""
    params: Dict[str, Any] = {
        "fuzzy_hashes_key": fuzzy_hashes_key,
        "db_fuzzy_hashes_key": db_fuzzy_hashes_key,
        "output_key": output_key,
        "threshold": threshold,
    }
    dag.add_node(
        node_id=node_id,
        node_type="diff_fuzzy",
        params=params,
        deps=deps,
        priority=priority,
    )
