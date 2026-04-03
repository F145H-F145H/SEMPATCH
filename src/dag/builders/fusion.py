"""Fusion/ semantic 流水线节点构建器。"""

from typing import Any, Dict, List, Optional

from ..model import JobDAG


def build_pcode_normalize_node(
    dag: JobDAG,
    node_id: str,
    deps: List[str],
    *,
    input_key: str = "ghidra_output",
    output_key: str = "ghidra_output",
    abstract_unique: bool = True,
    priority: int = 0,
) -> None:
    """添加 P-code 规范化节点（5.3）。"""
    params: Dict[str, Any] = {
        "input_key": input_key,
        "output_key": output_key,
        "abstract_unique": abstract_unique,
    }
    dag.add_node(
        node_id=node_id,
        node_type="pcode_normalize",
        params=params,
        deps=deps,
        priority=priority,
    )


def build_lsir_build_node(
    dag: JobDAG,
    node_id: str,
    deps: List[str],
    *,
    input_key: str = "ghidra_output",
    output_key: str = "lsir",
    normalize_pcode: bool = True,
    include_cfg: bool = True,
    include_dfg: bool = True,
    priority: int = 0,
) -> None:
    """添加 LSIR 构建节点。normalize_pcode=True 时在构建前执行 P-code 规范化（5.3）。dfg 始终写入函数 dict（include_dfg=False 时为空图）。"""
    params: Dict[str, Any] = {
        "input_key": input_key,
        "output_key": output_key,
        "normalize_pcode": normalize_pcode,
        "include_cfg": include_cfg,
        "include_dfg": include_dfg,
    }
    dag.add_node(
        node_id=node_id,
        node_type="lsir_build",
        params=params,
        deps=deps,
        priority=priority,
    )


def build_feature_extract_node(
    dag: JobDAG,
    node_id: str,
    deps: List[str],
    *,
    input_key: str = "lsir",
    output_key: str = "features",
    priority: int = 0,
) -> None:
    """添加特征提取节点。"""
    params: Dict[str, Any] = {"input_key": input_key, "output_key": output_key}
    dag.add_node(
        node_id=node_id,
        node_type="feature_extract",
        params=params,
        deps=deps,
        priority=priority,
    )


def build_embed_node(
    dag: JobDAG,
    node_id: str,
    deps: List[str],
    *,
    input_key: str = "features",
    output_key: str = "embeddings",
    priority: int = 0,
) -> None:
    """添加嵌入节点。"""
    params: Dict[str, Any] = {"input_key": input_key, "output_key": output_key}
    dag.add_node(
        node_id=node_id,
        node_type="embed",
        params=params,
        deps=deps,
        priority=priority,
    )


def build_load_db_node(
    dag: JobDAG,
    node_id: str,
    db_path: str,
    deps: Optional[List[str]] = None,
    *,
    db_format: str = "embeddings",
    output_key: Optional[str] = None,
    priority: int = 0,
) -> None:
    """添加漏洞库加载节点。db_format: embeddings | lsir | fuzzy_hashes。"""
    _defaults = {"embeddings": "db_embeddings", "lsir": "db_lsir", "fuzzy_hashes": "db_fuzzy_hashes"}
    out_key = output_key or _defaults.get(db_format, "db_embeddings")
    params: Dict[str, Any] = {
        "db_path": db_path,
        "db_format": db_format,
        "output_key": out_key,
    }
    dag.add_node(
        node_id=node_id,
        node_type="load_db",
        params=params,
        deps=deps or [],
        priority=priority,
    )


def build_diff_node(
    dag: JobDAG,
    node_id: str,
    deps: List[str],
    *,
    firmware_embeddings_key: str = "embeddings",
    db_embeddings_key: str = "db_embeddings",
    output_key: str = "diff_result",
    priority: int = 0,
) -> None:
    """添加差分/匹配节点。"""
    params: Dict[str, Any] = {
        "firmware_embeddings_key": firmware_embeddings_key,
        "db_embeddings_key": db_embeddings_key,
        "output_key": output_key,
    }
    dag.add_node(
        node_id=node_id,
        node_type="diff",
        params=params,
        deps=deps,
        priority=priority,
    )


def build_diff_faiss_node(
    dag: JobDAG,
    node_id: str,
    deps: List[str],
    *,
    firmware_embeddings_key: str = "embeddings",
    db_embeddings_key: str = "db_embeddings",
    output_key: str = "diff_result",
    k: int = 10,
    index_type: str = "flat",
    priority: int = 0,
) -> None:
    """添加 FAISS k-NN 匹配节点。"""
    params: Dict[str, Any] = {
        "firmware_embeddings_key": firmware_embeddings_key,
        "db_embeddings_key": db_embeddings_key,
        "output_key": output_key,
        "k": k,
        "index_type": index_type,
    }
    dag.add_node(
        node_id=node_id,
        node_type="diff_faiss",
        params=params,
        deps=deps,
        priority=priority,
    )


def build_diff_bipartite_node(
    dag: JobDAG,
    node_id: str,
    deps: List[str],
    *,
    firmware_embeddings_key: str = "embeddings",
    db_embeddings_key: str = "db_embeddings",
    output_key: str = "diff_result",
    similarity_metric: str = "cosine",
    priority: int = 0,
) -> None:
    """添加二分图匹配节点。"""
    params: Dict[str, Any] = {
        "firmware_embeddings_key": firmware_embeddings_key,
        "db_embeddings_key": db_embeddings_key,
        "output_key": output_key,
        "similarity_metric": similarity_metric,
    }
    dag.add_node(
        node_id=node_id,
        node_type="diff_bipartite",
        params=params,
        deps=deps,
        priority=priority,
    )
