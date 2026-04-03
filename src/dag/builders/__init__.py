"""DAG 构建器包。封装 add_node，按节点类型传入 params、deps。"""

from .fusion import (
    build_diff_bipartite_node,
    build_diff_faiss_node,
    build_diff_node,
    build_embed_node,
    build_feature_extract_node,
    build_load_db_node,
    build_lsir_build_node,
    build_pcode_normalize_node,
)
from .ghidra import build_ghidra_node
from .traditional import (
    build_acfg_extract_node,
    build_cfg_match_node,
    build_diff_fuzzy_node,
    build_fuzzy_hash_node,
)
from .unpack import build_unpack_node

__all__ = [
    "build_acfg_extract_node",
    "build_cfg_match_node",
    "build_diff_bipartite_node",
    "build_diff_faiss_node",
    "build_diff_fuzzy_node",
    "build_diff_node",
    "build_embed_node",
    "build_feature_extract_node",
    "build_fuzzy_hash_node",
    "build_ghidra_node",
    "build_load_db_node",
    "build_lsir_build_node",
    "build_pcode_normalize_node",
    "build_unpack_node",
]
