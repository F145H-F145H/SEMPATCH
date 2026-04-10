"""
DAG 节点数据契约：InputSpec / OutputSpec 定义。
供文档、测试与 ctx 键断言使用。
"""

from typing import Any, Dict, List, Optional, TypedDict

# ---------------------------------------------------------------------------
# LSIR / Ghidra 相关
# ---------------------------------------------------------------------------


class LSIRInstruction(TypedDict, total=False):
    """单条指令。"""

    address: str
    mnemonic: str
    operands: str
    pcode: List[Dict[str, Any]]


class LSIRBasicBlock(TypedDict, total=False):
    """基本块。"""

    start: str
    instructions: List[LSIRInstruction]


class LSIRFunction(TypedDict, total=False):
    """LSIR 函数。经 build_lsir 后应含 dfg（networkx 或 dict；可为空图）。"""

    name: str
    entry: str
    basic_blocks: List[LSIRBasicBlock]
    cfg: Any  # nx.DiGraph or dict
    dfg: Any  # nx.DiGraph or dict；LSIR 构建路径下始终存在


class LSIRRaw(TypedDict, total=False):
    """Ghidra 原始输出。"""

    functions: List[LSIRFunction]


class LSIR(TypedDict, total=False):
    """含 cfg/dfg 的中间表示。"""

    functions: List[LSIRFunction]


# ---------------------------------------------------------------------------
# 特征与嵌入
# ---------------------------------------------------------------------------


class FeaturesItem(TypedDict, total=False):
    """单个函数的特征。"""

    name: str
    features: Dict[str, Any]


class FeaturesDict(TypedDict, total=False):
    """特征字典。"""

    functions: List[FeaturesItem]


class EmbeddingItem(TypedDict, total=False):
    """单个函数的嵌入。"""

    name: str
    vector: List[float]
    cve: Optional[str]


class EmbeddingDict(TypedDict, total=False):
    """嵌入字典。"""

    functions: List[EmbeddingItem]


# ---------------------------------------------------------------------------
# 模糊哈希
# ---------------------------------------------------------------------------


class FuzzyHashItem(TypedDict, total=False):
    """单个函数的模糊哈希。"""

    name: str
    hash: str
    algorithm: str  # "ssdeep" | "tlsh"


class FuzzyHashDict(TypedDict, total=False):
    """模糊哈希字典。"""

    functions: List[FuzzyHashItem]


# ---------------------------------------------------------------------------
# CFG 签名
# ---------------------------------------------------------------------------


class CFGSigItem(TypedDict, total=False):
    """单个函数的 CFG 签名。"""

    name: str
    num_nodes: int
    num_edges: int
    edges: List[tuple]
    node_list: List[str]


class CFGSigDict(TypedDict, total=False):
    """CFG 签名字典。"""

    functions: List[CFGSigItem]


# ---------------------------------------------------------------------------
# 匹配结果
# ---------------------------------------------------------------------------


class DiffMatchItem(TypedDict, total=False):
    """单条匹配记录。"""

    firmware_func: str
    db_func: str
    similarity: float
    method: str
    mcs_ratio: Optional[float]


class DiffResult(TypedDict, total=False):
    """差分/匹配结果。"""

    matches: List[DiffMatchItem]


# ---------------------------------------------------------------------------
# 节点输入输出键映射（便于 assert）
# ---------------------------------------------------------------------------

NODE_INPUT_KEYS: Dict[str, List[str]] = {
    "ghidra": [],
    "pcode_normalize": ["ghidra_output"],
    "lsir_build": ["ghidra_output"],
    "feature_extract": ["lsir"],
    "embed": ["features"],
    "load_db": [],
    "diff": ["embeddings", "db_embeddings"],
    "unpack": [],
    "fuzzy_hash": ["lsir"],
    "cfg_match": ["lsir", "db_lsir"],
    "acfg_extract": ["lsir"],
    "diff_faiss": ["embeddings", "db_embeddings"],
    "diff_bipartite": ["embeddings", "db_embeddings"],
    "diff_fuzzy": ["fuzzy_hashes", "db_fuzzy_hashes"],
}

NODE_OUTPUT_KEYS: Dict[str, str] = {
    "ghidra": "ghidra_output",
    "pcode_normalize": "ghidra_output",
    "lsir_build": "lsir",
    "feature_extract": "features",
    "embed": "embeddings",
    "load_db": "db_embeddings",
    "diff": "diff_result",
    "unpack": "unpack_dir",
    "fuzzy_hash": "fuzzy_hashes",
    "cfg_match": "diff_result",
    "acfg_extract": "acfg_features",
    "diff_faiss": "diff_result",
    "diff_bipartite": "diff_result",
    "diff_fuzzy": "diff_result",
}


def assert_ctx_keys(ctx: Dict[str, Any], keys: List[str], prefix: str = "") -> None:
    """断言 ctx 中必须存在的键。"""
    for k in keys:
        if k not in ctx:
            raise KeyError(f"{prefix}ctx[{k}] required but not found")
        if ctx[k] is None:
            raise ValueError(f"{prefix}ctx[{k}] must not be None")


def validate_specs_registry_consistency() -> None:
    """校验 NODE_INPUT_KEYS / NODE_OUTPUT_KEYS 与 NODE_TYPE_REGISTRY 一致。"""
    try:
        from .nodes import NODE_TYPE_REGISTRY
    except ImportError:
        return
    spec_keys = set(NODE_INPUT_KEYS) | set(NODE_OUTPUT_KEYS)
    reg_keys = set(NODE_TYPE_REGISTRY)
    orphaned_specs = spec_keys - reg_keys
    missing_specs = reg_keys - spec_keys
    if orphaned_specs:
        import logging

        logging.getLogger(__name__).warning(
            "specs 中存在 registry 未注册的节点: %s", orphaned_specs
        )
    if missing_specs:
        import logging

        logging.getLogger(__name__).warning("registry 中存在 specs 未定义的节点: %s", missing_specs)
