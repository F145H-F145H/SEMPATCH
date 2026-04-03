"""DAG 节点子类与类型注册表。扩展时：新增类、实现 execute、在 NODE_TYPE_REGISTRY 注册。"""

from typing import Any, Dict, Type

from ..model import DAGNode

from .acfg_extract_node import ACFGExtractNode
from .cfg_match_node import CFGMatchNode
from .diff_bipartite_node import DiffBipartiteNode
from .diff_faiss_node import DiffFAISSNode
from .diff_fuzzy_node import DiffFuzzyNode
from .diff_node import DiffNode
from .embed_node import EmbedNode
from .feature_extract_node import FeatureExtractNode
from .fuzzy_hash_node import FuzzyHashNode
from .load_db_node import LoadDBNode
from .lsir_build_node import LSIRBuildNode
from .pcode_normalize_node import PcodeNormalizeNode
from .unpack_node import UnpackNode


class GhidraNode(DAGNode):
    """运行 Ghidra headless 提取 P-code 到 lsir_raw.json。"""

    NODE_TYPE = "ghidra"
    retriable = True

    def execute(self, ctx: Dict[str, Any]) -> None:
        from utils.ghidra_runner import run_ghidra_analysis

        p = self.params
        binary_path = p["binary_path"]
        output_dir = p["output_dir"]
        force = p.get("force", False)
        timeout = p.get("timeout")
        project_name = p.get("project_name", "SemPatchProject")

        result = run_ghidra_analysis(
            binary_path=binary_path,
            output_dir=output_dir,
            project_name=project_name,
            timeout=timeout,
            force=force,
            return_dict=True,
        )
        self.output = result
        ctx["ghidra_output"] = result
        self.done = True


NODE_TYPE_REGISTRY: Dict[str, Type[DAGNode]] = {
    "ghidra": GhidraNode,
    "pcode_normalize": PcodeNormalizeNode,
    "lsir_build": LSIRBuildNode,
    "feature_extract": FeatureExtractNode,
    "embed": EmbedNode,
    "load_db": LoadDBNode,
    "diff": DiffNode,
    "unpack": UnpackNode,
    "fuzzy_hash": FuzzyHashNode,
    "cfg_match": CFGMatchNode,
    "acfg_extract": ACFGExtractNode,
    "diff_faiss": DiffFAISSNode,
    "diff_bipartite": DiffBipartiteNode,
    "diff_fuzzy": DiffFuzzyNode,
}
