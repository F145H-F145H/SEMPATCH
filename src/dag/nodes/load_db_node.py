"""漏洞库加载节点：加载 db_path，支持 embeddings/lsir/fuzzy_hashes 格式。"""

import json
import os
from typing import Any, Dict

from ..model import DAGNode


def _default_output_key(db_format: str) -> str:
    if db_format == "lsir":
        return "db_lsir"
    if db_format == "fuzzy_hashes":
        return "db_fuzzy_hashes"
    return "db_embeddings"


def _load_db_file(db_path: str, db_format: str) -> Dict[str, Any]:
    """从文件加载漏洞库。"""
    if not db_path or not os.path.isfile(db_path):
        return {"functions": []}
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"functions": []}
    if not isinstance(data, dict):
        return {"functions": []}
    funcs = data.get("functions", [])
    if not isinstance(funcs, list):
        return {"functions": []}
    return {"functions": funcs}


class LoadDBNode(DAGNode):
    """加载漏洞库，按 db_format 产出 db_embeddings / db_lsir / db_fuzzy_hashes。"""

    NODE_TYPE = "load_db"

    def execute(self, ctx: Dict[str, Any]) -> None:
        db_path = self.params.get("db_path", "")
        db_format = self.params.get("db_format", "embeddings")
        output_key = self.params.get("output_key") or _default_output_key(db_format)

        result = _load_db_file(db_path, db_format)
        self.output = result
        ctx[output_key] = result
        self.done = True
