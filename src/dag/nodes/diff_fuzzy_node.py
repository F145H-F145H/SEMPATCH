"""模糊哈希匹配节点：固件 fuzzy_hashes vs 漏洞库 db_fuzzy_hashes。"""

import logging
from typing import Any, Dict, List

from ..model import DAGNode
from ..specs import assert_ctx_keys

logger = logging.getLogger(__name__)


def _fuzzy_compare(h1: str, h2: str, algorithm: str) -> float:
    """返回 0~100 的相似度。"""
    if not h1 or not h2:
        return 0.0
    if algorithm == "ssdeep":
        try:
            import ssdeep

            return float(ssdeep.compare(h1, h2))
        except ImportError:
            pass
        except Exception:
            pass
    if algorithm == "tlsh":
        try:
            import tlsh

            d = tlsh.diff(h1, h2)
            if d is not None:
                return max(0.0, 100.0 - min(float(d), 100.0))
        except ImportError:
            pass
        except Exception:
            pass
    return 0.0


class DiffFuzzyNode(DAGNode):
    """固件模糊哈希 vs 漏洞库模糊哈希，输出相似度匹配。"""

    NODE_TYPE = "diff_fuzzy"

    def execute(self, ctx: Dict[str, Any]) -> None:
        fw_key = self.params.get("fuzzy_hashes_key", "fuzzy_hashes")
        db_key = self.params.get("db_fuzzy_hashes_key", "db_fuzzy_hashes")
        output_key = self.params.get("output_key", "diff_result")
        threshold = float(self.params.get("threshold", 0.0))

        assert_ctx_keys(ctx, [fw_key, db_key], "DiffFuzzyNode: ")

        fw_hashes = ctx[fw_key]
        db_hashes = ctx[db_key]
        fw_funcs = fw_hashes.get("functions", [])
        db_funcs = db_hashes.get("functions", [])

        matches: List[Dict[str, Any]] = []
        for fe in fw_funcs:
            h1 = fe.get("hash", "")
            algo = fe.get("algorithm", "ssdeep")
            if not h1:
                continue
            for de in db_funcs:
                h2 = de.get("hash", "")
                if not h2:
                    continue
                db_algo = de.get("algorithm", "ssdeep")
                if algo != db_algo:
                    logger.warning(
                        "模糊哈希算法不匹配: firmware=%s, db=%s, 跳过 %s vs %s",
                        algo,
                        db_algo,
                        fe.get("name", "?"),
                        de.get("name", "?"),
                    )
                    continue
                sim = _fuzzy_compare(h1, h2, algo)
                if sim >= threshold:
                    matches.append(
                        {
                            "firmware_func": fe.get("name", ""),
                            "db_func": de.get("name", ""),
                            "similarity": sim,
                            "method": "fuzzy_hash",
                        }
                    )

        result = {"matches": matches}
        self.output = result
        ctx[output_key] = result
        self.done = True
