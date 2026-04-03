"""Ghidra 提取节点构建器。"""

import os
from typing import Any, Dict, List, Optional

from utils.config import DAG_GHIDRA_THREAD_SLOTS

from ..model import JobDAG


def build_ghidra_node(
    dag: JobDAG,
    node_id: str,
    binary_path: str,
    output_dir: str,
    deps: Optional[List[str]] = None,
    *,
    force: bool = False,
    timeout: Optional[int] = None,
    project_name: str = "SemPatchProject",
    priority: int = 0,
) -> None:
    """添加 Ghidra 提取 P-code 节点。"""
    binary_path = os.path.abspath(binary_path)
    output_dir = os.path.abspath(output_dir)
    params: Dict[str, Any] = {
        "binary_path": binary_path,
        "output_dir": output_dir,
        "force": force,
        "timeout": timeout,
        "project_name": project_name,
    }
    dag.add_node(
        node_id=node_id,
        node_type="ghidra",
        params=params,
        deps=deps or [],
        priority=priority,
        thread_slots=DAG_GHIDRA_THREAD_SLOTS,
    )
