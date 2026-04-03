"""固件解包节点构建器。"""

import os
from typing import Any, Dict, List, Optional

from ..model import JobDAG


def build_unpack_node(
    dag: JobDAG,
    node_id: str,
    firmware_path: str,
    output_dir: Optional[str] = None,
    deps: Optional[List[str]] = None,
    *,
    binwalk_cmd: Optional[str] = None,
    timeout: Optional[int] = 300,
    output_key: str = "unpack_dir",
    priority: int = 0,
) -> None:
    """添加 binwalk 解包固件节点。"""
    firmware_path = os.path.abspath(firmware_path)
    params: Dict[str, Any] = {
        "firmware_path": firmware_path,
        "output_dir": output_dir,
        "timeout": timeout,
        "output_key": output_key,
    }
    if binwalk_cmd:
        params["binwalk_cmd"] = binwalk_cmd
    dag.add_node(
        node_id=node_id,
        node_type="unpack",
        params=params,
        deps=deps or [],
        priority=priority,
    )
