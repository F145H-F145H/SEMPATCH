"""DAG 导出：Mermaid、DOT、HTML。"""

import logging
import re
from typing import Dict, Optional

from .model import DAGNode, JobDAG

logger = logging.getLogger(__name__)


def _sanitize_id(nid: str) -> str:
    """将 node_id 转为 Mermaid/DOT 合法标识符。"""
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", nid)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe


def _node_style(node: DAGNode) -> str:
    if node.failed:
        return "fill:#fdd"
    if node.done:
        return "fill:#dfd"
    return "fill:#ffd"


def export_mermaid(dag: JobDAG, path: Optional[str] = None) -> str:
    """导出 Mermaid 格式。"""
    id_map: Dict[str, str] = {}
    collision_check: Dict[str, str] = {}
    for nid in dag.nodes:
        safe = _sanitize_id(nid)
        if safe in collision_check and collision_check[safe] != nid:
            logger.warning(
                "Mermaid 导出 ID 碰撞: %s 和 %s 均映射为 %s", collision_check[safe], nid, safe
            )
        collision_check[safe] = nid
        id_map[nid] = safe

    lines = ["flowchart TB"]
    for nid, node in dag.nodes.items():
        safe = id_map[nid]
        label = node.display_label(nid)
        lines.append(f'    {safe}["{label}"]')
    for nid, node in dag.nodes.items():
        safe = id_map[nid]
        style = _node_style(node)
        lines.append(f"    style {safe} {style}")
    for nid, node in dag.nodes.items():
        src = id_map[nid]
        for d in node.deps:
            dst = id_map.get(d, _sanitize_id(d))
            lines.append(f"    {dst} --> {src}")
    text = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def export_dot(dag: JobDAG, path: Optional[str] = None) -> str:
    """导出 DOT 格式。"""
    id_map: Dict[str, str] = {}
    for nid in dag.nodes:
        id_map[nid] = _sanitize_id(nid)

    lines = ["digraph G {", "  rankdir=TB;"]
    for nid, node in dag.nodes.items():
        label = node.display_label(nid)
        color = "red" if node.failed else ("green" if node.done else "yellow")
        nid_safe = id_map[nid]
        lines.append(f'  {nid_safe} [label="{label}", style=filled, fillcolor={color}];')
    for nid, node in dag.nodes.items():
        src = id_map[nid]
        for d in node.deps:
            dst = id_map.get(d, _sanitize_id(d))
            lines.append(f"  {dst} -> {src};")
    lines.append("}")
    text = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def export_html(dag: JobDAG, path: Optional[str] = None) -> str:
    """导出带状态着色的 HTML 摘要。"""
    rows = []
    for nid, node in sorted(dag.nodes.items()):
        status = "failed" if node.failed else ("done" if node.done else "pending")
        rows.append(f"<tr><td>{nid}</td><td>{node.display_label(nid)}</td><td>{status}</td></tr>")
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>SemPatch DAG</title></head>
<body>
<h1>DAG 状态</h1>
<table border="1">
<tr><th>Node</th><th>Label</th><th>Status</th></tr>
{chr(10).join(rows)}
</table>
</body>
</html>"""
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
    return html
