"""DAG 导出：Mermaid、DOT、HTML。"""

from typing import Optional

from .model import DAGNode, JobDAG


def _node_style(node: DAGNode) -> str:
    if node.failed:
        return "fill:#fdd"
    if node.done:
        return "fill:#dfd"
    return "fill:#ffd"


def export_mermaid(dag: JobDAG, path: Optional[str] = None) -> str:
    """导出 Mermaid 格式。"""
    lines = ["flowchart TB"]
    for nid, node in dag.nodes.items():
        safe = nid.replace("-", "_")
        label = node.display_label(nid)
        lines.append(f'    {safe}["{label}"]')
    for nid, node in dag.nodes.items():
        safe = nid.replace("-", "_")
        style = _node_style(node)
        lines.append(f"    style {safe} {style}")
    for nid, node in dag.nodes.items():
        src = nid.replace("-", "_")
        for d in node.deps:
            dst = d.replace("-", "_")
            lines.append(f"    {dst} --> {src}")
    text = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def export_dot(dag: JobDAG, path: Optional[str] = None) -> str:
    """导出 DOT 格式。"""
    lines = ["digraph G {", "  rankdir=TB;"]
    for nid, node in dag.nodes.items():
        label = node.display_label(nid)
        color = "red" if node.failed else ("green" if node.done else "yellow")
        nid_safe = nid.replace("-", "_")
        lines.append(f'  {nid_safe} [label="{label}", style=filled, fillcolor={color}];')
    for nid, node in dag.nodes.items():
        src = nid.replace("-", "_")
        for d in node.deps:
            dst = d.replace("-", "_")
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
