"""Ghidra headless 节点：运行 Ghidra 提取 P-code 到 lsir_raw.json。"""

from typing import Any, Dict

from ..model import DAGNode


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
