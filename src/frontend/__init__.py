"""
前端接口模块
提供与外部工具的集成接口（固件/二进制拆解与反汇编）。

主要组件:
- ghidra_runner.py: Ghidra Headless 调用封装（反汇编与 P-code 提取）
- ghidra_scripts/: Ghidra 脚本目录

预留扩展:
- Binwalk 集成（固件拆解）待后续接入；当前版本主要对已提取的二进制使用 Ghidra 进行分析。
"""

from .ghidra_runner import batch_run_ghidra, run_ghidra_analysis

__all__ = ["batch_run_ghidra", "run_ghidra_analysis"]
