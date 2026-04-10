"""Unpack 节点：使用 binwalk 解包固件镜像。"""

import os
import shutil
import subprocess
from typing import Any, Dict, List

from ..model import DAGNode


def _find_elf_binaries(root_dir: str) -> List[str]:
    """递归查找目录中的 ELF 可执行文件。"""
    bins: List[str] = []
    try:
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                try:
                    if os.path.isfile(fp) and os.access(fp, os.R_OK):
                        with open(fp, "rb") as f:
                            header = f.read(4)
                        if header == b"\x7fELF":
                            bins.append(fp)
                except (OSError, IOError):
                    continue
    except OSError:
        pass
    return bins


class UnpackNode(DAGNode):
    """使用 binwalk 解包固件镜像，输出解包目录及可分析二进制列表。"""

    NODE_TYPE = "unpack"

    def execute(self, ctx: Dict[str, Any]) -> None:
        from config import BINWALK_CMD, PROJECT_ROOT, UNPACK_OUTPUT_DIR

        p = self.params
        firmware_path = os.path.abspath(p["firmware_path"])
        output_dir = p.get("output_dir")
        if output_dir:
            output_dir = (
                os.path.join(PROJECT_ROOT, output_dir)
                if not os.path.isabs(output_dir)
                else output_dir
            )
        else:
            output_dir = UNPACK_OUTPUT_DIR

        base_name = os.path.splitext(os.path.basename(firmware_path))[0]
        extract_dir = os.path.join(output_dir, base_name)
        os.makedirs(extract_dir, exist_ok=True)

        binwalk_cmd = p.get("binwalk_cmd") or BINWALK_CMD
        if not shutil.which(binwalk_cmd):
            raise RuntimeError(
                f"binwalk 未安装或未在 PATH 中 (cmd={binwalk_cmd!r})。"
                "请安装: apt install binwalk 或 pip install binwalk"
            )
        # binwalk v2.x 使用 -d/--directory，旧版使用 -C
        cmd = [binwalk_cmd, "-e", "-d", extract_dir, firmware_path]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=p.get("timeout", 300),
            check=False,
        )
        if result.returncode != 0 and result.stderr:
            raise RuntimeError(
                f"binwalk 解包失败 (exit {result.returncode}): {result.stderr[:500]}"
            )

        unpack_binaries = _find_elf_binaries(extract_dir)
        self.output = {
            "unpack_dir": extract_dir,
            "unpack_binaries": unpack_binaries,
        }
        ctx["unpack_dir"] = extract_dir
        ctx["unpack_binaries"] = unpack_binaries
        self.done = True
