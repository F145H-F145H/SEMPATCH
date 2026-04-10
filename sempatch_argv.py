"""
./sempatch 入口参数改写：供根目录 `sempatch` 脚本与单元测试共用。

此模块被 ``sempatch`` 包装脚本导入使用（非废弃代码）。

- 已知子命令 compare / extract / match / unpack 原样传递。
- 「ELF + 两阶段库目录」双位置参数 → match --query-binary … --two-stage-dir …
- 其余非选项首参 → 兼容旧用法 extract <binary> …
"""

from __future__ import annotations

import os
from typing import List

KNOWN_SUBCOMMANDS = frozenset({"compare", "extract", "match", "unpack"})


def looks_like_two_stage_lib_dir(path: str) -> bool:
    """目录下是否存在库侧约定 JSON（其一即可）。"""
    if not path or not os.path.isdir(path):
        return False
    lib_feat = os.path.join(path, "library_features.json")
    lib_emb = os.path.join(path, "library_safe_embeddings.json")
    return os.path.isfile(lib_feat) or os.path.isfile(lib_emb)


def rewrite_sempatch_argv(argv: List[str]) -> List[str]:
    if not argv:
        return argv
    if argv[0] in ("-h", "--help"):
        return list(argv)
    if argv[0] in KNOWN_SUBCOMMANDS:
        return list(argv)
    if len(argv) >= 2:
        bin_path = argv[0]
        lib_dir = os.path.expanduser(argv[1])
        if (
            os.path.isfile(bin_path)
            and os.path.isdir(lib_dir)
            and looks_like_two_stage_lib_dir(lib_dir)
        ):
            return ["match", "--query-binary", bin_path, "--two-stage-dir", lib_dir] + list(
                argv[2:]
            )
    if argv[0].startswith("-"):
        return list(argv)
    return ["extract"] + list(argv)
