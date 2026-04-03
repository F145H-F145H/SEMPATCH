"""
filter_index_by_pcode_len 的断点续跑辅助函数。

职责：
- 计算输入索引摘要
- 生成默认 checkpoint 路径
- checkpoint 原子写入/读取
- 续跑元数据一致性校验
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Tuple

CHECKPOINT_VERSION = 1


def build_default_checkpoint_path(output_path: str) -> str:
    """默认 checkpoint 与输出索引放在一起。"""
    abs_out = os.path.abspath(output_path)
    return f"{abs_out}.filter_checkpoint.json"


def compute_file_sha256(path: str, *, chunk_size: int = 1024 * 1024) -> str:
    """流式计算文件 sha256，避免一次性读入内存。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(path: str) -> Dict[str, Any]:
    abs_path = os.path.abspath(path)
    with open(abs_path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("checkpoint 文件格式错误：顶层必须是 JSON 对象")
    return raw


def save_checkpoint_atomic(path: str, payload: Dict[str, Any]) -> None:
    """先写临时文件再 replace，避免中断留下半截 checkpoint。"""
    abs_path = os.path.abspath(path)
    tmp_path = f"{abs_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, abs_path)


def validate_checkpoint_meta(
    state: Dict[str, Any],
    expected_meta: Dict[str, Any],
) -> Tuple[bool, str]:
    """校验 checkpoint 元数据与当前运行参数是否匹配。"""
    version = state.get("version")
    if version != CHECKPOINT_VERSION:
        return False, f"checkpoint 版本不匹配: got={version}, expect={CHECKPOINT_VERSION}"

    meta = state.get("meta")
    if not isinstance(meta, dict):
        return False, "checkpoint 缺少 meta"

    for k, exp_v in expected_meta.items():
        got_v = meta.get(k)
        if got_v != exp_v:
            return False, f"checkpoint 元数据不匹配: {k} got={got_v!r}, expect={exp_v!r}"
    return True, "ok"
