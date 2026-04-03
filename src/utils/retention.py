"""
磁盘占用控制：限制目录总大小，只保留近期文件。

用于训练/验证阶段产生的临时文件与日志滚动备份清理。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import shutil
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RetentionStats:
    removed_files: int
    removed_bytes: int
    kept_files: int
    kept_bytes: int


def _iter_files(root: str) -> Iterable[str]:
    for base, _dirs, files in os.walk(root):
        for name in files:
            yield os.path.join(base, name)


def _dir_total_bytes(root: str) -> int:
    total = 0
    for path in _iter_files(root):
        try:
            if os.path.isfile(path):
                total += int(os.stat(path).st_size)
        except FileNotFoundError:
            continue
        except OSError:
            continue
    return total


def enforce_subdir_retention(
    root: str,
    *,
    keep_recent_dirs: int = 5,
    name_prefix: Optional[str] = None,
) -> Tuple[int, int]:
    """
    仅保留 root 下最近 keep_recent_dirs 个子目录（按目录 mtime 倒序）。

    返回 (removed_dirs, removed_bytes_estimated)。
    """
    if not root or not os.path.isdir(root):
        return 0, 0
    keep_n = max(0, int(keep_recent_dirs))

    dirs = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if name_prefix and not name.startswith(name_prefix):
            continue
        try:
            mtime = float(os.stat(path).st_mtime)
        except FileNotFoundError:
            continue
        except OSError:
            continue
        dirs.append((path, mtime))

    dirs.sort(key=lambda x: x[1], reverse=True)
    to_remove = dirs[keep_n:]
    removed_dirs = 0
    removed_bytes = 0
    for path, _mtime in to_remove:
        try:
            removed_bytes += _dir_total_bytes(path)
            shutil.rmtree(path, ignore_errors=True)
            removed_dirs += 1
        except Exception:
            continue
    return removed_dirs, removed_bytes


def enforce_dir_size_limit(
    root: str,
    *,
    max_total_bytes: int,
    keep_recent: int = 20,
    allow_extensions: Optional[Sequence[str]] = None,
) -> RetentionStats:
    """
    将 root 目录内文件总大小限制在 max_total_bytes 以内。

    - keep_recent: 无论大小上限如何，至少保留最近 keep_recent 个文件（按 mtime 倒序）。
    - allow_extensions: 若指定，仅对这些后缀的文件参与统计与清理（如 [".json", ".pt", ".log"]）。
    """
    if max_total_bytes <= 0:
        return RetentionStats(0, 0, 0, 0)
    if not root or not os.path.isdir(root):
        return RetentionStats(0, 0, 0, 0)

    allow_ext = None
    if allow_extensions:
        allow_ext = tuple(e.lower() for e in allow_extensions)

    entries = []
    for path in _iter_files(root):
        if allow_ext is not None:
            _, ext = os.path.splitext(path)
            if ext.lower() not in allow_ext:
                continue
        try:
            st = os.stat(path)
        except FileNotFoundError:
            continue
        if not os.path.isfile(path):
            continue
        entries.append((path, float(st.st_mtime), int(st.st_size)))

    entries.sort(key=lambda x: x[1], reverse=True)  # newest first
    kept = entries[: max(0, int(keep_recent))]
    rest = entries[len(kept) :]

    kept_bytes = sum(sz for _p, _m, sz in kept)
    total_bytes = kept_bytes + sum(sz for _p, _m, sz in rest)
    removed_files = 0
    removed_bytes = 0

    # 从最老的开始删除，直到满足上限
    if total_bytes > max_total_bytes:
        rest.sort(key=lambda x: x[1])  # oldest first
        for path, _mtime, size in rest:
            if total_bytes <= max_total_bytes:
                break
            try:
                os.remove(path)
                removed_files += 1
                removed_bytes += size
                total_bytes -= size
            except FileNotFoundError:
                continue
            except OSError:
                # 权限/占用等原因删除失败，跳过
                continue

    # 重新计算 kept_files/bytes（不再遍历磁盘，近似足够）
    kept_files = len(entries) - removed_files
    kept_bytes_final = max(0, total_bytes)
    return RetentionStats(removed_files, removed_bytes, kept_files, kept_bytes_final)

