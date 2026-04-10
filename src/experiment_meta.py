"""实验元数据收集与确定性种子管理。

提供三个公开函数：
  set_deterministic(seed)  — 设置全部随机种子 + cuDNN 确定性标志
  collect_metadata(args)   — 收集 git commit、依赖版本、CLI 参数等
  save_metadata(path, args, extra) — 将 metadata 写入 <model_path>.metadata.json
"""

import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def set_deterministic(seed: int) -> None:
    """设置全部随机源为确定性模式，保证可复现。"""
    import random

    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy

        numpy.random.seed(seed)
    except ImportError:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _git_info() -> Dict[str, Any]:
    """安全获取 git commit hash 与 dirty 状态。"""
    info: Dict[str, Any] = {}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        info["git_commit"] = commit
    except Exception:
        info["git_commit"] = "unknown"
    try:
        subprocess.check_call(
            ["git", "diff", "--quiet", "--exit-code"],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            timeout=5,
        )
        info["git_dirty"] = False
    except subprocess.CalledProcessError:
        info["git_dirty"] = True
    except Exception:
        info["git_dirty"] = "unknown"
    return info


def _env_info() -> Dict[str, Any]:
    """收集运行环境信息。"""
    import torch

    info: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "torch_version": torch.__version__,
    }
    try:
        import numpy

        info["numpy_version"] = numpy.__version__
    except ImportError:
        info["numpy_version"] = None
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
    else:
        info["cuda_version"] = None
        info["gpu_name"] = None
    return info


def collect_metadata(args: Any, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """收集完整实验元数据。args 为 argparse.Namespace。"""
    meta: Dict[str, Any] = {}
    meta.update(_git_info())
    meta["seed"] = getattr(args, "seed", None)
    meta["cli_args"] = vars(args) if hasattr(args, "__dict__") else {}
    meta.update(_env_info())
    if extra:
        meta["extra"] = extra
    return meta


def save_metadata(
    model_path: str,
    args: Any,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """将 metadata 写入 <model_path>.metadata.json，返回写入路径。"""
    meta = collect_metadata(args, extra=extra)
    meta_path = model_path + ".metadata.json"
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    logger.info("实验 metadata 已保存至 %s", meta_path)
    return meta_path
