"""启动时配置自检：检查关键路径与工具是否可用。"""

import logging
import os
import shutil
from typing import List

logger = logging.getLogger(__name__)


def validate_config() -> List[str]:
    """
    检查 GHIDRA_HOME、analyzeHeadless、DATA_DIR 等关键配置。
    返回问题列表；每个问题是一条人可读的警告信息。
    """
    problems: List[str] = []

    # 延迟导入避免循环依赖
    try:
        from config import ANALYZE_HEADLESS, DATA_DIR, GHIDRA_HOME
    except ImportError:
        return ["无法导入 config 模块，跳过配置校验"]

    # Ghidra 目录
    if not GHIDRA_HOME or not os.path.isdir(GHIDRA_HOME):
        problems.append(f"GHIDRA_HOME 目录不存在: {GHIDRA_HOME}")

    # analyzeHeadless
    if not ANALYZE_HEADLESS or not os.path.isfile(ANALYZE_HEADLESS):
        problems.append(f"analyzeHeadless 不存在: {ANALYZE_HEADLESS}")

    # DATA_DIR
    if not DATA_DIR or not os.path.isdir(DATA_DIR):
        problems.append(f"DATA_DIR 目录不存在: {DATA_DIR}")

    # binwalk（可选）
    if not shutil.which("binwalk"):
        problems.append("binwalk 未安装（固件解包功能不可用，可忽略）")

    # 统一日志输出
    for p in problems:
        logger.warning("配置校验: %s", p)

    return problems
