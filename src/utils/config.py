"""
兼容导入：从 src.config 主入口转发。
保留 utils.config 以兼容现有 from utils.config import ... 写法。
"""
from config import (  # noqa: F401
    ANALYZE_HEADLESS,
    BINARY_CACHE_DIR,
    BINWALK_CMD,
    DAG_GHIDRA_THREAD_SLOTS,
    DAG_MAX_WORKERS,
    FEATURE_GRAPH_ENABLED,
    FEATURE_SEQ_ENABLED,
    GHIDRA_HOME,
    IR_DEFAULT_OPTS,
    LOG_DIR,
    LOG_LEVEL,
    MATCHER_FAISS_INDEX_TYPE,
    MIN_PARALLEL_WORKERS,
    OUTPUT_DIR,
    PARALLEL_WORKERS,
    PROJECT_ROOT,
    UNPACK_OUTPUT_DIR,
)
