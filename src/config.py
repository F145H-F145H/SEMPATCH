"""
SemPatch 配置主入口。从 sempatch.cfg 或环境变量加载。
整合 paths、ghidra、parallel、dag、ir、feature、matcher 等段。
"""

import os
from configparser import RawConfigParser
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_cfg_parser: Optional[RawConfigParser] = None


def _load_cfg() -> Optional[RawConfigParser]:
    global _cfg_parser
    if _cfg_parser is not None:
        return _cfg_parser
    for name in ("sempatch.cfg", "sempatch.cfg.example"):
        path = os.path.join(PROJECT_ROOT, name)
        if os.path.isfile(path):
            try:
                p = RawConfigParser()
                p.read(path, encoding="utf-8")
                _cfg_parser = p
                return _cfg_parser
            except Exception:
                pass
    return None


def _cfg_get(section: str, option: str, env_key: Optional[str], default: str) -> str:
    raw = (os.environ.get(env_key) or "").strip() if env_key else ""
    if raw:
        return raw
    cfg = _load_cfg()
    if cfg and cfg.has_section(section) and cfg.has_option(section, option):
        return cfg.get(section, option).strip()
    return default


def _cfg_get_int(section: str, option: str, env_key: Optional[str], default: int) -> int:
    if env_key and os.environ.get(env_key, "").strip():
        try:
            return int(os.environ.get(env_key, "").strip())
        except ValueError:
            pass
    cfg = _load_cfg()
    if cfg and cfg.has_section(section) and cfg.has_option(section, option):
        try:
            return int(cfg.get(section, option).strip())
        except ValueError:
            pass
    return default


# paths
_output_dir_rel = _cfg_get("paths", "output_dir", "SEMPATCH_OUTPUT_DIR", "output")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, _output_dir_rel)
_log_dir_rel = _cfg_get("paths", "log_dir", None, os.path.join(_output_dir_rel, "logs"))
LOG_DIR = os.path.join(PROJECT_ROOT, _log_dir_rel)

_data_dir_rel = _cfg_get("paths", "data_dir", "SEMPATCH_DATA_DIR", "data").strip() or "data"
if os.path.isabs(_data_dir_rel):
    DATA_DIR = _data_dir_rel
else:
    DATA_DIR = os.path.join(PROJECT_ROOT, _data_dir_rel)

_two_stage_rel = (
    _cfg_get("paths", "two_stage_rel", "SEMPATCH_TWO_STAGE_REL", "two_stage").strip() or "two_stage"
)
DEFAULT_TWO_STAGE_DIR = os.path.join(DATA_DIR, _two_stage_rel)
DEFAULT_TWO_STAGE_PATH = os.path.normpath(os.path.join(_data_dir_rel, _two_stage_rel))

_match_run_subdir = (
    _cfg_get("paths", "match_run_subdir", "SEMPATCH_MATCH_RUN_SUBDIR", "match_run").strip()
    or "match_run"
)
DEFAULT_MATCH_OUTPUT_DIR = os.path.join(OUTPUT_DIR, _match_run_subdir)
DEFAULT_MATCH_OUTPUT_PATH = os.path.normpath(os.path.join(_output_dir_rel, _match_run_subdir))

_unpack_output_rel = _cfg_get(
    "paths", "unpack_output_dir", "SEMPATCH_UNPACK_DIR", "output/unpacked"
).strip() or "output/unpacked"
UNPACK_OUTPUT_DIR = (
    os.path.join(PROJECT_ROOT, _unpack_output_rel)
    if not os.path.isabs(_unpack_output_rel)
    else _unpack_output_rel
)
BINWALK_CMD = _cfg_get("unpack", "binwalk_cmd", "SEMPATCH_BINWALK_CMD", "binwalk")

_binary_cache_rel = _cfg_get(
    "paths", "binary_cache_dir", "SEMPATCH_BINARY_CACHE_DIR", "output/binary_cache"
).strip()
if _binary_cache_rel:
    BINARY_CACHE_DIR = (
        os.path.join(PROJECT_ROOT, _binary_cache_rel)
        if not os.path.isabs(_binary_cache_rel)
        else _binary_cache_rel
    )
else:
    BINARY_CACHE_DIR = None

# ghidra
_ghidra_rel = _cfg_get("ghidra", "ghidra_home", "GHIDRA_HOME", "").strip()
if _ghidra_rel and os.path.isabs(_ghidra_rel):
    GHIDRA_HOME = _ghidra_rel
else:
    GHIDRA_HOME = os.environ.get("GHIDRA_HOME") or (
        os.path.join(PROJECT_ROOT, _ghidra_rel or "third_party/ghidra_12.0_PUBLIC")
    )
ANALYZE_HEADLESS = os.path.join(GHIDRA_HOME, "support", "analyzeHeadless")

# parallel
_parallel_val = _cfg_get_int("parallel", "parallel_workers", "SEMPATCH_PARALLEL_WORKERS", 8)
PARALLEL_WORKERS = min(max(0, _parallel_val), 64)
MIN_PARALLEL_WORKERS = _cfg_get_int(
    "parallel", "min_parallel_workers", "SEMPATCH_MIN_PARALLEL_WORKERS", 1
)

# dag
DAG_GHIDRA_THREAD_SLOTS = _cfg_get_int("dag", "ghidra_thread_slots", "SEMPATCH_DAG_GHIDRA_THREAD_SLOTS", 2)
DAG_MAX_WORKERS = _cfg_get_int("dag", "max_workers", "SEMPATCH_DAG_MAX_WORKERS", 4)
PIPELINE_STRATEGY = _cfg_get(
    "dag", "pipeline_strategy", "SEMPATCH_PIPELINE_STRATEGY", "semantic_embed"
).strip().lower() or "semantic_embed"

# ir (扩展)
IR_DEFAULT_OPTS = _cfg_get("ir", "default_opts", None, "")

# feature (扩展)
FEATURE_GRAPH_ENABLED = _cfg_get("feature", "graph_enabled", "SEMPATCH_FEATURE_GRAPH", "true").lower() == "true"
FEATURE_SEQ_ENABLED = _cfg_get("feature", "seq_enabled", "SEMPATCH_FEATURE_SEQ", "true").lower() == "true"

# matcher (扩展)
MATCHER_FAISS_INDEX_TYPE = _cfg_get("matcher", "faiss_index_type", "SEMPATCH_FAISS_INDEX", "flat")

# log
LOG_LEVEL = _cfg_get("log", "log_level", "SEMPATCH_LOG_LEVEL", "INFO").upper()
if LOG_LEVEL not in frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}):
    LOG_LEVEL = "INFO"
