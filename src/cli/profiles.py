"""预设配置 profile：简化 CLI 参数输入。"""

from __future__ import annotations

from typing import Any, Dict

PROFILES: Dict[str, Dict[str, Any]] = {
    "quick": {
        "coarse_k": 50,
        "top_k": 5,
        "match_filter": "top_k",
        "max_queries": 10,
    },
    "standard": {
        "coarse_k": 100,
        "top_k": 10,
        "match_filter": "unique",
        "min_similarity": 0.95,
    },
    "full": {
        "coarse_k": 500,
        "top_k": 50,
        "match_filter": "all_above",
        "min_similarity": 0.9,
    },
}

# profile 可覆盖的参数及其硬编码默认值（无 profile 时使用）
PROFILE_KEYS = {"coarse_k", "top_k", "match_filter", "min_similarity", "max_queries"}
HARDCODED_DEFAULTS: Dict[str, Any] = {
    "coarse_k": 100,
    "top_k": 10,
    "match_filter": "top_k",
    "min_similarity": 0.95,
    "max_queries": 0,
}


def get_profile(name: str) -> Dict[str, Any]:
    """返回 profile 配置；名称无效时抛出 ValueError。"""
    if name not in PROFILES:
        raise ValueError(f"未知 profile: {name!r}，可选: {', '.join(PROFILES)}")
    return dict(PROFILES[name])


def resolve_with_profile(args, profile_name: str | None) -> Dict[str, Any]:
    """解析参数值：CLI 显式指定 > profile 默认 > 硬编码默认。"""
    result: Dict[str, Any] = dict(HARDCODED_DEFAULTS)

    # Layer 1: profile defaults
    if profile_name:
        result.update(get_profile(profile_name))

    # Layer 2: CLI explicit values (not SUPPRESS)
    for key in PROFILE_KEYS:
        val = getattr(args, key, None)
        if val is not None:
            result[key] = val

    return result
