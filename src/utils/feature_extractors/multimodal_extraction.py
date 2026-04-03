"""
从 lsir_raw 提取单函数 multimodal 特征的共享逻辑。

供 build_library_features、filter_index_by_pcode_len 复用，作为 lsir_raw -> multimodal
提取链的单一事实来源。
"""

from typing import Any, Dict, List


def _norm_entry(entry: str) -> str:
    """统一 entry 格式：小写、0x 前缀。"""
    s = (entry or "").strip().lower()
    if s.startswith("0x"):
        return s
    return f"0x{s}" if s else "0x0"


def _entry_matches(a: str, b: str) -> bool:
    """判断两个 entry 是否表示同一地址。"""
    na = _norm_entry(a)
    nb = _norm_entry(b)
    if na == nb:
        return True
    try:
        return int(na, 16) == int(nb, 16)
    except ValueError:
        return False


def extract_multimodal_from_lsir_raw(lsir_raw_funcs: List[Dict[str, Any]], entry: str) -> Dict[str, Any]:
    """
    从 lsir_raw 函数列表中按 entry 匹配，提取单函数 multimodal 特征。

    供 build_library_features、filter_index_by_pcode_len 复用。
    未找到或提取失败时抛出 ValueError。

    Args:
        lsir_raw_funcs: lsir_raw 的 functions 列表
        entry: 目标函数入口地址（十六进制字符串，如 0x401000）

    Returns:
        multimodal dict，含 graph、sequence；默认含 dfg（与 graph 同 schema，可为空图；参见 @architecture.md §1.3）

    Raises:
        ValueError: 未找到函数、LSIR 构建失败或融合后无 multimodal
    """
    from utils.ir_builder import build_lsir
    from utils.pcode_normalizer import normalize_lsir_raw
    from utils.feature_extractors import (
        extract_acfg_features,
        extract_graph_features,
        extract_sequence_features,
        fuse_features,
    )

    target = None
    for f in lsir_raw_funcs:
        if _entry_matches(f.get("entry", ""), entry):
            target = f
            break
    if target is None:
        raise ValueError(f"函数 entry={entry} 在 lsir_raw 中未找到")

    raw = {"functions": [target]}
    raw = normalize_lsir_raw(raw)
    lsir = build_lsir(raw, include_cfg=True, include_dfg=True)
    fn_list = lsir.get("functions", [])
    if not fn_list:
        raise ValueError(f"函数 entry={entry} 构建 LSIR 后无结果")

    fn = fn_list[0]
    gf = extract_graph_features(fn)
    sf = extract_sequence_features(fn)
    acfg = extract_acfg_features(fn)
    fused = fuse_features(gf, sf, acfg_feats=acfg)
    multimodal = fused.get("multimodal")
    if not multimodal:
        raise ValueError(f"函数 entry={entry} 融合后无 multimodal 特征")

    return multimodal
