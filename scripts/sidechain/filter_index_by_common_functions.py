#!/usr/bin/env python3
"""
同源程序交叉过滤：对同一 project_id 的多个编译变体，只保留在所有变体中都出现的函数。

核心逻辑：
  1. 用 binkit_provenance.derive_project_id 对 binary 路径分组（同源弱键）
  2. 按函数名（name）求交集：只保留「在该 project 所有变体中都存在」的函数
  3. 无函数名（stripped binary）的变体不参与交集，但保留原样输出（避免误杀）

典型场景：
  coreutils-9.1_gcc-10.3.0_x86_64_O2_fmt
  coreutils-9.1_gcc-10.3.0_x86_64_O3_fmt
  coreutils-9.1_clang-8.0.0_x86_64_O2_fmt
  → 只保留三个变体共有的函数名

用法:
  # 接在 filter_index_by_pcode_len 后面跑
  python scripts/sidechain/filter_index_by_common_functions.py \
    -i data/binkit_functions_filtered.json \
    -o data/binkit_functions_common.json

  # 直接跑原始数据（先按 pcode len 过滤，再交叉过滤）
  python scripts/sidechain/filter_index_by_common_functions.py \
    -i data/binkit_functions.json \
    -o data/binkit_functions_common.json \
    --min-pcode-len 16

  # 按函数名 hash（地址无关）匹配，适合跨架构场景
  python scripts/sidechain/filter_index_by_common_functions.py \
    -i data/binkit_functions_filtered.json \
    -o data/binkit_functions_common.json \
    --match-by name

  # 只处理有 >= N 个变体的 project（单变体 project 保留全部）
  python scripts/sidechain/filter_index_by_common_functions.py \
    -i data/binkit_functions_filtered.json \
    -o data/binkit_functions_common.json \
    --min-variants 2
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

logger = logging.getLogger(__name__)


# GCC/Clang IPA 优化后缀：isra/constprop/part/lto_priv/cold/clone 等
# 可能叠加（如 .isra.0.constprop.1），用循环剥离
_IPA_SUFFIX_PAT = (
    r"\.(isra|constprop|part|lto_priv|cold|hot|clone|llvm\.\d+)(?:\.\d+)*$"
)

# 编译器生成的 thunk/stub 前缀
_THUNK_PREFIX_RE = re.compile(r"^(_GLOBAL__sub_I_|__static_initialization|__cxx_global_var_init)")


def _normalize_fn_name(name: str) -> str:
    """剥离 GCC/Clang IPA 优化后缀（可能叠加），使同源函数名可匹配。"""
    name = name.strip()
    if not name:
        return ""
    prev = None
    while prev != name:
        prev = name
        name = re.sub(_IPA_SUFFIX_PAT, "", name, flags=re.IGNORECASE)
    return name


def _function_key(fn: dict, match_by: str) -> Optional[str]:
    """为函数生成匹配键。name 模式用函数名（经 IPA 后缀归一化）；entry 模式用 name|entry 组合。"""
    raw_name = (fn.get("name") or "").strip()
    if not raw_name:
        return None  # stripped binary，无法匹配
    name = _normalize_fn_name(raw_name)
    if not name or _THUNK_PREFIX_RE.match(name):
        return None  # 编译器生成的 thunk/stub，跳过
    if match_by == "name":
        return name
    # name|entry：同名但地址不同视为不同函数（极少出现，保留安全）
    entry = (fn.get("entry") or "").strip().lower()
    return f"{name}|{entry}"


def _count_pcode_tokens(fn: dict) -> int:
    """数函数的 pcode tokens（轻量，不走管道）。"""
    ntok = 0
    for bb in fn.get("lsir_raw", {}).get("basic_blocks", []) if isinstance(fn.get("lsir_raw"), dict) else []:
        if not isinstance(bb, dict):
            continue
        for inst in bb.get("instructions", []) or []:
            if not isinstance(inst, dict):
                continue
            ntok += len(inst.get("pcode") or [])
    # 如果 lsir_raw 不在索引中（被 filter_index_by_pcode_len 过滤后只保留结构），返回 0
    return ntok


def filter_common_functions(
    index_items: List[dict],
    match_by: str = "name",
    min_pcode_len: int = 0,
    min_variants: int = 2,
    min_ratio: float = 1.0,
    project_root: str = "",
) -> Tuple[List[dict], Dict[str, Any]]:
    """
    按 project_id 分组，对同源变体做函数过滤。

    min_ratio=1.0: 全交集（所有变体共有的函数）
    min_ratio=0.5: 多数投票（≥50% 变体中出现的函数保留）

    Returns:
        (filtered_items, stats)
    """
    from utils.binkit_provenance import derive_project_id

    # ── 第一遍：按 project_id 分组，收集每个 project 的 {variant: {func_key: fn}} ──
    project_variants: Dict[str, Dict[str, Dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    # variant = binary_rel，一个 binary_rel 对应一个变体

    ungroupable: List[dict] = []  # 无法分组的条目（无 binary 字段等）

    for item in index_items:
        binary_rel = item.get("binary", "")
        if not binary_rel:
            ungroupable.append(item)
            continue

        project_id = derive_project_id(binary_rel)
        funcs = item.get("functions", [])
        variant_key = binary_rel

        fn_map: Dict[str, dict] = {}
        for fn in funcs:
            key = _function_key(fn, match_by)
            if key is not None:
                fn_map[key] = fn

        project_variants[project_id][variant_key] = fn_map

    # ── 第二遍：对每个 project 求函数名交集 ──
    stats = {
        "total_projects": 0,
        "single_variant_projects": 0,
        "multi_variant_projects": 0,
        "total_original_functions": 0,
        "total_kept_functions": 0,
        "total_dropped_functions": 0,
        "projects_affected": 0,
    }

    kept_items: List[dict] = []

    for project_id, variants in sorted(project_variants.items()):
        stats["total_projects"] += 1
        n_variants = len(variants)

        if n_variants < min_variants:
            # 变体不足阈值，保留全部
            stats["single_variant_projects"] += 1
            for variant_key, fn_map in variants.items():
                all_funcs = []
                for fn_map_val in fn_map.values():
                    all_funcs.append(fn_map_val)
                # 也加入无法按 name 匹配的函数（name 为空的 stripped 函数）
                for item in index_items:
                    if item.get("binary", "") == variant_key:
                        for fn in item.get("functions", []):
                            if _function_key(fn, match_by) is None:
                                all_funcs.append(fn)
                        break
                kept_items.append({"binary": variant_key, "functions": all_funcs})
                stats["total_original_functions"] += len(all_funcs)
                stats["total_kept_functions"] += len(all_funcs)
            continue

        stats["multi_variant_projects"] += 1

        # 统计每个函数名出现在几个变体中
        fn_variant_count: Dict[str, int] = defaultdict(int)
        for fn_map in variants.values():
            for k in fn_map:
                fn_variant_count[k] += 1

        # 按 min_ratio 保留：函数出现在 ≥ min_ratio * n_variants 个变体中
        required = max(1, int(min_ratio * n_variants))
        if min_ratio >= 1.0:
            required = n_variants  # 全交集：必须在所有变体中出现
        common_keys: Set[str] = {k for k, cnt in fn_variant_count.items() if cnt >= required}

        # 统计：按实际每个变体中保留的函数数计算
        total_in_project = 0
        kept_in_project = 0
        for fn_map in variants.values():
            total_in_project += len(fn_map)
            kept_in_project += sum(1 for k in fn_map if k in common_keys)
        dropped_in_project = total_in_project - kept_in_project

        stats["total_original_functions"] += total_in_project
        stats["total_kept_functions"] += kept_in_project
        stats["total_dropped_functions"] += dropped_in_project

        if dropped_in_project > 0:
            stats["projects_affected"] += 1

        logger.info(
            "[%s] %d 变体 (阈值≥%d), %d 保留函数 / %d 总函数 (丢弃 %d)",
            project_id,
            n_variants,
            required,
            len(common_keys),
            total_in_project,
            dropped_in_project,
        )

        # 构建输出
        for variant_key, fn_map in variants.items():
            kept_funcs = [fn_map[k] for k in sorted(common_keys) if k in fn_map]
            # 加入 stripped 函数（无法匹配的）
            for item in index_items:
                if item.get("binary", "") == variant_key:
                    for fn in item.get("functions", []):
                        if _function_key(fn, match_by) is None:
                            kept_funcs.append(fn)
                    break
            kept_items.append({"binary": variant_key, "functions": kept_funcs})

    # 加入无法分组的条目
    kept_items.extend(ungroupable)

    return kept_items, stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="同源变体交叉过滤：只保留所有变体共有的函数")
    parser.add_argument("--input", "-i", required=True, help="输入索引路径")
    parser.add_argument("--output", "-o", required=True, help="输出索引路径")
    parser.add_argument(
        "--match-by",
        choices=("name", "name_entry"),
        default="name",
        help="匹配方式：name 按函数名（默认，跨架构友好）；name_entry 名+地址联合",
    )
    parser.add_argument(
        "--min-pcode-len",
        type=int,
        default=0,
        help="最低 pcode 长度过滤（0=不过滤；通常先跑 filter_index_by_pcode_len）",
    )
    parser.add_argument(
        "--min-variants",
        type=int,
        default=2,
        help="最少变体数才进行交叉过滤（默认 2；单变体 project 保留全部）",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.5,
        help="函数保留的最低变体覆盖率（默认 0.5 = 出现在 ≥50%% 变体中就保留；1.0 = 全交集）",
    )
    parser.add_argument(
        "--project-root",
        default=PROJECT_ROOT,
        help="项目根目录",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"错误: 输入索引不存在 {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        index_items = [raw] if isinstance(raw, dict) else []
    else:
        index_items = raw

    if not index_items:
        print("错误: 输入索引为空", file=sys.stderr)
        sys.exit(1)

    project_root = os.path.abspath(args.project_root)

    # 先做 pcode len 预过滤（如果指定了）
    if args.min_pcode_len > 0:
        filtered: List[dict] = []
        for item in index_items:
            kept_funcs = []
            for fn in item.get("functions", []):
                ntok = _count_pcode_tokens(fn)
                if ntok >= args.min_pcode_len:
                    kept_funcs.append(fn)
            if kept_funcs:
                filtered.append({"binary": item.get("binary", ""), "functions": kept_funcs})
        index_items = filtered
        logger.info("pcode len >= %d 预过滤后: %d 个二进制, %d 个函数",
                     args.min_pcode_len, len(index_items),
                     sum(len(x["functions"]) for x in index_items))

    kept_items, stats = filter_common_functions(
        index_items,
        match_by=args.match_by,
        min_variants=args.min_variants,
        min_ratio=args.min_ratio,
        project_root=project_root,
    )

    out_path = os.path.abspath(args.output)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept_items, f, indent=2, ensure_ascii=False)

    total_funcs = sum(len(x["functions"]) for x in kept_items)
    print(f"\n{'='*60}")
    print(f"同源交叉过滤完成")
    print(f"  项目数:       {stats['total_projects']}")
    print(f"  单变体项目:   {stats['single_variant_projects']} (保留全部)")
    print(f"  多变体项目:   {stats['multi_variant_projects']}")
    print(f"  受影响项目:   {stats['projects_affected']} (有函数被过滤)")
    print(f"  原始函数:     {stats['total_original_functions']}")
    print(f"  保留函数:     {stats['total_kept_functions']}")
    print(f"  丢弃函数:     {stats['total_dropped_functions']}")
    if stats["total_original_functions"] > 0:
        drop_pct = 100.0 * stats["total_dropped_functions"] / stats["total_original_functions"]
        print(f"  过滤率:       {drop_pct:.1f}%")
    print(f"  输出:         {out_path} ({len(kept_items)} 个二进制, {total_funcs} 个函数)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()