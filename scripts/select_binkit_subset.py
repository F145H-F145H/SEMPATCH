#!/usr/bin/env python3
"""
Phase 0: 从大量 BinKit ELF 中智能筛选训练子集。
- 无需 Ghidra，纯文件名解析
- 按 project_id 分组，优先保留多变体项目
- 输出选中子集的 binary 索引 JSON
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from utils.binkit_provenance import parse_binary_provenance, classify_pair_relation, VariantHints


def scan_binaries(scan_root: str) -> list[dict]:
    """递归扫描目录，返回 [{binary_abs, binary_rel, project_id, arch, compiler, opt}, ...]"""
    results = []
    for dirpath, _dirnames, filenames in os.walk(scan_root):
        for fn in sorted(filenames):
            fp = os.path.join(dirpath, fn)
            if not os.path.isfile(fp):
                continue
            # 快速 ELF 检查
            try:
                with open(fp, "rb") as f:
                    if f.read(4) != b"\x7fELF":
                        continue
            except (OSError, IOError):
                continue
            rel = os.path.relpath(fp, scan_root)
            pid, hints = parse_binary_provenance(rel)
            results.append({
                "binary": fp,
                "binary_rel": rel,
                "project_id": pid,
                "arch": hints.arch,
                "compiler": hints.compiler,
                "opt": hints.opt,
                "fingerprint": hints.fingerprint(),
            })
    return results


def select_subset(all_binaries: list[dict], *, max_total: int = 800, min_variants: int = 2, max_variants: int = 5) -> list[dict]:
    """
    策略:
    1. 按 project_id 分组
    2. 优先选择变体数 >= min_variants 的项目（有正对）
    3. 每个项目保留 max_variants 个变体，优先覆盖不同 (arch, compiler, opt) 组合
    4. 总量不超过 max_total
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for b in all_binaries:
        groups[b["project_id"]].append(b)

    # 统计
    proj_sizes = [(pid, len(items)) for pid, items in groups.items()]
    proj_sizes.sort(key=lambda x: -x[1])

    multi_variant = [(pid, n) for pid, n in proj_sizes if n >= min_variants]
    single_variant = [(pid, n) for pid, n in proj_sizes if n < min_variants]

    print(f"  项目总数: {len(groups)}")
    print(f"  多变体项目 (>={min_variants}): {len(multi_variant)} (共 {sum(n for _,n in multi_variant)} 个二进制)")
    print(f"  单变体项目: {len(single_variant)} (共 {sum(n for _,n in single_variant)} 个二进制)")

    selected = []
    selected_projects = 0

    # 优先选多变体项目
    for pid, _ in proj_sizes:
        if len(selected) >= max_total:
            break
        items = groups[pid]
        if len(items) < min_variants:
            continue
        # 每个项目选 max_variants 个，优先不同 fingerprint
        chosen = _pick_diverse_variants(items, max_variants)
        remaining_slots = max_total - len(selected)
        chosen = chosen[:remaining_slots]
        selected.extend(chosen)
        selected_projects += 1

    # 如果还有余量，补充单变体项目（增加负对多样性）
    if len(selected) < max_total:
        for pid, _ in single_variant:
            if len(selected) >= max_total:
                break
            items = groups[pid]
            selected.append(items[0])

    # 重新统计正对数
    selected_groups = defaultdict(list)
    for b in selected:
        selected_groups[b["project_id"]].append(b)
    total_positive_pairs = sum(
        n * (n - 1) // 2 for n in (len(v) for v in selected_groups.values()) if n >= 2
    )

    # 架构/编译器覆盖统计
    arch_counter = Counter(b["arch"] for b in selected if b["arch"])
    compiler_counter = Counter(b["compiler"] for b in selected if b["compiler"])
    opt_counter = Counter(b["opt"] for b in selected if b["opt"])

    print(f"\n  选中: {len(selected)} 个二进制, {selected_projects} 个多变体项目")
    print(f"  潜在正对数 (同项目函数名匹配): {total_positive_pairs}")
    print(f"  架构分布: {dict(arch_counter)}")
    print(f"  编译器分布: {dict(compiler_counter)}")
    print(f"  优化级分布: {dict(opt_counter)}")

    return selected, {
        "total_binaries": len(selected),
        "multi_variant_projects": selected_projects,
        "potential_positive_pairs": total_positive_pairs,
        "arch_coverage": dict(arch_counter),
        "compiler_coverage": dict(compiler_counter),
        "opt_coverage": dict(opt_counter),
    }


def _pick_diverse_variants(items: list[dict], k: int) -> list[dict]:
    """从同一项目的多个变体中选 k 个，优先覆盖不同 fingerprint。"""
    if len(items) <= k:
        return list(items)
    chosen = []
    seen_fp = set()
    # 第一轮：选不同 fingerprint
    for item in items:
        fp = item.get("fingerprint", "")
        if fp and fp not in seen_fp:
            chosen.append(item)
            seen_fp.add(fp)
            if len(chosen) >= k:
                return chosen
    # 第二轮：填充剩余
    for item in items:
        if item not in chosen:
            chosen.append(item)
            if len(chosen) >= k:
                break
    return chosen


def main():
    parser = argparse.ArgumentParser(description="从大量 BinKit ELF 中智能筛选训练子集")
    parser.add_argument("--scan-root", required=True, help="BinKit 二进制根目录（递归扫描）")
    parser.add_argument("-o", "--output", default="data/binkit_subset_index.json", help="输出子集索引")
    parser.add_argument("--max-total", type=int, default=800, help="最多选中二进制数")
    parser.add_argument("--min-variants", type=int, default=2, help="项目最少变体数（低于此仅作负对补充）")
    parser.add_argument("--max-variants", type=int, default=5, help="每项目最多保留变体数")
    parser.add_argument("--stats-only", action="store_true", help="仅输出统计信息，不写文件")
    args = parser.parse_args()

    scan_root = os.path.abspath(args.scan_root)
    if not os.path.isdir(scan_root):
        print(f"错误: 目录不存在: {scan_root}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/3] 扫描 ELF: {scan_root}")
    all_bins = scan_binaries(scan_root)
    print(f"  发现 {len(all_bins)} 个 ELF")

    if not all_bins:
        print("错误: 未发现 ELF 文件", file=sys.stderr)
        sys.exit(1)

    # 全局统计
    all_groups = defaultdict(list)
    for b in all_bins:
        all_groups[b["project_id"]].append(b)
    all_multi = sum(1 for v in all_groups.values() if len(v) >= 2)
    total_possible_pairs = sum(
        n * (n - 1) // 2 for n in (len(v) for v in all_groups.values()) if n >= 2
    )
    print(f"\n  全局统计:")
    print(f"    独立项目数: {len(all_groups)}")
    print(f"    多变体项目 (>=2): {all_multi}")
    print(f"    最大潜在正对数: {total_possible_pairs}")

    # Top-20 最大项目
    proj_sorted = sorted(all_groups.items(), key=lambda x: -len(x[1]))
    print(f"\n  Top-20 最大项目 (正对数最多):")
    for pid, items in proj_sorted[:20]:
        fps = set(x.get("fingerprint","") for x in items)
        pairs = len(items) * (len(items)-1) // 2
        print(f"    {pid}: {len(items)} 变体, {len(fps)} 独立配置, {pairs} 正对")

    if args.stats_only:
        return

    print(f"\n[2/3] 筛选子集 (max={args.max_total})")
    selected, stats = select_subset(
        all_bins,
        max_total=args.max_total,
        min_variants=args.min_variants,
        max_variants=args.max_variants,
    )

    print(f"\n[3/3] 写入: {args.output}")
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    output = [{"binary": b["binary"]} for b in selected]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  完成: {out_path}")

    # 同时写入统计文件
    stats_path = out_path.replace(".json", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  统计: {stats_path}")


if __name__ == "__main__":
    main()