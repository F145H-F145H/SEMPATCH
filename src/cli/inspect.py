"""匹配结果分析工具：分数分布、过滤统计、阈值扫描。"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional


def load_matches(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def inspect_matches(matches_path: str) -> Dict[str, Any]:
    """解析 matches.json，返回分数分布与过滤统计。"""
    doc = load_matches(matches_path)
    queries = doc.get("queries") or []

    all_scores: List[float] = []
    cve_hits: List[Dict[str, Any]] = []
    filter_modes: Dict[str, int] = {}
    reject_reasons: Dict[str, int] = {}

    for q in queries:
        qid = q.get("query_function_id", "?")
        for c in q.get("candidates") or []:
            score = c.get("similarity", c.get("rerank_score", 0.0))
            if isinstance(score, (int, float)):
                all_scores.append(float(score))
            cves = c.get("cve") or []
            if cves:
                cve_hits.append({"query": qid, "cve": cves, "score": score})

        meta = q.get("filter_meta") or {}
        mode = meta.get("mode", "unknown")
        filter_modes[mode] = filter_modes.get(mode, 0) + 1
        reason = meta.get("reject_reason")
        if reason:
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    all_scores.sort(reverse=True)

    return {
        "total_queries": len(queries),
        "total_candidates": len(all_scores),
        "score_distribution": _score_stats(all_scores),
        "cve_hits": cve_hits,
        "filter_modes": filter_modes,
        "reject_reasons": reject_reasons,
    }


def _score_stats(scores: List[float]) -> Dict[str, Any]:
    if not scores:
        return {"count": 0}
    return {
        "count": len(scores),
        "max": scores[0],
        "min": scores[-1],
        "mean": sum(scores) / len(scores),
        "top_5": scores[:5],
        "above_0.95": sum(1 for s in scores if s >= 0.95),
        "above_0.90": sum(1 for s in scores if s >= 0.90),
        "above_0.80": sum(1 for s in scores if s >= 0.80),
    }


def print_inspect_report(info: Dict[str, Any]) -> None:
    """格式化输出分析报告到 stdout。"""
    print("=" * 60)
    print("Match Inspection Report")
    print("=" * 60)

    print(f"\nQueries: {info['total_queries']}")
    print(f"Total candidates: {info['total_candidates']}")

    dist = info["score_distribution"]
    if dist.get("count", 0) > 0:
        print(f"\nScore distribution:")
        print(f"  Max:  {dist['max']:.6f}")
        print(f"  Min:  {dist['min']:.6f}")
        print(f"  Mean: {dist['mean']:.6f}")
        print(f"  Top-5: {', '.join(f'{s:.6f}' for s in dist['top_5'])}")
        print(f"  Above 0.95: {dist['above_0.95']}")
        print(f"  Above 0.90: {dist['above_0.90']}")
        print(f"  Above 0.80: {dist['above_0.80']}")

    if info["filter_modes"]:
        print(f"\nFilter modes:")
        for mode, count in sorted(info["filter_modes"].items()):
            print(f"  {mode}: {count}")

    if info["reject_reasons"]:
        print(f"\nReject reasons:")
        for reason, count in sorted(info["reject_reasons"].items()):
            print(f"  {reason}: {count}")

    if info["cve_hits"]:
        print(f"\nCVE hits ({len(info['cve_hits'])}):")
        for hit in info["cve_hits"][:10]:
            print(f"  {hit['query']} → {hit['cve']} (score={hit['score']:.6f})")
        if len(info["cve_hits"]) > 10:
            print(f"  ... and {len(info['cve_hits']) - 10} more")

    print()


def threshold_sweep(
    matches_path: str,
    thresholds: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """扫描不同阈值下的匹配数。"""
    if thresholds is None:
        thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]

    doc = load_matches(matches_path)
    queries = doc.get("queries") or []

    results = []
    for t in thresholds:
        matched = 0
        total_cves = 0
        for q in queries:
            q_matched = False
            for c in q.get("candidates") or []:
                score = c.get("similarity", c.get("rerank_score", 0.0))
                if isinstance(score, (int, float)) and float(score) >= t:
                    q_matched = True
                    total_cves += len(c.get("cve") or [])
            if q_matched:
                matched += 1
        results.append({
            "threshold": t,
            "queries_matched": matched,
            "total_cve_hits": total_cves,
        })

    return results


def print_threshold_sweep(sweep: List[Dict[str, Any]]) -> None:
    print("\nThreshold sweep:")
    print(f"  {'Threshold':>10}  {'Queries':>8}  {'CVE hits':>8}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}")
    for r in sweep:
        print(f"  {r['threshold']:>10.2f}  {r['queries_matched']:>8}  {r['total_cve_hits']:>8}")
    print()


def inspect_from_path(matches_path: str) -> None:
    """入口：从路径加载并打印分析报告。"""
    if not os.path.isfile(matches_path):
        print(f"Error: {matches_path} not found", file=sys.stderr)
        sys.exit(1)
    info = inspect_matches(matches_path)
    print_inspect_report(info)
