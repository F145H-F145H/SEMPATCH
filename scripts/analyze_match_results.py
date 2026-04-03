#!/usr/bin/env python3
"""
分析匹配结果：分数分布、过滤统计、阈值扫描。

用法:
  PYTHONPATH=src .venv/bin/python scripts/analyze_match_results.py output/matches.json
  PYTHONPATH=src .venv/bin/python scripts/analyze_match_results.py output/matches.json --sweep 0.8,0.85,0.9,0.95,0.99
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 matches.json 匹配结果")
    parser.add_argument("matches_path", help="matches.json 路径")
    parser.add_argument(
        "--sweep",
        default=None,
        help="逗号分隔的阈值列表（如 0.8,0.85,0.9,0.95,0.99）",
    )
    args = parser.parse_args()

    from cli.inspect import (
        inspect_from_path,
        load_matches,
        print_threshold_sweep,
        threshold_sweep,
    )

    inspect_from_path(args.matches_path)

    if args.sweep:
        thresholds = [float(x.strip()) for x in args.sweep.split(",")]
        sweep = threshold_sweep(args.matches_path, thresholds)
        print_threshold_sweep(sweep)


if __name__ == "__main__":
    main()
