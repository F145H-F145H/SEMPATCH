#!/usr/bin/env python3
"""
BinCodex 数据集获取占位。

BinCodex (TBench 2024) 综合性多层级数据集，公开下载链接待论文/附录确认。
获取建议：
  - 查阅论文及补充材料: https://www.sciengine.com/doi/10.1016/j.tbench.2024.100163
  - 联系论文作者获取数据
  - 使用 BinKit 替代: python scripts/download_binkit.py
  - 其他基准: FirmVulLinker (github.com/a101e-lab/FirmVulLinker)；自备 ELF 见 docs/VULNERABILITY_LIBRARY.md
"""
import sys


def main():
    print("BinCodex 下载暂未实现（数据链接待论文/附录确认）。")
    print("  论文: https://www.sciengine.com/doi/10.1016/j.tbench.2024.100163")
    print("  替代: python scripts/download_binkit.py  # BinKit BCSA Benchmark")
    sys.exit(1)


if __name__ == "__main__":
    main()
