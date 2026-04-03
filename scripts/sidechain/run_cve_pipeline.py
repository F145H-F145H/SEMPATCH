#!/usr/bin/env python3
"""
侧链兼容入口：CVE 匹配生产线。

唯一推荐入口：项目根目录 `python sempatch.py match ...`
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from cli.cve_match import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
