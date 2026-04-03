#!/usr/bin/env python3
"""转发至 scripts/sidechain/show_lsir_dfg.py"""
import os
import runpy
import sys

_root = os.path.dirname(os.path.abspath(__file__))
_path = os.path.join(_root, "sidechain", "show_lsir_dfg.py")

if __name__ == "__main__":
    runpy.run_path(_path, run_name="__main__")
