#!/usr/bin/env python3
"""转发至 scripts/sidechain/orchestrate_binkit_two_stage_train.py"""
import os
import runpy
import sys

_root = os.path.dirname(os.path.abspath(__file__))
_base = os.path.basename(__file__)
_path = os.path.join(_root, "sidechain", _base)

if __name__ == "__main__":
    runpy.run_path(_path, run_name="__main__")
