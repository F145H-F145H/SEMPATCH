#!/usr/bin/env python3
"""转发至 scripts/sidechain/train_multimodal.py；产品入口: python sempatch.py match"""
import os
import runpy
import sys

_root = os.path.dirname(os.path.abspath(__file__))
_base = os.path.basename(__file__)
_path = os.path.join(_root, "sidechain", _base)

if __name__ == "__main__":
    runpy.run_path(_path, run_name="__main__")
else:
    _g = runpy.run_path(_path, run_name="sempatch_sidechain")
    _m = sys.modules[__name__]
    for _k, _v in _g.items():
        if _k in ("__builtins__", "__loader__", "__spec__", "__package__", "__cached__"):
            continue
        if _k.startswith("__") and _k not in ("__doc__",):
            continue
        setattr(_m, _k, _v)
