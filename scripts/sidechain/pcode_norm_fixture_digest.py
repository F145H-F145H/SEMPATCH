#!/usr/bin/env python3
"""
阶段 D.2：对固定 fixture 做 P-code 规范化并输出可重复摘要（pcode 条数 + 规范化后 SHA-256）。

用法（项目根目录）：
  PYTHONPATH=src .venv/bin/python scripts/pcode_norm_fixture_digest.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _count_pcode_ops(lsir: dict) -> int:
    n = 0
    for fn in lsir.get("functions") or []:
        if not isinstance(fn, dict):
            continue
        for bb in fn.get("basic_blocks") or []:
            if not isinstance(bb, dict):
                continue
            for inst in bb.get("instructions") or []:
                if isinstance(inst, dict):
                    n += len(inst.get("pcode") or [])
    return n


def main() -> None:
    src_root = ROOT / "src"
    sys.path.insert(0, str(src_root))
    from utils.pcode_normalizer import normalize_lsir_raw

    fixture = ROOT / "tests" / "fixtures" / "lsir_raw_mock.json"
    raw = json.loads(fixture.read_text(encoding="utf-8"))
    before_ops = _count_pcode_ops(raw)
    norm = normalize_lsir_raw(raw, in_place=False)
    after_ops = _count_pcode_ops(norm)
    blob = json.dumps(norm, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    print(f"fixture={fixture.relative_to(ROOT)} pcode_ops_before={before_ops} pcode_ops_after={after_ops}")
    print(f"normalized_sha256={digest}")


if __name__ == "__main__":
    main()
