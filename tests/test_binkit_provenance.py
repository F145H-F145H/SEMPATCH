"""binkit_provenance 启发式同源解析。"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def test_derive_project_id_strips_opt_compiler() -> None:
    from utils.binkit_provenance import derive_project_id

    assert derive_project_id("data/foo_O2_gcc.elf") == "foo"
    assert derive_project_id("data/foo_O3_clang.elf") == "foo"


def test_parse_binary_provenance_arch_and_variant() -> None:
    from utils.binkit_provenance import classify_pair_relation, parse_binary_provenance

    pid1, h1 = parse_binary_provenance("build/x86_64/netifd_O2_gcc.elf")
    pid2, h2 = parse_binary_provenance("build/aarch64/netifd_O2_gcc.elf")
    assert pid1 == pid2 == "netifd"
    assert h1.arch == "x86_64"
    assert h2.arch == "aarch64"
    assert classify_pair_relation(h1, h2) == "cross_arch"


def test_summarize_provenance() -> None:
    from utils.binkit_provenance import summarize_provenance

    s = summarize_provenance(
        [
            "a/x86_64/f_O2_gcc.elf",
            "a/x86_64/f_O3_gcc.elf",
        ]
    )
    assert s["total_binaries"] == 2
    assert s["unique_project_id"] == 1
    assert s["binaries_with_arch_hint"] == 2
