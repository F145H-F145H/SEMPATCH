#!/usr/bin/env python3
"""
从任意目录递归扫描可分析二进制（.elf / .bin / .so），生成供 build_binkit_index.py
`--from-index-file` 使用的索引 JSON（每项含 binary 相对项目根或绝对路径、functions: []）。

SemPatch 流水线需要 **ELF 等可执行文件**；若某数据集仅提供 .arrow / Parquet 等表格格式，
那是预计算特征或元数据，**不能**替代 Ghidra 提 LSIR，需自行取得对应二进制或改用其他工具链。

用法:
  PYTHONPATH=src python scripts/build_library_binary_index.py --scan-root data/my_vuln_bins
  PYTHONPATH=src python scripts/build_library_binary_index.py --scan-root /abs/path -o data/vuln_index.json
  PYTHONPATH=src python scripts/build_library_binary_index.py --scan-root data/my_vuln_bins --validate-only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_BINARY_EXTS = (".elf", ".bin", ".so")
_CVE_PATTERN = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)


def find_binaries_under(root: str) -> list[tuple[str, str | None]]:
    """递归查找二进制，返回 [(abs_path, path_inferred_cve_or_none)]。"""
    results: list[tuple[str, str | None]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext not in _BINARY_EXTS:
                continue
            abspath = os.path.join(dirpath, f)
            if not os.path.isfile(abspath):
                continue
            inferred: str | None = None
            for part in (dirpath, f, abspath):
                m = _CVE_PATTERN.search(str(part))
                if m:
                    inferred = m.group(0).upper()
                    break
            results.append((abspath, inferred))
    return results


def load_cve_mapping_sidecar(scan_root: str) -> dict[str, str]:
    """读取 scan_root/cve_mapping.json（若存在），供后续 annotate_library_embeddings_cve 使用。"""
    path = os.path.join(scan_root, "cve_mapping.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[str, str] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if k == "mappings":
                continue
            if isinstance(v, str) and _CVE_PATTERN.match(v):
                out[str(k)] = v.upper()
        mappings = data.get("mappings")
        if isinstance(mappings, list):
            for m in mappings:
                if not isinstance(m, dict):
                    continue
                bin_rel = m.get("binary", "")
                cve = m.get("cve", "")
                name = m.get("name", "")
                entry = m.get("entry", "")
                if cve and _CVE_PATTERN.match(str(cve)):
                    key = f"{bin_rel}:{name or entry}" if (name or entry) else str(bin_rel)
                    out[key] = str(cve).upper()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="扫描目录生成漏洞库/评估用二进制清单索引（非数据集绑定）"
    )
    parser.add_argument(
        "--scan-root",
        required=True,
        help="包含 .elf/.bin/.so 的根目录",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "data", "vuln_library_binary_index.json"),
        help="输出索引 JSON 路径",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="仅检查扫描结果，不写文件；无二进制时 exit 2",
    )
    args = parser.parse_args()

    scan_root = os.path.abspath(args.scan_root)
    if not os.path.isdir(scan_root):
        print(f"错误: 目录不存在 {scan_root}", file=sys.stderr)
        print("", file=sys.stderr)
        print("说明:", file=sys.stderr)
        print("  - 本脚本只索引可执行 ELF/共享库等，供 Ghidra 分析。", file=sys.stderr)
        print("  - 若数据集仅有 .arrow/.parquet，需另寻官方二进制分发或自建样本。", file=sys.stderr)
        sys.exit(1)

    binaries = find_binaries_under(scan_root)
    sidecar = load_cve_mapping_sidecar(scan_root)

    if args.validate_only:
        print(f"二进制库准备度: scan_root={scan_root}")
        print(f"  扫描到二进制: {len(binaries)} 个 (.elf/.bin/.so)")
        print(f"  cve_mapping.json 键数: {len(sidecar)}")
        if not binaries:
            print("  状态: 未找到可分析二进制。", file=sys.stderr)
            sys.exit(2)
        print("  状态: 可去掉 --validate-only 写入索引 JSON")
        sys.exit(0)

    if not binaries:
        print(f"警告: 未在 {scan_root} 下找到任何 .elf/.bin/.so", file=sys.stderr)

    index: list[dict] = []
    for abspath, _inferred in sorted(binaries):
        rel = os.path.relpath(abspath, PROJECT_ROOT)
        if rel.startswith(".."):
            rel = abspath
        index.append({"binary": rel, "functions": []})

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"已写入 {out_path}，共 {len(index)} 个二进制")
    if sidecar:
        print(f"提示: 已检测到 cve_mapping.json（{len(sidecar)} 条）；嵌入生成后可用 annotate_library_embeddings_cve.py 写入 cve 字段")


if __name__ == "__main__":
    main()
