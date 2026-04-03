#!/usr/bin/env python3
"""
为 SAFE/嵌入 JSON（EmbeddingDict：functions[].name 为 function_id，如 binary_rel|0x401000）
写入或合并 cve 字段，供 demo_cve_match / run_cve_pipeline / eval_bcsd --mode cve 使用。

CVE 来源（可组合）：
  --per-binary-cve JSON：{ "path/to.elf": "CVE-2021-1" 或 ["CVE-A","CVE-B"] }
  --cve-mapping-json：与 build_library_binary_index 旁路的 cve_mapping.json 相同结构；
    支持整二进制条目（仅 binary+cve）或 binary+name/entry 细粒度键
  --infer-from-path：在 function_id 字符串上匹配 CVE-xxxx-xxxxx

用法:
  python scripts/annotate_library_embeddings_cve.py -i lib.raw.json -o lib.json --infer-from-path
  python scripts/annotate_library_embeddings_cve.py -i lib.raw.json -o lib.json --per-binary-cve data/cve_by_binary.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

_CVE_PATTERN = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)


def _norm_entry(entry: str) -> str:
    s = (entry or "").strip().lower()
    if s.startswith("0x"):
        return s
    return f"0x{s}" if s else "0x0"


def _parse_mapping_file(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, str] = {}
    if not isinstance(data, dict):
        return out
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


def _per_binary_dict(raw: dict) -> dict[str, list[str]]:
    """path -> list of CVE strings"""
    m: dict[str, list[str]] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, str) and _CVE_PATTERN.match(v):
            m.setdefault(k, []).append(v.upper())
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, str) and _CVE_PATTERN.match(x):
                    m.setdefault(k, []).append(x.upper())
    return m


def _function_id_parts(fid: str) -> tuple[str, str]:
    if "|" not in fid:
        return fid, ""
    binary_part, entry = fid.rsplit("|", 1)
    return binary_part, _norm_entry(entry)


def main() -> None:
    parser = argparse.ArgumentParser(description="为库嵌入 JSON 写入 cve 元数据")
    parser.add_argument("-i", "--input", required=True, help="输入嵌入 JSON")
    parser.add_argument("-o", "--output", required=True, help="输出路径")
    parser.add_argument(
        "--per-binary-cve",
        default=None,
        help='JSON：{"相对或绝对路径.elf": "CVE-..." 或 ["CVE-..."]}',
    )
    parser.add_argument(
        "--cve-mapping-json",
        default=None,
        help="cve_mapping.json（与 build_library_binary_index 旁路格式一致）",
    )
    parser.add_argument(
        "--infer-from-path",
        action="store_true",
        help="从每条 function_id 字符串用正则提取 CVE",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        blob = json.load(f)
    funcs = blob.get("functions")
    if not isinstance(funcs, list):
        print("错误: 输入 JSON 缺少 functions 数组", file=sys.stderr)
        sys.exit(1)

    per_bin: dict[str, list[str]] = {}
    if args.per_binary_cve:
        with open(args.per_binary_cve, encoding="utf-8") as f:
            per_bin = _per_binary_dict(json.load(f))

    mapping_flat: dict[str, str] = {}
    if args.cve_mapping_json:
        mapping_flat = _parse_mapping_file(args.cve_mapping_json)

    for item in funcs:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("function_id") or ""
        if not isinstance(name, str):
            name = str(name)
        cves: set[str] = set()
        existing = item.get("cve")
        if isinstance(existing, str) and _CVE_PATTERN.match(existing):
            cves.add(existing.upper())
        elif isinstance(existing, list):
            for x in existing:
                if isinstance(x, str) and _CVE_PATTERN.match(x):
                    cves.add(x.upper())

        if args.infer_from_path:
            for m in _CVE_PATTERN.finditer(name):
                cves.add(m.group(0).upper())

        binary_part, entry_hex = _function_id_parts(name)
        # per-binary file: try exact binary_part, normpath
        for key, lst in per_bin.items():
            cand = {binary_part, os.path.normpath(binary_part)}
            if key in cand or os.path.normpath(key) in cand:
                cves.update(lst)

        # mapping: whole-binary keys
        for mk, mcve in mapping_flat.items():
            if ":" not in mk:
                if mk == binary_part or os.path.normpath(mk) == os.path.normpath(binary_part):
                    cves.add(mcve)
            else:
                mb, rest = mk.split(":", 1)
                if mb != binary_part and os.path.normpath(mb) != os.path.normpath(binary_part):
                    continue
                # rest is function name or raw entry
                if rest.lower() == entry_hex.lower():
                    cves.add(mcve)
                # name match on item if present
                fn = item.get("function_name") or item.get("symbol")
                if fn and rest == str(fn):
                    cves.add(mcve)

        item["cve"] = sorted(cves) if cves else []

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, ensure_ascii=False)

    print(f"已写入 {args.output}，共 {len(funcs)} 条")


if __name__ == "__main__":
    main()
