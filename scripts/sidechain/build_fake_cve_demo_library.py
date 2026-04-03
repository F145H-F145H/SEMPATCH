#!/usr/bin/env python3
"""
从「二进制路径 + 漏洞函数名（可选入口）+ 伪 CVE 标签」清单构建自给自足的小型漏洞库：
与正式流水线一致：Ghidra → lsir_raw（binary_cache）→ multimodal → SAFE 嵌入。

产物（默认写入 --output-dir）：
  - library_features.json       供 TwoStage 精排 / run_cve_pipeline
  - library_safe_embeddings.json 供粗筛 + demo_cve_match 的 cve 查表

manifest JSON 格式（数组或 {"entries": [...]}）每项：
  - binary: 相对项目根或绝对路径
  - function_name: Ghidra 导出符号名（与 lsir_raw 中 name 一致）
  - cve: 字符串或字符串列表（如 FAKE-CVE-0001、CVE-2099-00001 等）
  - entry: 可选，十六进制入口；省略则按 function_name 唯一匹配，否则报错

用法:
  PYTHONPATH=src python scripts/build_fake_cve_demo_library.py \\
    --manifest data/my_fake_cve_manifest.json -o output/fake_cve_lib \\
    --write-library-safe-embeddings
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        venv_py = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
        hint = (
            f"  PYTHONPATH=src {venv_py} scripts/build_fake_cve_demo_library.py ...\n"
            if os.path.isfile(venv_py)
            else "  请先安装 torch（requirements.txt）。\n"
        )
        print("错误: 需要 PyTorch。\n" + hint, file=sys.stderr, end="")
        sys.exit(1)


def _norm_entry(entry: str) -> str:
    s = (entry or "").strip().lower()
    if s.startswith("0x"):
        return s
    return f"0x{s}" if s else "0x0"


def _function_id(binary_rel: str, entry: str) -> str:
    return f"{binary_rel}|{_norm_entry(entry)}"


def _load_manifest(path: str) -> list[dict[str, Any]]:
    with open(os.path.abspath(path), encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "entries" in data:
        raw = data["entries"]
    elif isinstance(data, list):
        raw = data
    else:
        raise ValueError("manifest 应为 JSON 数组或 {\"entries\": [...]}")
    if not isinstance(raw, list) or not raw:
        raise ValueError("manifest 条目列表为空")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"条目 {i} 不是对象")
        out.append(row)
    return out


def _coerce_cve(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    if isinstance(raw, list):
        acc: list[str] = []
        for x in raw:
            if isinstance(x, str) and x.strip():
                acc.append(x.strip())
        return acc
    return []


def _load_lsir_raw(binary_abs: str):
    from utils.ghidra_runner import peek_binary_cache, run_ghidra_analysis

    raw = peek_binary_cache(binary_abs)
    if raw is None:
        tmp = tempfile.mkdtemp(prefix="fake_cve_lib_")
        try:
            raw = run_ghidra_analysis(
                binary_path=binary_abs,
                output_dir=tmp,
                project_name="FakeCveLib",
                script_name="extract_lsir_raw.java",
                script_output_name="lsir_raw.json",
                return_dict=True,
            )
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
    return raw


def _resolve_entry(
    funcs: list[dict[str, Any]],
    function_name: str,
    entry_override: str | None,
    name_ci: bool,
) -> tuple[str, str]:
    """返回 (norm_entry, resolved_symbol_name)。"""
    if entry_override and str(entry_override).strip():
        ne = _norm_entry(str(entry_override).strip())
        for f in funcs:
            if _norm_entry(f.get("entry", "")) == ne:
                return ne, str(f.get("name") or function_name)
        raise ValueError(f"未找到 entry={ne} 的函数")

    key = function_name.strip()
    if not key:
        raise ValueError("function_name 为空且未提供 entry")

    def name_ok(fn_name: str) -> bool:
        n = (fn_name or "").strip()
        if name_ci:
            return n.lower() == key.lower()
        return n == key

    matches = [f for f in funcs if name_ok(str(f.get("name") or ""))]
    if not matches:
        raise ValueError(f"未找到函数名 {key!r}（可与 Ghidra lsir_raw 中 name 核对）")
    if len(matches) > 1:
        raise ValueError(
            f"函数名 {key!r} 匹配到 {len(matches)} 个符号，请在 manifest 中为该项填写唯一 entry"
        )
    entry = matches[0].get("entry", "")
    return _norm_entry(str(entry)), str(matches[0].get("name") or key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 manifest（二进制+漏洞函数名+CVE）构建 library_features + SAFE 嵌入库"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="JSON：entries 列表，见文件头注释",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="SAFE 检查点路径（可选；默认随机初始化基线）",
    )
    parser.add_argument(
        "--write-library-safe-embeddings",
        action="store_true",
        help="写出 library_safe_embeddings.json（与 run_cve_pipeline 默认文件名一致）",
    )
    parser.add_argument(
        "--name-ci",
        action="store_true",
        help="按函数名匹配时忽略大小写",
    )
    args = parser.parse_args()
    _require_torch()

    rows = _load_manifest(args.manifest)
    from utils.feature_extractors.multimodal_extraction import extract_multimodal_from_lsir_raw

    library_features: dict[str, Any] = {}
    manifest_by_fid: dict[str, dict[str, Any]] = {}

    for i, row in enumerate(rows):
        b = (row.get("binary") or "").strip()
        if not b:
            print(f"错误: 条目 {i} 缺少 binary", file=sys.stderr)
            sys.exit(1)
        binary_abs = os.path.abspath(b if os.path.isabs(b) else os.path.join(PROJECT_ROOT, b))
        if not os.path.isfile(binary_abs):
            print(f"错误: 二进制不存在 {binary_abs}", file=sys.stderr)
            sys.exit(1)
        binary_rel = (
            b if not os.path.isabs(b) else os.path.relpath(binary_abs, PROJECT_ROOT)
        )
        if binary_rel.startswith(".."):
            binary_rel = binary_abs

        fn_name = str(row.get("function_name") or row.get("name") or "").strip()
        entry_opt = row.get("entry")
        if entry_opt is not None and str(entry_opt).strip():
            entry_opt = str(entry_opt).strip()
        else:
            entry_opt = None

        cve_list = _coerce_cve(row.get("cve"))
        if not cve_list:
            print(f"错误: 条目 {i} 缺少 cve（字符串或列表）", file=sys.stderr)
            sys.exit(1)

        raw = _load_lsir_raw(binary_abs)
        funcs = raw.get("functions") or []
        if not funcs:
            print(f"错误: {binary_rel} lsir_raw 无 functions", file=sys.stderr)
            sys.exit(1)

        try:
            entry, sym = _resolve_entry(
                funcs, fn_name, entry_opt, name_ci=args.name_ci
            )
        except ValueError as e:
            print(f"错误 条目{i} {binary_rel}: {e}", file=sys.stderr)
            sys.exit(1)

        fid = _function_id(binary_rel, entry)
        try:
            mm = extract_multimodal_from_lsir_raw(funcs, entry)
        except (ValueError, RuntimeError) as e:
            print(f"错误 特征提取 {fid}: {e}", file=sys.stderr)
            sys.exit(1)

        if fid in library_features:
            print(f"错误: 重复的 function_id {fid}（同一二进制同一入口只能对应一条 CVE 条目）", file=sys.stderr)
            sys.exit(1)
        library_features[fid] = mm
        manifest_by_fid[fid] = {
            "symbol_name": sym,
            "cve": list(cve_list),
        }

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    feat_path = os.path.join(out_dir, "library_features.json")
    emb_raw_path = os.path.join(out_dir, "library_embeddings.raw.json")
    emb_path = os.path.join(out_dir, "library_embeddings.json")

    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(library_features, f, ensure_ascii=False)

    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "scripts", "build_embeddings_db.py"),
        "--features-file",
        feat_path,
        "--model",
        "safe",
        "-o",
        emb_raw_path,
    ]
    if args.model_path:
        cmd.extend(["--model-path", os.path.abspath(args.model_path)])
    print("运行:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    with open(emb_raw_path, encoding="utf-8") as f:
        emb_doc = json.load(f)
    items = emb_doc.get("functions") or []
    for item in items:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("function_id", ""))
        info = manifest_by_fid.get(fid)
        if info:
            item["name"] = info["symbol_name"]
            item["cve"] = list(info["cve"])
    emb_out = {"functions": items}
    with open(emb_path, "w", encoding="utf-8") as f:
        json.dump(emb_out, f, indent=2, ensure_ascii=False)

    if args.write_library_safe_embeddings:
        safe_path = os.path.join(out_dir, "library_safe_embeddings.json")
        with open(safe_path, "w", encoding="utf-8") as f:
            json.dump(emb_out, f, indent=2, ensure_ascii=False)

    print("已写入:", feat_path, flush=True)
    print("已写入:", emb_path, flush=True)
    if args.write_library_safe_embeddings:
        print("已写入:", os.path.join(out_dir, "library_safe_embeddings.json"), flush=True)
    print("function_id → cve:", flush=True)
    for fid in sorted(manifest_by_fid.keys()):
        print(f"  {fid} -> {manifest_by_fid[fid]['cve']}", flush=True)


if __name__ == "__main__":
    main()
