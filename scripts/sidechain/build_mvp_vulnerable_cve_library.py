#!/usr/bin/env python3
"""
从 examples/mvp_vulnerable/vulnerable 构建带 CVE 元数据的库特征与 SAFE 嵌入，
供 scripts/demo_cve_match.py 使用。function_id 规则与 demo_cve_match 一致。

用法:
  PYTHONPATH=src python scripts/build_mvp_vulnerable_cve_library.py -o output/mvp_cve_lib
  # 与 run_cve_pipeline 一致：统一 CVE + 写出 library_safe_embeddings.json
  PYTHONPATH=src python scripts/build_mvp_vulnerable_cve_library.py -o data/cve_quick_demo \\
    --unified-cve CVE-2018-10822 --write-library-safe-embeddings
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        venv_py = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
        hint = (
            f"  PYTHONPATH=src {venv_py} scripts/build_mvp_vulnerable_cve_library.py ...\n"
            if os.path.isfile(venv_py)
            else "  请先创建 venv 并 pip install -r requirements.txt（至少含 torch）。\n"
        )
        print(
            "错误: 构建 SAFE 嵌入需要 PyTorch，当前解释器未安装 torch。\n" + hint,
            file=sys.stderr,
            end="",
        )
        sys.exit(1)


# 与 scripts/demo_cve_match.py 中 _function_id 一致
def _norm_entry(entry: str) -> str:
    s = (entry or "").strip().lower()
    if s.startswith("0x"):
        return s
    return f"0x{s}" if s else "0x0"


def _function_id(binary_rel: str, entry: str) -> str:
    return f"{binary_rel}|{_norm_entry(entry)}"


def _load_lsir_raw(binary_abs: str):
    from utils.ghidra_runner import peek_binary_cache, run_ghidra_analysis

    raw = peek_binary_cache(binary_abs)
    if raw is None:
        tmp = tempfile.mkdtemp(prefix="mvp_cve_lib_")
        try:
            raw = run_ghidra_analysis(
                binary_path=binary_abs,
                output_dir=tmp,
                project_name="MvpCveLib",
                script_name="extract_lsir_raw.java",
                script_output_name="lsir_raw.json",
                return_dict=True,
            )
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 MVP 漏洞库特征 + 嵌入（含 CVE）")
    parser.add_argument(
        "--binary",
        default=os.path.join(PROJECT_ROOT, "examples", "mvp_vulnerable", "vulnerable"),
        help="vulnerable ELF 绝对或相对项目根的路径",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="输出目录（写入 library_features.json、library_embeddings.json）",
    )
    parser.add_argument(
        "--unified-cve",
        default=None,
        help="若指定，则库中每条 vuln_* 均带同一 CVE 列表（如 CVE-2018-10822，供「正常」漏洞库演示）",
    )
    parser.add_argument(
        "--write-library-safe-embeddings",
        action="store_true",
        help="额外写入 library_safe_embeddings.json（与 run_cve_pipeline / two_stage 默认文件名一致）",
    )
    args = parser.parse_args()
    _require_torch()

    binary_abs = os.path.abspath(
        args.binary if os.path.isabs(args.binary) else os.path.join(PROJECT_ROOT, args.binary)
    )
    if not os.path.isfile(binary_abs):
        print(f"错误: 二进制不存在 {binary_abs}", file=sys.stderr)
        sys.exit(1)

    binary_rel = os.path.relpath(binary_abs, PROJECT_ROOT)
    from utils.feature_extractors.multimodal_extraction import extract_multimodal_from_lsir_raw

    raw = _load_lsir_raw(binary_abs)
    funcs = raw.get("functions") or []
    if not funcs:
        print("错误: lsir_raw 无 functions", file=sys.stderr)
        sys.exit(1)

    # 仅库化名称以 vuln_ 开头的符号（与 MVP 示例一致）
    default_cve_by_name = {
        "vuln_copy": ["CVE-MVP-0001"],
        "vuln_loop": ["CVE-MVP-0002"],
    }
    unified = (args.unified_cve or "").strip()
    if unified:
        cve_by_name = {n: [unified] for n in default_cve_by_name}
    else:
        cve_by_name = default_cve_by_name
    library_features: dict = {}
    manifest: dict[str, dict] = {}

    for fn in funcs:
        name = (fn.get("name") or "").strip()
        if not name.startswith("vuln_") or name not in cve_by_name:
            continue
        entry = fn.get("entry", "")
        if not entry:
            continue
        fid = _function_id(binary_rel, entry)
        try:
            mm = extract_multimodal_from_lsir_raw(funcs, entry)
        except (ValueError, RuntimeError) as e:
            print(f"警告: 跳过 {name} entry={entry}: {e}", file=sys.stderr)
            continue
        library_features[fid] = mm
        manifest[fid] = {"name": name, "cve": cve_by_name[name]}

    if not library_features:
        print("错误: 未提取到任何 vuln_* 库函数特征", file=sys.stderr)
        sys.exit(1)

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
    print("运行:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    with open(emb_raw_path, encoding="utf-8") as f:
        emb_doc = json.load(f)
    items = emb_doc.get("functions") or []
    for item in items:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("function_id", ""))
        info = manifest.get(fid)
        if info:
            item["name"] = info["name"]
            item["cve"] = list(info["cve"])
    emb_doc_out = {"functions": items}
    with open(emb_path, "w", encoding="utf-8") as f:
        json.dump(emb_doc_out, f, indent=2, ensure_ascii=False)

    safe_emb_path = os.path.join(out_dir, "library_safe_embeddings.json")
    if args.write_library_safe_embeddings:
        with open(safe_emb_path, "w", encoding="utf-8") as f:
            json.dump(emb_doc_out, f, indent=2, ensure_ascii=False)

    print("已写入:", feat_path, flush=True)
    print("已写入:", emb_path, flush=True)
    if args.write_library_safe_embeddings:
        print("已写入:", safe_emb_path, flush=True)
    print("库 function_id 与 CVE:", flush=True)
    for fid, info in sorted(manifest.items()):
        print(f"  {fid} -> {info['name']} {info['cve']}", flush=True)


if __name__ == "__main__":
    main()
