#!/usr/bin/env python3
"""
CVE 库提取器：从「二进制路径 + 漏洞函数名 + CVE 标签」清单构建自给自足的漏洞库。

完整流程：Ghidra → lsir_raw → multimodal 特征 → 训练 SAFE 模型 → SAFE 嵌入。

产物（写入 --output-dir）：
  - library_features.json           多模态特征（TwoStage 精排用）
  - library_safe_embeddings.json    SAFE 粗筛嵌入（带 CVE 标签）
  - library_cve_map.json            function_id → CVE 列表
  - safe_model.pt                   训练后的 SAFE 模型（供查询复用）

manifest JSON 格式（数组）每项：
  - binary: 相对项目根或绝对路径
  - function_name: Ghidra 导出符号名
  - cve: 字符串或字符串列表
  - entry: 可选，十六进制入口

用法:
  PYTHONPATH=src .venv/bin/python scripts/sidechain/extract_cve_library.py \\
    --manifest examples/mvp_library/manifest.json -o data/mvp_library_cve
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
            f"  PYTHONPATH=src {venv_py} scripts/sidechain/extract_cve_library.py ...\n"
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
    return [row for row in raw if isinstance(row, dict)]


def _coerce_cve(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    if isinstance(raw, list):
        return [x.strip() for x in raw if isinstance(x, str) and x.strip()]
    return []


def _load_lsir_raw(binary_abs: str) -> dict[str, Any]:
    from utils.ghidra_runner import peek_binary_cache, run_ghidra_analysis

    raw = peek_binary_cache(binary_abs)
    if raw is None:
        tmp = tempfile.mkdtemp(prefix="extract_cve_lib_")
        try:
            raw = run_ghidra_analysis(
                binary_path=binary_abs,
                output_dir=tmp,
                project_name="ExtractCveLib",
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

    matches = [f for f in funcs if (f.get("name") or "").strip() == key]
    if not matches:
        raise ValueError(f"未找到函数名 {key!r}")
    if len(matches) > 1:
        raise ValueError(
            f"函数名 {key!r} 匹配到 {len(matches)} 个符号，请填写唯一 entry"
        )
    entry = matches[0].get("entry", "")
    return _norm_entry(str(entry)), str(matches[0].get("name") or key)


def _train_safe_model(
    features_path: str,
    output_model_path: str,
    *,
    epochs: int = 2,
    num_pairs: int = 200,
) -> None:
    """在提取的特征上训练 SAFE 模型（使用合成数据对）。"""
    from features.baselines.safe import (
        _SafeEncoder,
        collect_vocab_from_features_file,
        safe_save_model,
        safe_tokenize,
    )
    from features.losses import ContrastiveLoss

    import torch
    from torch.utils.data import DataLoader

    vocab = collect_vocab_from_features_file(features_path)
    vocab_size = max(len(vocab), 256)
    device = torch.device("cpu")
    model = _SafeEncoder(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = ContrastiveLoss(margin=1.0)

    # 生成合成训练对：同函数的不同扰动 = 正对，不同函数 = 负对
    with open(features_path, encoding="utf-8") as f:
        all_feats = json.load(f)
    func_ids = list(all_feats.keys())
    if len(func_ids) < 2:
        print("警告: 特征不足 2 个函数，跳过 SAFE 训练（使用随机初始化）", file=sys.stderr)
        safe_save_model(model, vocab, output_model_path)
        return

    import random
    rng = random.Random(42)
    max_len = 512

    def _augment(mm: dict) -> dict:
        """轻量增强：随机截断序列 token。"""
        seq = mm.get("sequence") or {}
        tokens = list(seq.get("pcode_tokens") or [])
        if len(tokens) > 8:
            keep = rng.randint(max(4, len(tokens) // 2), len(tokens))
            tokens = tokens[:keep]
        return {**mm, "sequence": {**seq, "pcode_tokens": tokens}}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        count = 0
        for _ in range(num_pairs):
            if rng.random() < 0.5:
                # 正对：同一函数 + 增强
                fid = rng.choice(func_ids)
                mm = all_feats[fid]
                f1, f2 = mm, _augment(mm)
                label = 1.0
            else:
                # 负对：不同函数
                f1 = all_feats[rng.choice(func_ids)]
                f2 = all_feats[rng.choice(func_ids)]
                label = 0.0

            ids1, pad1 = safe_tokenize(f1, vocab, max_len=max_len)
            ids2, pad2 = safe_tokenize(f2, vocab, max_len=max_len)
            t1 = torch.tensor([ids1], dtype=torch.long, device=device)
            p1 = torch.tensor([pad1], dtype=torch.bool, device=device)
            t2 = torch.tensor([ids2], dtype=torch.long, device=device)
            p2 = torch.tensor([pad2], dtype=torch.bool, device=device)
            y = torch.tensor([label], dtype=torch.float32, device=device)

            v1 = model(t1, p1)
            v2 = model(t2, p2)
            loss = loss_fn(v1, v2, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1

        avg = total_loss / max(count, 1)
        print(f"  SAFE 训练 epoch {epoch + 1}/{epochs}, avg_loss={avg:.4f}", flush=True)

    safe_save_model(model, vocab, output_model_path)
    print(f"  SAFE 模型已保存: {output_model_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 manifest（二进制+漏洞函数名+CVE）构建完整 CVE 库"
    )
    parser.add_argument("--manifest", required=True, help="JSON manifest 路径")
    parser.add_argument("-o", "--output-dir", required=True, help="输出目录")
    parser.add_argument(
        "--safe-epochs",
        type=int,
        default=2,
        help="SAFE 训练轮数（默认 2）",
    )
    parser.add_argument(
        "--safe-pairs",
        type=int,
        default=200,
        help="SAFE 每轮训练对数（默认 200）",
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
            print(f"错误: 条目 {i} 缺少 cve", file=sys.stderr)
            sys.exit(1)

        print(f"[{i + 1}/{len(rows)}] Ghidra 提取: {binary_rel}", flush=True)
        raw = _load_lsir_raw(binary_abs)
        funcs = raw.get("functions") or []
        if not funcs:
            print(f"错误: {binary_rel} lsir_raw 无 functions", file=sys.stderr)
            sys.exit(1)

        try:
            entry, sym = _resolve_entry(funcs, fn_name, entry_opt)
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
            print(f"错误: 重复的 function_id {fid}", file=sys.stderr)
            sys.exit(1)
        library_features[fid] = mm
        manifest_by_fid[fid] = {
            "symbol_name": sym,
            "cve": list(cve_list),
        }
        print(f"  ✓ {fid} → {cve_list}", flush=True)

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    feat_path = os.path.join(out_dir, "library_features.json")
    model_path = os.path.join(out_dir, "safe_model.pt")
    emb_path = os.path.join(out_dir, "library_safe_embeddings.json")
    cve_map_path = os.path.join(out_dir, "library_cve_map.json")

    # 1) 写特征
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(library_features, f, ensure_ascii=False)
    print(f"已写入: {feat_path}", flush=True)

    # 2) 训练 SAFE 模型
    print("训练 SAFE 模型...", flush=True)
    _train_safe_model(
        feat_path,
        model_path,
        epochs=args.safe_epochs,
        num_pairs=args.safe_pairs,
    )

    # 3) 用训练模型生成库嵌入
    print("生成库嵌入...", flush=True)
    from features.baselines.safe import embed_batch_safe

    emb_items: list[dict[str, Any]] = []
    for fid, mm in library_features.items():
        result = embed_batch_safe(
            {"functions": [{"name": fid, "features": {"multimodal": mm}}]},
            model_path=model_path,
        )
        if result:
            info = manifest_by_fid.get(fid, {})
            emb_items.append({
                "function_id": fid,
                "name": info.get("symbol_name", ""),
                "cve": info.get("cve", []),
                "vector": result[0]["vector"],
            })

    emb_out = {"functions": emb_items}
    with open(emb_path, "w", encoding="utf-8") as f:
        json.dump(emb_out, f, indent=2, ensure_ascii=False)
    print(f"已写入: {emb_path}", flush=True)

    # 4) CVE 映射
    cve_map = {
        fid: info["cve"]
        for fid, info in manifest_by_fid.items()
    }
    with open(cve_map_path, "w", encoding="utf-8") as f:
        json.dump(cve_map, f, indent=2, ensure_ascii=False)
    print(f"已写入: {cve_map_path}", flush=True)

    print("\nfunction_id → cve:", flush=True)
    for fid in sorted(manifest_by_fid.keys()):
        print(f"  {fid} → {manifest_by_fid[fid]['cve']}", flush=True)


if __name__ == "__main__":
    main()
