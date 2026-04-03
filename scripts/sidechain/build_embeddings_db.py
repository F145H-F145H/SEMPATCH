#!/usr/bin/env python3
"""
从 LSIR 或 lsir_raw 构建 embeddings 格式漏洞库，供 fusion/semantic_embed 策略使用。

用法:
  python scripts/build_embeddings_db.py data/vulnerability_db/test_vuln_lsir.json -o data/vulnerability_db/test_embeddings.json
  python scripts/build_embeddings_db.py output/ghidra_out/lsir_raw.json -o vuln_emb.json  # 从 lsir_raw
  python scripts/build_embeddings_db.py --index-file data/binkit_functions.json -o data/binkit_embeddings.json  # 批量
  python scripts/build_embeddings_db.py --input-dir data/binkit_subset -o data/binkit_embeddings.json  # 目录批量
  python scripts/build_embeddings_db.py --features-file data/two_stage/library_features.json -o data/two_stage/library_safe_embeddings.json  # SAFE
  python scripts/build_embeddings_db.py --features-file data/two_stage/library_features.jsonl -o out.json --model jtrans_style  # jTrans 风格
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

_BINARY_EXTS = (".elf", ".bin", ".so")
_DEFAULT_TEMP_DIR = os.path.join(PROJECT_ROOT, "output", "binkit_emb_temp")


def _process_single_lsir(
    raw_data: dict,
    model_path: str | None,
    filter_substr: str | None,
    model_type: str = "sempatch",
) -> list:
    """从 lsir_raw 提取特征并嵌入，返回 EmbeddingItem 列表。"""
    from utils.ir_builder import build_lsir
    from utils.pcode_normalizer import normalize_lsir_raw
    from utils.feature_extractors import (
        extract_acfg_features,
        extract_graph_features,
        extract_sequence_features,
        fuse_features,
    )
    from features.inference import embed_batch
    from features.baselines.jtrans_style import embed_batch_jtrans_style
    from features.baselines.safe import embed_batch_safe

    funcs = raw_data.get("functions", [])
    if filter_substr:
        funcs = [f for f in funcs if filter_substr.lower() in (f.get("name") or "").lower()]
        print(f"过滤后保留 {len(funcs)} 个函数")

    raw = {"functions": funcs}
    raw = normalize_lsir_raw(raw)
    lsir = build_lsir(raw, include_cfg=True, include_dfg=True)

    feats_list = []
    for fn in lsir.get("functions", []):
        gf = extract_graph_features(fn)
        sf = extract_sequence_features(fn)
        acfg = extract_acfg_features(fn)
        fused = fuse_features(gf, sf, acfg_feats=acfg)
        feats_list.append({"name": fn.get("name", ""), "features": fused})

    features = {"functions": feats_list}
    if model_type == "safe":
        return embed_batch_safe(features, model_path=model_path)
    if model_type == "jtrans_style":
        return embed_batch_jtrans_style(features, model_path=model_path)
    return embed_batch(features, model_path=model_path)


def _collect_binaries_from_index(index_path: str) -> list[str]:
    """从 binkit_functions.json 读取并返回去重后的二进制绝对路径列表。"""
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    seen: set[str] = set()
    binaries: list[str] = []
    for item in index:
        if not isinstance(item, dict):
            continue
        rel = item.get("binary", "")
        if not rel:
            continue
        abs_path = os.path.join(PROJECT_ROOT, rel) if not os.path.isabs(rel) else rel
        if abs_path not in seen and os.path.isfile(abs_path):
            seen.add(abs_path)
            binaries.append(abs_path)
    return binaries


def _process_features_file(
    features_path: str,
    model_path: str | None = None,
    safe_batch_size: int = 1024,
    embed_kind: str = "safe",
) -> list:
    """
    从 library_features.json 格式读取特征并计算基线嵌入，返回 [{function_id, vector}]。
    embed_kind: safe | jtrans_style；model_path 为对应基线检查点，未指定则用随机初始化。
    """
    from features.baselines.jtrans_style import embed_batch_jtrans_style
    from features.baselines.safe import embed_batch_safe
    from utils.precomputed_multimodal_io import is_jsonl_sidecar_path, iter_jsonl_sidecar

    batch = max(1, int(safe_batch_size))
    out: list[dict] = []
    chunk: list[dict] = []

    def _flush_chunk() -> None:
        nonlocal chunk
        if not chunk:
            return
        if embed_kind == "jtrans_style":
            result = embed_batch_jtrans_style({"functions": chunk}, model_path=model_path)
        else:
            result = embed_batch_safe({"functions": chunk}, model_path=model_path)
        out.extend(
            {"function_id": item["name"], "vector": item["vector"]}
            for item in result
        )
        chunk = []

    if is_jsonl_sidecar_path(features_path):
        records_iter = iter_jsonl_sidecar(features_path)
    else:
        with open(features_path, encoding="utf-8") as f:
            features_dict = json.load(f)
        if not isinstance(features_dict, dict):
            raise ValueError("特征文件格式应为 {function_id: multimodal_dict}")
        records_iter = features_dict.items()

    for fid, mm in records_iter:
        if not isinstance(fid, str) or not isinstance(mm, dict):
            continue
        chunk.append({"name": fid, "features": {"multimodal": mm}})
        if len(chunk) >= batch:
            _flush_chunk()
    _flush_chunk()
    return out


def _collect_binaries_from_dir(input_dir: str) -> list[str]:
    """从目录收集 .elf/.bin/.so 二进制绝对路径。"""
    binaries = []
    for f in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, f)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext in _BINARY_EXTS:
            binaries.append(os.path.abspath(path))
    return binaries


def main() -> None:
    parser = argparse.ArgumentParser(description="从 LSIR/lsir_raw 构建 embeddings 漏洞库")
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="lsir_raw.json 或 LSIR JSON 路径（与 --input-dir/--index-file 互斥）",
    )
    parser.add_argument("-o", "--output", required=True, help="输出 embeddings JSON")
    parser.add_argument("--filter", default=None, help="函数名过滤（子串匹配）")
    parser.add_argument(
        "--model-path",
        default=None,
        help="训练模型路径（可选，也可用 SEMPATCH_MODEL_PATH 环境变量）",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="二进制目录，遍历所有 .elf/.bin/.so 批量构建",
    )
    parser.add_argument(
        "--index-file",
        default=None,
        help="binkit_functions.json 索引路径，按索引中的 binary 批量构建",
    )
    parser.add_argument(
        "--temp-dir",
        default=_DEFAULT_TEMP_DIR,
        help="Ghidra 批量处理临时目录",
    )
    parser.add_argument(
        "--model",
        choices=["sempatch", "safe", "jtrans_style"],
        default="sempatch",
        help="嵌入模型：sempatch=MultiModalFusion；safe=SAFE 序列基线；jtrans_style=块序 jTrans 风格基线",
    )
    parser.add_argument(
        "--features-file",
        default=None,
        help="预计算特征文件路径（格式 {function_id: multimodal_dict}），直接构建 SAFE 嵌入，与 input/--input-dir/--index-file 互斥",
    )
    parser.add_argument(
        "--safe-batch-size",
        type=int,
        default=1024,
        help="--features-file 模式下 SAFE 分块嵌入批大小",
    )
    args = parser.parse_args()

    # 互斥：input | --input-dir | --index-file | --features-file 仅可指定其一
    modes = sum([
        bool(args.input),
        bool(args.input_dir),
        bool(args.index_file),
        bool(args.features_file),
    ])
    if modes == 0:
        print("错误: 必须指定 input、--input-dir、--index-file 或 --features-file 之一", file=sys.stderr)
        sys.exit(1)
    if modes > 1:
        print("错误: input、--input-dir、--index-file、--features-file 互斥，只能指定一个", file=sys.stderr)
        sys.exit(1)

    all_embeddings: list = []

    if args.features_file:
        fe_path = os.path.abspath(args.features_file)
        if not os.path.isfile(fe_path):
            print(f"错误: 特征文件不存在 {fe_path}", file=sys.stderr)
            sys.exit(1)
        baseline_path = (
            args.model_path if args.model in ("safe", "jtrans_style") else None
        )
        fe_kind = args.model if args.model in ("safe", "jtrans_style") else "safe"
        all_embeddings = _process_features_file(
            fe_path,
            model_path=baseline_path,
            safe_batch_size=args.safe_batch_size,
            embed_kind=fe_kind,
        )
        print(
            f"从特征文件 {args.features_file} 构建 {len(all_embeddings)} 个嵌入 "
            f"（{fe_kind}）"
        )
    elif args.input:
        # 单文件模式
        path = os.path.abspath(args.input)
        if not os.path.isfile(path):
            print(f"错误: 文件不存在 {path}", file=sys.stderr)
            sys.exit(1)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        all_embeddings = _process_single_lsir(
            data, args.model_path, args.filter, model_type=args.model
        )
    else:
        # 批量模式：--input-dir 或 --index-file
        if args.index_file:
            idx_path = os.path.abspath(args.index_file)
            if not os.path.isfile(idx_path):
                print(f"错误: 索引文件不存在 {idx_path}", file=sys.stderr)
                sys.exit(1)
            binaries = _collect_binaries_from_index(idx_path)
            print(f"从索引 {args.index_file} 获取 {len(binaries)} 个二进制")
        else:
            input_dir = os.path.abspath(args.input_dir)
            if not os.path.isdir(input_dir):
                print(f"错误: 目录不存在 {input_dir}", file=sys.stderr)
                sys.exit(1)
            binaries = _collect_binaries_from_dir(input_dir)
            print(f"从目录 {args.input_dir} 获取 {len(binaries)} 个二进制")

        if not binaries:
            print("错误: 未找到任何二进制文件", file=sys.stderr)
            sys.exit(1)

        from utils.ghidra_runner import run_ghidra_analysis

        temp_dir = os.path.abspath(args.temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        for idx, bin_path in enumerate(binaries, start=1):
            rel_path = os.path.relpath(bin_path, PROJECT_ROOT)
            output_dir = os.path.join(temp_dir, f"v{idx}")
            os.makedirs(output_dir, exist_ok=True)
            try:
                lsir_raw = run_ghidra_analysis(
                    binary_path=bin_path,
                    output_dir=output_dir,
                    project_name=f"BinkitEmb_{idx}",
                    script_name="extract_lsir_raw.java",
                    script_output_name="lsir_raw.json",
                    return_dict=True,
                )
            except Exception as e:
                print(f"警告: 处理失败 {rel_path}: {e}", file=sys.stderr)
                continue
            if not lsir_raw or not lsir_raw.get("functions"):
                print(f"警告: 无函数 {rel_path}", file=sys.stderr)
                continue
            emb = _process_single_lsir(
                lsir_raw, args.model_path, args.filter, model_type=args.model
            )
            all_embeddings.extend(emb)
            print(f"  [{idx}/{len(binaries)}] {rel_path}: {len(emb)} 函数嵌入")

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"functions": all_embeddings}, f, indent=2, ensure_ascii=False)

    print(f"已写入 {out_path} ({len(all_embeddings)} 个函数嵌入)")


if __name__ == "__main__":
    main()
