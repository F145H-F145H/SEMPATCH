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
import gc
import json
import logging
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

log = logging.getLogger("build_embeddings_db")

_BINARY_EXTS = (".elf", ".bin", ".so")
_DEFAULT_TEMP_DIR = os.path.join(PROJECT_ROOT, "output", "binkit_emb_temp")


# ---------------------------------------------------------------------------
# 流式 JSON 对象解析器：逐条 yield (key, value)，不将整个文件加载到内存
# ---------------------------------------------------------------------------

def _iter_json_object_records(fp):
    """
    流式解析 JSON 对象 {key: value, ...}，逐条 yield (key, value)。
    不将整个文件加载到内存，避免大 features 文件导致 OOM。
    支持 value 为任意合法 JSON 类型（对象、数组、字符串、数字等）。
    """
    raw = fp.read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")

    n = len(raw)
    i = 0

    while i < n and raw[i] in b' \t\r\n':
        i += 1
    if i >= n or raw[i] != 123:  # '{'
        return
    i += 1

    def skip_string(s, pos):
        pos += 1
        while pos < len(s):
            if s[pos] == 92:  # '\'
                pos += 2
                continue
            if s[pos] == 34:  # '"'
                return pos + 1
            pos += 1
        return len(s)

    def skip_value(s, pos):
        while pos < len(s) and s[pos] in b' \t\r\n':
            pos += 1
        if pos >= len(s):
            return pos
        c = s[pos]
        if c == 34:  # '"'
            return skip_string(s, pos)
        if c == 91:  # '['
            pos += 1
            depth = 1
            while pos < len(s) and depth > 0:
                if s[pos] == 34:
                    pos = skip_string(s, pos)
                    continue
                if s[pos] == 91:
                    depth += 1
                elif s[pos] == 93:
                    depth -= 1
                pos += 1
            return pos
        if c == 123:  # '{'
            pos += 1
            depth = 1
            while pos < len(s) and depth > 0:
                if s[pos] == 34:
                    pos = skip_string(s, pos)
                    continue
                if s[pos] == 123:
                    depth += 1
                elif s[pos] == 125:
                    depth -= 1
                pos += 1
            return pos
        if c in b'-0123456789tfn':  # number / true / false / null
            while pos < len(s) and s[pos] not in b',} \t\r\n':
                pos += 1
            return pos
        return pos

    while i < n:
        while i < n and raw[i] in b' \t\r\n':
            i += 1
        if i >= n:
            break
        if raw[i] == 125:  # '}'
            break
        if raw[i] != 34:  # '"'
            break
        key_start = i
        key_end = skip_string(raw, i)
        try:
            key = json.loads(raw[key_start:key_end].decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            break
        i = key_end

        while i < n and raw[i] in b' \t\r\n':
            i += 1
        if i >= n or raw[i] != 58:  # ':'
            break
        i += 1

        while i < n and raw[i] in b' \t\r\n':
            i += 1
        if i >= n:
            break
        val_start = i
        val_end = skip_value(raw, i)
        try:
            value = json.loads(raw[val_start:val_end].decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            break
        i = val_end

        yield key, value
        del value

        while i < n and raw[i] in b' \t\r\n':
            i += 1
        if i < n and raw[i] == 44:  # ','
            i += 1


# ---------------------------------------------------------------------------
# 特征提取辅助
# ---------------------------------------------------------------------------

def _extract_pcode_tokens_from_lsir_func(lsir_func: dict) -> list[str]:
    """从单个规范化的 lsir 函数节点提取 pcode token 列表。"""
    tokens: list[str] = []
    for bb in lsir_func.get("basic_blocks", []):
        for insn in bb.get("instructions", []):
            mnemonic = insn.get("mnemonic", "")
            if mnemonic:
                tokens.append(mnemonic)
            for op in insn.get("operands", []):
                if isinstance(op, str) and op:
                    tokens.append(op)
            for pcode_op in insn.get("pcode", []):
                if isinstance(pcode_op, dict):
                    op_str = pcode_op.get("op", "")
                    if op_str:
                        tokens.append(op_str)
                    for inp in pcode_op.get("inputs", []):
                        if isinstance(inp, str) and inp:
                            tokens.append(inp)
                elif isinstance(pcode_op, str) and pcode_op:
                    tokens.append(pcode_op)
    return tokens


def _extract_pcode_tokens_from_raw(raw_data: dict) -> list[dict]:
    """
    快速路径：直接从 lsir_raw 提取 pcode token 序列，跳过 graph/acfg/fuse。
    SAFE 模型仅需要 sequence.pcode_tokens，无需完整特征提取流水线。
    """
    from utils.pcode_normalizer import normalize_lsir_raw

    raw = normalize_lsir_raw(raw_data)
    functions = []
    for fn in raw.get("functions", []):
        tokens = _extract_pcode_tokens_from_lsir_func(fn)
        if tokens:
            functions.append(
                {
                    "name": fn.get("name", ""),
                    "features": {"multimodal": {"sequence": {"pcode_tokens": tokens}}},
                }
            )
    return functions


def _extract_training_features_from_raw(
    raw_data: dict,
    binary_rel: str,
) -> list[dict]:
    """
    一次 lsir_raw 遍历同时提取：
    - SAFE 所需 pcode_tokens
    - MultiModal 所需完整 multimodal features (graph + sequence + acfg + dfg)
    返回训练用记录列表，每条含 function_id / multimodal / safe_tokens。
    function_id 格式与 dataset._function_id 一致: {binary_rel}|0x{entry_lower}
    """
    from utils.pcode_normalizer import normalize_lsir_raw
    from utils.ir_builder import build_lsir
    from utils.feature_extractors import (
        extract_acfg_features,
        extract_graph_features,
        extract_sequence_features,
        fuse_features,
    )

    raw = normalize_lsir_raw(raw_data)
    funcs_raw = raw.get("functions", [])

    all_safe_tokens: list[list[str]] = []
    for fn in funcs_raw:
        all_safe_tokens.append(_extract_pcode_tokens_from_lsir_func(fn))

    lsir = build_lsir(raw, include_cfg=True, include_dfg=True)

    records: list[dict] = []
    for i, fn in enumerate(lsir.get("functions", [])):
        name = fn.get("name", "")
        entry_raw = fn.get("entry", "")
        entry_norm = entry_raw.strip().lower()
        if entry_norm and not entry_norm.startswith("0x"):
            entry_norm = f"0x{entry_norm}"
        fid = f"{binary_rel}|{entry_norm}" if entry_norm else f"{binary_rel}|{name}"

        gf = extract_graph_features(fn)
        sf = extract_sequence_features(fn)
        acfg = extract_acfg_features(fn)
        fused = fuse_features(gf, sf, acfg_feats=acfg)

        records.append(
            {
                "function_id": fid,
                "multimodal": fused.get("multimodal", {}),
                "safe_tokens": all_safe_tokens[i] if i < len(all_safe_tokens) else [],
            }
        )
    return records


def _process_single_lsir(
    raw_data: dict,
    model_path: str | None,
    filter_substr: str | None,
    model_type: str = "sempatch",
) -> list:
    """从 lsir_raw 提取特征并嵌入，返回 EmbeddingItem 列表。"""
    from features.baselines.jtrans_style import embed_batch_jtrans_style
    from features.baselines.safe import embed_batch_safe
    from features.inference import embed_batch

    funcs_raw = raw_data.get("functions", [])
    if filter_substr:
        funcs_raw = [f for f in funcs_raw if filter_substr.lower() in (f.get("name") or "").lower()]
        print(f"过滤后保留 {len(funcs_raw)} 个函数")

    if model_type == "safe":
        features = {"functions": _extract_pcode_tokens_from_raw(raw_data)}
        if filter_substr:
            features["functions"] = [
                f
                for f in features["functions"]
                if filter_substr.lower() in f.get("name", "").lower()
            ]
        return embed_batch_safe(features, model_path=model_path)

    from utils.ir_builder import build_lsir
    from utils.pcode_normalizer import normalize_lsir_raw
    from utils.feature_extractors import (
        extract_acfg_features,
        extract_graph_features,
        extract_sequence_features,
        fuse_features,
    )

    raw = {"functions": funcs_raw}
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


# ---------------------------------------------------------------------------
# 核心：流式处理 features 文件 → 嵌入
# ---------------------------------------------------------------------------

def _process_features_file(
    features_path: str,
    model_path: str | None = None,
    safe_batch_size: int = 1024,
    embed_kind: str = "safe",
    output_path: str | None = None,
) -> list:
    """
    从 library_features.json 格式读取特征并计算基线嵌入，返回 [{function_id, vector}]。
    embed_kind: safe | jtrans_style；model_path 为对应基线检查点，未指定则用随机初始化。

    内存优化：流式读取 JSON 对象，不将整个文件加载到内存。分块嵌入并显式释放中间数据。
    进度报告：每 10 秒打印一次进度，每个批次完成时打印摘要。
    """
    from features.baselines.jtrans_style import embed_batch_jtrans_style
    from features.baselines.safe import embed_batch_safe
    from utils.precomputed_multimodal_io import is_jsonl_sidecar_path, iter_jsonl_sidecar

    batch = max(1, int(safe_batch_size))
    out: list[dict] = []
    chunk: list[dict] = []
    vocab: dict[str, int] = {"[PAD]": 0, "[UNK]": 1}

    n_records = 0
    n_flushes = 0
    t_start = time.monotonic()
    last_log_t = t_start
    LOG_INTERVAL = 10.0

    def _log_progress(force=False):
        nonlocal last_log_t
        now = time.monotonic()
        if not force and (now - last_log_t) < LOG_INTERVAL:
            return
        elapsed = now - t_start
        speed = n_records / elapsed if elapsed > 0 else 0
        mem_mb = 0
        try:
            import resource
            mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            pass
        log.info(
            "嵌入进度: %d 条记录, %d 批已完成, 词表 %d tokens, "
            "速度 %.0f 条/s, 耗时 %.0fs, 内存 %.0fMB",
            n_records, n_flushes, len(vocab), speed, elapsed, mem_mb,
        )
        last_log_t = now

    def _update_vocab(mm):
        seq = mm.get("sequence") or {}
        for t in seq.get("pcode_tokens") or []:
            if t and t not in vocab:
                vocab[t] = len(vocab)
        graph = mm.get("graph") or {}
        for nf in graph.get("node_features") or []:
            opcodes = nf if isinstance(nf, list) else (nf.get("pcode_opcodes") or [])
            for op in opcodes:
                if op and op not in vocab:
                    vocab[op] = len(vocab)

    def _flush_chunk():
        nonlocal chunk, n_flushes
        if not chunk:
            return
        n_flushes += 1
        chunk_size = len(chunk)
        t_flush = time.monotonic()
        if embed_kind == "jtrans_style":
            result = embed_batch_jtrans_style({"functions": chunk}, model_path=model_path)
        else:
            result = embed_batch_safe({"functions": chunk}, model_path=model_path)
        flush_elapsed = time.monotonic() - t_flush
        out.extend({"function_id": item["name"], "vector": item["vector"]} for item in result)
        log.info(
            "  批 %d: %d 条嵌入完成 (%.1fs), 累计 %d",
            n_flushes, chunk_size, flush_elapsed, len(out),
        )
        chunk = []
        gc.collect()

    is_jsonl = is_jsonl_sidecar_path(features_path)

    if is_jsonl:
        records_iter = iter_jsonl_sidecar(features_path)
        log.info("开始处理 JSONL 特征文件: %s (safe=%s)", features_path, embed_kind)
    else:
        file_size_mb = os.path.getsize(features_path) / (1024 * 1024)
        log.info(
            "开始流式处理 JSON 特征文件: %s (%.1f MB, safe=%s)",
            features_path, file_size_mb, embed_kind,
        )
        fp = open(features_path, "rb")
        records_iter = _iter_json_object_records(fp)

    try:
        for fid, mm in records_iter:
            if not isinstance(fid, str) or not isinstance(mm, dict):
                continue
            n_records += 1
            _update_vocab(mm)
            chunk.append({"name": fid, "features": {"multimodal": mm}})
            if len(chunk) >= batch:
                _flush_chunk()
                _log_progress()
        _flush_chunk()
        _log_progress(force=True)
    finally:
        if not is_jsonl:
            try:
                fp.close()
            except Exception:
                pass
        gc.collect()

    elapsed = time.monotonic() - t_start
    speed = n_records / elapsed if elapsed > 0 else 0
    log.info(
        "嵌入完成: %d 条记录 → %d 个嵌入向量, 词表 %d tokens, "
        "总耗时 %.1fs (%.0f 条/s)",
        n_records, len(out), len(vocab), elapsed, speed,
    )
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
    # 配置日志：确保用户在终端看到进度输出，无需额外设置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑：如果输出文件已存在，加载已有结果并跳过已完成的二进制",
    )
    parser.add_argument(
        "--intermediate-save",
        action="store_true",
        default=True,
        help="启用中间结果保存（默认开启）",
    )
    parser.add_argument(
        "--no-intermediate-save",
        action="store_true",
        help="禁用中间结果保存",
    )
    parser.add_argument(
        "--intermediate-save-every",
        type=int,
        default=50,
        help="每处理 N 个二进制保存一次中间结果（默认 50）",
    )
    parser.add_argument(
        "--emit-training-features",
        action="store_true",
        help="同时输出训练用 JSONL（含完整 multimodal + safe_tokens），供两个训练脚本消费",
    )
    parser.add_argument(
        "--training-features-output",
        default=None,
        help="训练特征 JSONL 输出路径（默认：-o 参数去掉后缀 + .training.jsonl）",
    )
    args = parser.parse_args()

    if args.no_intermediate_save:
        args.intermediate_save = False

    modes = sum(
        [
            bool(args.input),
            bool(args.input_dir),
            bool(args.index_file),
            bool(args.features_file),
        ]
    )
    if modes == 0:
        print(
            "错误: 必须指定 input、--input-dir、--index-file 或 --features-file 之一",
            file=sys.stderr,
        )
        sys.exit(1)
    if modes > 1:
        print(
            "错误: input、--input-dir、--index-file、--features-file 互斥，只能指定一个",
            file=sys.stderr,
        )
        sys.exit(1)

    all_embeddings: list = []

    if args.features_file:
        fe_path = os.path.abspath(args.features_file)
        if not os.path.isfile(fe_path):
            log.error("特征文件不存在 %s", fe_path)
            sys.exit(1)
        baseline_path = args.model_path if args.model in ("safe", "jtrans_style") else None
        fe_kind = args.model if args.model in ("safe", "jtrans_style") else "safe"
        all_embeddings = _process_features_file(
            fe_path,
            model_path=baseline_path,
            safe_batch_size=args.safe_batch_size,
            embed_kind=fe_kind,
            output_path=os.path.abspath(args.output),
        )

        # --emit-training-features：从 features 文件流式生成 .training.jsonl
        if args.emit_training_features:
            from utils.precomputed_multimodal_io import (
                is_jsonl_sidecar_path,
                iter_jsonl_sidecar,
            )

            if args.training_features_output:
                training_out_path = os.path.abspath(args.training_features_output)
            else:
                base, _ = os.path.splitext(os.path.abspath(args.output))
                training_out_path = base + ".training.jsonl"
            os.makedirs(os.path.dirname(training_out_path) or ".", exist_ok=True)

            if is_jsonl_sidecar_path(fe_path):
                src_iter = iter_jsonl_sidecar(fe_path)
            else:
                _fp = open(fe_path, "rb")
                src_iter = _iter_json_object_records(_fp)

            training_count = 0
            t_emit = time.monotonic()
            LOG_EVERY = 50000
            log.info("开始生成训练特征 JSONL → %s", training_out_path)
            try:
                with open(training_out_path, "w", encoding="utf-8") as tfp:
                    for fid, mm in src_iter:
                        if not isinstance(fid, str) or not isinstance(mm, dict):
                            continue
                        seq = mm.get("sequence") or {}
                        safe_tokens = seq.get("pcode_tokens") or []
                        rec = {
                            "function_id": fid,
                            "multimodal": mm,
                            "safe_tokens": safe_tokens,
                        }
                        tfp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        training_count += 1
                        if training_count % LOG_EVERY == 0:
                            elapsed = time.monotonic() - t_emit
                            speed = training_count / elapsed if elapsed > 0 else 0
                            log.info("  训练特征: %d 条 (%.0f 条/s)", training_count, speed)
            finally:
                if not is_jsonl_sidecar_path(fe_path):
                    try:
                        _fp.close()
                    except Exception:
                        pass
            elapsed = time.monotonic() - t_emit
            log.info("训练特征已写入 %s (%d 条, %.1fs)", training_out_path, training_count, elapsed)

    elif args.input:
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

        import time as _time

        out_path = os.path.abspath(args.output)
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        training_fp = None
        training_out_path = None
        training_count = 0
        if args.emit_training_features:
            if args.training_features_output:
                training_out_path = os.path.abspath(args.training_features_output)
            else:
                base, _ = os.path.splitext(out_path)
                training_out_path = base + ".training.jsonl"
            os.makedirs(os.path.dirname(training_out_path) or ".", exist_ok=True)
            training_fp = open(training_out_path, "a", encoding="utf-8")
            print(f"训练特征将写入: {training_out_path} (JSONL, append)")

        processed_rels: set[str] = set()
        if args.resume and os.path.isfile(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                if prev.get("functions"):
                    print(f"  [resume] 已有 {len(prev['functions'])} 个函数嵌入，继续追加")
                    all_embeddings.extend(prev["functions"])
                    for emb_item in prev["functions"]:
                        fid = emb_item.get("function_id", emb_item.get("name", ""))
                        if "|" in fid:
                            processed_rels.add(fid.rsplit("|", 1)[0])
            except (OSError, json.JSONDecodeError):
                pass

        total_funcs = 0
        t_start = _time.monotonic()

        try:
            for idx, bin_path in enumerate(binaries, start=1):
                rel_path = os.path.relpath(bin_path, PROJECT_ROOT)
                if rel_path in processed_rels:
                    print(f"  [{idx}/{len(binaries)}] 跳过已完成 {rel_path}")
                    continue
                output_dir = os.path.join(temp_dir, f"v{idx}")
                os.makedirs(output_dir, exist_ok=True)
                t_bin = _time.monotonic()
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
                    print(f"  [{idx}/{len(binaries)}] ⚠ 处理失败 {rel_path}: {e}", file=sys.stderr)
                    continue
                if not lsir_raw or not lsir_raw.get("functions"):
                    print(f"  [{idx}/{len(binaries)}] ⚠ 无函数 {rel_path}", file=sys.stderr)
                    continue

                t_feats = None
                if training_fp is not None:
                    try:
                        t_feats = _extract_training_features_from_raw(lsir_raw, rel_path)
                        training_buf = (
                            "\n".join(json.dumps(rec, ensure_ascii=False) for rec in t_feats) + "\n"
                        )
                        training_fp.write(training_buf)
                        training_fp.flush()
                        training_count += len(t_feats)
                    except Exception as e:
                        print(
                            f"  [{idx}/{len(binaries)}] ⚠ 训练特征提取失败 {rel_path}: {e}",
                            file=sys.stderr,
                        )
                        t_feats = None

                if t_feats is not None:
                    from features.baselines.jtrans_style import embed_batch_jtrans_style
                    from features.baselines.safe import embed_batch_safe
                    from features.inference import embed_batch

                    flt = args.filter.lower() if args.filter else None

                    if args.model == "safe":
                        safe_funcs = []
                        for rec in t_feats:
                            if flt and flt not in rec["function_id"].lower():
                                continue
                            tokens = rec.get("safe_tokens") or []
                            if tokens:
                                safe_funcs.append(
                                    {
                                        "name": rec["function_id"],
                                        "features": {
                                            "multimodal": {"sequence": {"pcode_tokens": tokens}}
                                        },
                                    }
                                )
                        emb = embed_batch_safe(
                            {"functions": safe_funcs}, model_path=args.model_path
                        )
                    else:
                        embed_input = []
                        for rec in t_feats:
                            if flt and flt not in rec["function_id"].lower():
                                continue
                            embed_input.append(
                                {
                                    "name": rec["function_id"],
                                    "features": {"multimodal": rec["multimodal"]},
                                }
                            )
                        if args.model == "jtrans_style":
                            emb = embed_batch_jtrans_style(
                                {"functions": embed_input}, model_path=args.model_path
                            )
                        else:
                            emb = embed_batch(
                                {"functions": embed_input}, model_path=args.model_path
                            )
                else:
                    emb = _process_single_lsir(
                        lsir_raw, args.model_path, args.filter, model_type=args.model
                    )

                del lsir_raw

                if idx % 10 == 0:
                    try:
                        from utils._ghidra_helpers import clear_peek_cache

                        clear_peek_cache()
                    except ImportError:
                        pass
                all_embeddings.extend(emb)
                total_funcs += len(emb)
                elapsed = _time.monotonic() - t_bin
                elapsed_total = _time.monotonic() - t_start
                speed = total_funcs / elapsed_total if elapsed_total > 0 else 0
                eta_sec = (len(binaries) - idx) / (idx / elapsed_total) if elapsed_total > 0 else 0
                extra = f" | 训练特征 {training_count}" if training_fp else ""
                print(
                    f"  [{idx}/{len(binaries)}] {rel_path}: {len(emb)} 函数 "
                    f"({elapsed:.1f}s) | 累计 {total_funcs} 函数{extra} | "
                    f"速度 {speed:.0f} fn/s | ETA {eta_sec/60:.1f}min"
                )

                if args.intermediate_save and idx % args.intermediate_save_every == 0:
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump({"functions": all_embeddings}, f, indent=2, ensure_ascii=False)
                    if training_fp:
                        training_fp.flush()
                    print(
                        f"  [checkpoint] 已保存 {len(all_embeddings)} 嵌入 + {training_count} 训练特征"
                    )
        finally:
            if training_fp is not None:
                training_fp.close()
                print(f"训练特征已写入 {training_out_path} ({training_count} 条)")

        elapsed_total = _time.monotonic() - t_start
        print(
            f"\n批量处理完成: {total_funcs} 个函数, {elapsed_total:.1f}s, 平均 {total_funcs/elapsed_total:.0f} fn/s"
        )

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"functions": all_embeddings}, f, indent=2, ensure_ascii=False)

    log.info("已写入 %s (%d 个函数嵌入)", out_path, len(all_embeddings))


if __name__ == "__main__":
    main()
