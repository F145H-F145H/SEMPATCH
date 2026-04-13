#!/usr/bin/env python3
"""
将已有的 .training.jsonl + 索引转换为预计算 npz 格式，供训练时零 dict 操作加载。

这是数据准备的最后一步：已有 JSONL 侧车 → 预计算 npz。
完整的从零开始流程仍使用原有 6 步脚本（或 future: prepare_training_data.py 全合一版）。

用法:
  PYTHONPATH=src python scripts/sidechain/jsonl_to_npz.py \
    --jsonl data/binkit_functions_common.training.jsonl \
    --index data/binkit_functions_common.json \
    -o data/training/features.npz

  # 自定义维度（需与训练脚本参数一致）
  PYTHONPATH=src python scripts/sidechain/jsonl_to_npz.py \
    --jsonl data/features.jsonl \
    --index data/index.json \
    -o data/features.npz \
    --max-seq-len 512 --max-graph-nodes 128 --max-dfg-nodes 64

输出:
  - features.npz: 预计算 tensor 数组
  - features.fid_map.json: function_id → array_index 映射
  - features.vocab.json: pcode vocab（从 JSONL 扫描构建）
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("jsonl_to_npz")


def _collect_needed_ids(index_path: str) -> Set[str]:
    """从索引文件收集所有需要的 function_id。"""
    import json as _json

    needed: Set[str] = set()
    with open(index_path, encoding="utf-8") as f:
        raw = _json.load(f)
    if not isinstance(raw, list):
        raw = [raw] if isinstance(raw, dict) else []

    for item in raw:
        if not isinstance(item, dict):
            continue
        binary_rel = item.get("binary", "")
        funcs = item.get("functions") or []
        for fn in funcs:
            entry = fn.get("entry", "")
            if not entry:
                continue
            s = entry.strip().lower()
            if not s.startswith("0x"):
                s = f"0x{s}" if s else "0x0"
            fid = f"{binary_rel}|{s}"
            needed.add(fid)
    return needed


def _collect_vocab_from_mm(mm: Dict[str, Any], vocab: Dict[str, int]) -> None:
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


def main():
    parser = argparse.ArgumentParser(description="JSONL → 预计算 npz 转换器")
    parser.add_argument("--jsonl", required=True, help=".training.jsonl 侧车文件")
    parser.add_argument("--index", required=True, help="binkit_functions.json 索引文件")
    parser.add_argument("-o", "--output", required=True, help="输出 npz 路径")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-graph-nodes", type=int, default=128)
    parser.add_argument("--max-dfg-nodes", type=int, default=64)
    parser.add_argument("--pcode-vocab-size", type=int, default=256)
    parser.add_argument("--max-edges", type=int, default=512)
    parser.add_argument("--max-dfg-edges", type=int, default=256)
    args = parser.parse_args()

    from utils.precomputed_multimodal_io import iter_jsonl_sidecar
    from utils.npz_features import build_precomputed_npz
    from features.baselines.safe import collect_vocab_from_features_jsonl

    # 1. 收集需要的 function_ids
    log.info("步骤 1/4: 收集索引中的 function_ids…")
    needed = _collect_needed_ids(args.index)
    log.info("  需要 %d 个函数", len(needed))

    # 2. 从 JSONL 加载需要的 multimodal + 构建 vocab
    log.info("步骤 2/4: 从 JSONL 加载 multimodal 特征…")
    multimodals: List[Tuple[str, Dict[str, Any]]] = []
    vocab: Dict[str, int] = {"[PAD]": 0, "[UNK]": 1}
    t0 = time.perf_counter()
    loaded = 0
    for fid, mm in iter_jsonl_sidecar(args.jsonl):
        if fid not in needed:
            continue
        multimodals.append((fid, mm))
        _collect_vocab_from_mm(mm, vocab)
        loaded += 1
        if loaded % 10000 == 0:
            elapsed = time.perf_counter() - t0
            log.info("  已加载 %d/%d (%.0f 条/s)", loaded, len(needed), loaded / elapsed if elapsed > 0 else 0)

    elapsed = time.perf_counter() - t0
    log.info("  加载完成: %d 个函数, %d vocab tokens, %.1fs", loaded, len(vocab), elapsed)

    missing = needed - {fid for fid, _ in multimodals}
    if missing:
        log.warning("  %d 个函数在 JSONL 中未找到（将跳过）", len(missing))

    # 3. 转换为 npz
    log.info("步骤 3/4: 转换为预计算 npz 数组…")
    fid_to_idx = build_precomputed_npz(
        multimodals, vocab, args.output,
        max_seq_len=args.max_seq_len,
        max_graph_nodes=args.max_graph_nodes,
        max_dfg_nodes=args.max_dfg_nodes,
        pcode_vocab_size=args.pcode_vocab_size,
        max_edges=args.max_edges,
        max_dfg_edges=args.max_dfg_edges,
    )

    # 4. 写入映射文件
    log.info("步骤 4/4: 写入辅助文件…")
    base = os.path.splitext(args.output)[0]

    fid_map_path = base + ".fid_map.json"
    with open(fid_map_path, "w", encoding="utf-8") as f:
        json.dump(fid_to_idx, f, ensure_ascii=False)
    log.info("  function_id → index 映射: %s (%d 条)", fid_map_path, len(fid_to_idx))

    vocab_path = base + ".vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    log.info("  vocab: %s (%d tokens)", vocab_path, len(vocab))

    log.info("完成！训练命令示例:")
    log.info("")
    log.info("  # SAFE 训练")
    log.info("  PYTHONPATH=src python scripts/sidechain/train_safe.py \\")
    log.info("    --npz %s \\", args.output)
    log.info("    --fid-map %s \\", fid_map_path)
    log.info("    --index %s \\", args.index)
    log.info("    --vocab %s \\", vocab_path)
    log.info("    --epochs 10 --batch-size 4 --num-pairs 10000 --lr 1e-3 \\")
    log.info("    --save-path output/safe_best_model.pt --no-tb")
    log.info("")
    log.info("  # MultiModal 训练")
    log.info("  PYTHONPATH=src python scripts/sidechain/train_multimodal.py \\")
    log.info("    --npz %s \\", args.output)
    log.info("    --fid-map %s \\", fid_map_path)
    log.info("    --index %s \\", args.index)
    log.info("    --vocab %s \\", vocab_path)
    log.info("    --epochs 20 --batch-size 4 --num-pairs 20000 --lr 1e-4 \\")
    log.info("    --save-path output/best_model.pth --no-tb")


if __name__ == "__main__":
    main()
