#!/usr/bin/env python3
"""
生成合成 multimodal 特征对，用于快速验证训练流程（无需 Ghidra）。
"""
import argparse
import json
import os
import random
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def main():
    parser = argparse.ArgumentParser(description="生成合成 multimodal 特征对")
    parser.add_argument("-o", "--output", required=True, help="输出 JSON 路径")
    parser.add_argument("-n", "--num-pairs", type=int, default=100, help="生成对数")
    parser.add_argument("--positive-ratio", type=float, default=0.5, help="正对比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--with-dfg",
        action="store_true",
        help="为每样本 multimodal 增加 dfg 子图（阶段 H 训练/消融）",
    )
    args = parser.parse_args()

    try:
        from features.models.multimodal_fusion import get_default_vocab
        vocab = get_default_vocab()
    except ImportError:
        vocab = {"[PAD]": 0, "[UNK]": 1, "COPY": 2, "INT_ADD": 3, "INT_SUB": 4}
    vocab_keys = [k for k in vocab.keys() if k not in ("[PAD]", "[UNK]")]
    if not vocab_keys:
        vocab_keys = ["COPY", "INT_ADD", "INT_SUB"]

    rng = random.Random(args.seed)

    def make_one(seq_len=32, num_nodes=8):
        tokens = [rng.choice(vocab_keys) for _ in range(seq_len)]
        jump_mask = [1 if rng.random() < 0.1 else 0 for _ in range(seq_len)]
        node_features = [
            {"pcode_opcodes": [rng.choice(vocab_keys) for _ in range(rng.randint(1, 5))]}
            for _ in range(num_nodes)
        ]
        edges_src = list(range(num_nodes - 1))
        edges_dst = list(range(1, num_nodes))
        mm: dict = {
            "graph": {
                "num_nodes": num_nodes,
                "edge_index": [edges_src, edges_dst],
                "node_list": [f"bb_{i}" for i in range(num_nodes)],
                "node_features": node_features,
            },
            "sequence": {"pcode_tokens": tokens, "jump_mask": jump_mask, "seq_len": seq_len},
        }
        if args.with_dfg:
            nd = max(2, num_nodes // 2)
            dfg_nodes = [f"0x{1000 + i * 8:x}:(register, 0x{i:x}, 8)" for i in range(nd)]
            dfg_edges_src = list(range(nd - 1))
            dfg_edges_dst = list(range(1, nd))
            mm["dfg"] = {
                "num_nodes": nd,
                "edge_index": [dfg_edges_src, dfg_edges_dst],
                "node_list": dfg_nodes,
                "node_features": [2 + (i * 17 % 500) for i in range(nd)],
            }
        return mm

    pairs = []
    for _ in range(args.num_pairs):
        f1 = make_one()
        if rng.random() < args.positive_ratio:
            import copy
            f2 = copy.deepcopy(f1)
            label = 1
        else:
            f2 = make_one()
            label = 0
        pairs.append({"feature1": f1, "feature2": f2, "label": label})

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"pairs": pairs}, f, indent=2, ensure_ascii=False)
    print(f"已生成 {len(pairs)} 对至 {out_path}")


if __name__ == "__main__":
    main()
