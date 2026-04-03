#!/usr/bin/env python3
"""
验证嵌入一致性：同函数嵌入相似度应接近 1.0，不同函数应明显更低。
支持 --model-path 与 SEMPATCH_MODEL_PATH 环境变量。
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def _cosine_sim(a, b):
    """余弦相似度。"""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)


def _make_feature_from_pair(pair_item: dict) -> dict:
    """从合成对中提取单函数特征结构（features 格式）。"""
    f = pair_item.get("feature1", pair_item)
    return {"name": "func", "features": {"multimodal": f}}


def main():
    parser = argparse.ArgumentParser(description="验证嵌入一致性")
    parser.add_argument("--model-path", default=None, help="训练模型路径（也可用 SEMPATCH_MODEL_PATH）")
    parser.add_argument("--synthetic-file", default=None, help="合成特征 JSON（默认自动生成）")
    parser.add_argument("-n", "--num-funcs", type=int, default=3, help="测试函数数量")
    args = parser.parse_args()

    from features.inference import embed_batch

    # 使用合成特征或生成
    if args.synthetic_file and os.path.isfile(args.synthetic_file):
        with open(args.synthetic_file, encoding="utf-8") as f:
            data = json.load(f)
        pairs = data.get("pairs", [])[: max(3, args.num_funcs)]
        if not pairs:
            print("错误: 合成文件中无 pairs", file=sys.stderr)
            sys.exit(1)
        features_list = []
        for i, p in enumerate(pairs[: args.num_funcs]):
            feat = _make_feature_from_pair(p)
            feat["name"] = f"func_{i}"
            feat["features"]["multimodal"] = p.get("feature1", p)
            features_list.append(feat)
    else:
        # 生成 minimal 合成特征
        from features.models.multimodal_fusion import get_default_vocab

        vocab = get_default_vocab()
        vocab_keys = [k for k in vocab if k not in ("[PAD]", "[UNK]")]
        if not vocab_keys:
            vocab_keys = ["COPY", "INT_ADD", "INT_SUB"]

        features_list = []
        for i in range(args.num_funcs):
            tok = vocab_keys[i % len(vocab_keys)]
            tokens = [tok] * 16
            node_feats = [{"pcode_opcodes": [vocab_keys[j % len(vocab_keys)]]} for j in range(6)]
            features_list.append({
                "name": f"func_{i}",
                "features": {
                    "multimodal": {
                        "graph": {
                            "num_nodes": 6,
                            "edge_index": [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]],
                            "node_list": [f"bb{k}" for k in range(6)],
                            "node_features": node_feats,
                        },
                        "sequence": {"pcode_tokens": tokens, "jump_mask": [0] * 16, "seq_len": 16},
                    }
                },
            })

    features = {"functions": features_list}
    emb1 = embed_batch(features, model_path=args.model_path)
    emb2 = embed_batch(features, model_path=args.model_path)

    if len(emb1) != len(features_list) or len(emb2) != len(features_list):
        print("错误: 嵌入数量与输入不一致", file=sys.stderr)
        sys.exit(1)

    print("=== 嵌入一致性验证 ===\n")
    print(f"模型路径: {args.model_path or os.environ.get('SEMPATCH_MODEL_PATH', '(未指定，使用随机/环境变量)')}\n")

    # 同函数两次嵌入
    same_sims = []
    for i, (e1, e2) in enumerate(zip(emb1, emb2)):
        s = _cosine_sim(e1["vector"], e2["vector"])
        same_sims.append(s)
        print(f"同函数 {e1['name']} 两次嵌入相似度: {s:.6f}")
    avg_same = sum(same_sims) / len(same_sims) if same_sims else 0
    print(f"\n平均同函数相似度: {avg_same:.6f} (期望接近 1.0)\n")

    # 不同函数
    diff_sims = []
    for i in range(len(emb1)):
        for j in range(i + 1, len(emb2)):
            s = _cosine_sim(emb1[i]["vector"], emb2[j]["vector"])
            diff_sims.append(s)
            print(f"不同函数 {emb1[i]['name']} vs {emb2[j]['name']} 相似度: {s:.6f}")
    avg_diff = sum(diff_sims) / len(diff_sims) if diff_sims else 0
    print(f"\n平均不同函数相似度: {avg_diff:.6f} (期望低于同函数)")

    print("\n--- 结论 ---")
    if avg_same > 0.99:
        print("同函数嵌入一致性良好 (相似度 > 0.99)")
    else:
        print(f"同函数嵌入一致性: {avg_same:.4f} (未训练时可能 < 1.0)")
    if avg_diff < avg_same:
        print("不同函数相似度低于同函数，符合预期")
    else:
        print("不同函数相似度未明显低于同函数（未训练模型时常见）")


if __name__ == "__main__":
    main()
