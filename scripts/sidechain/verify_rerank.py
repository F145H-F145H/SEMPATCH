#!/usr/bin/env python3
"""
两阶段精排验证脚本：粗筛 + 精排流程快速自测。

从 query_features.json 取 1 个查询，粗筛得 Top-K 候选，精排后打印 Top-5。
需提前准备好 data/two_stage/ 下 library_features、query_features、library_safe_embeddings。

用法:
  python scripts/verify_rerank.py
  python scripts/verify_rerank.py --query-id "path|0x401000" --top-k 10
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from matcher.faiss_library import LibraryFaissIndex, retrieve_coarse
from matcher.rerank import compute_rerank_scores, load_candidate_features


def main():
    parser = argparse.ArgumentParser(description="两阶段精排验证")
    parser.add_argument(
        "--query-id",
        default=None,
        help="指定查询 function_id；未指定则从 ground_truth 取第一个",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="粗筛返回候选数",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="精排模型权重路径（默认 output/best_model.pth）",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(PROJECT_ROOT, "data", "two_stage"),
        help="两阶段数据目录",
    )
    args = parser.parse_args()

    qf_path = os.path.join(args.data_dir, "query_features.json")
    lib_feat_path = os.path.join(args.data_dir, "library_features.json")
    emb_path = os.path.join(args.data_dir, "library_safe_embeddings.json")
    gt_path = os.path.join(args.data_dir, "ground_truth.json")

    for p, name in [
        (qf_path, "query_features.json"),
        (lib_feat_path, "library_features.json"),
        (emb_path, "library_safe_embeddings.json"),
    ]:
        if not os.path.isfile(p):
            print(f"错误: 缺少 {name}，请先运行 prepare_two_stage_data 与 build_library_features、build_embeddings_db")
            sys.exit(1)

    with open(qf_path, encoding="utf-8") as f:
        query_features = json.load(f)
    with open(gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    query_id = args.query_id
    if not query_id:
        valid = [q for q in ground_truth if q in query_features]
        if not valid:
            print("错误: ground_truth 中无有效查询")
            sys.exit(1)
        query_id = valid[0]

    if query_id not in query_features:
        print(f"错误: 查询 {query_id} 不在 query_features 中")
        sys.exit(1)

    mm = query_features[query_id]
    idx = LibraryFaissIndex(emb_path)
    candidates = retrieve_coarse(mm, idx, k=args.top_k)
    if not candidates:
        print("粗筛未返回候选")
        sys.exit(0)

    cand_features = load_candidate_features(candidates, lib_feat_path)
    if not cand_features:
        print("无法加载候选特征")
        sys.exit(1)

    model_path = args.model_path or os.path.join(PROJECT_ROOT, "output", "best_model.pth")
    scores = compute_rerank_scores(mm, cand_features, model_path=model_path)

    positives = ground_truth.get(query_id, [])

    print(f"查询: {query_id}")
    print(f"粗筛候选数: {len(candidates)}, 精排后 Top-5:")
    for i, (cid, score) in enumerate(scores[:5], 1):
        mark = " [正样本]" if cid in positives else ""
        print(f"  {i}. {cid}  score={score:.4f}{mark}")
    print("完成")


if __name__ == "__main__":
    main()
