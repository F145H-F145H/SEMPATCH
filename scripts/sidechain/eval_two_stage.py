#!/usr/bin/env python3
"""
两阶段流水线评估：ground_truth + TwoStagePipeline，计算 Recall@K、Precision@K、MRR。

对 ground_truth 中每个查询调用 retrieve_and_rerank，取 Top-K 候选，
根据 ground_truth[query_id] 正样本列表计算指标。与 eval_bcsd 逻辑一致。
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# 单文件 json.load 前体积上限：超过则默认拒绝，避免数 GB 的 two_stage 产物拖垮宿主机 / OOM。
_DEFAULT_MAX_INPUT_BYTES = 256 * 1024 * 1024


def _human_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(x)} B"
            return f"{x:.1f} {unit}"
        x /= 1024.0
    return f"{x:.1f} TiB"


def _refuse_unbounded_json_loads(
    paths: list[tuple[str, str]],
    *,
    max_bytes: int,
    allow_large: bool,
) -> None:
    """在整文件 json.load 之前检查体积；防止误对巨型 two_stage 目录跑 CLI。"""
    if allow_large:
        return
    bad: list[tuple[str, str, int]] = []
    for path, label in paths:
        try:
            sz = os.path.getsize(path)
        except OSError:
            continue
        if sz > max_bytes:
            bad.append((label, path, sz))
    if not bad:
        return
    lines = [
        "错误: 以下输入文件过大，整文件加载极易 OOM 或拖垮 IDE/终端：",
    ]
    for label, path, sz in bad:
        lines.append(f"  - {label}: {_human_bytes(sz)}  ({path})")
    lines.append(
        f"CLI 默认拒绝单文件大于 {_human_bytes(max_bytes)} 的输入。"
        "若你确认内存与磁盘充足，请添加 --allow-large-inputs。"
        "冒烟/开发请使用: --data-dir benchmarks/smoke/two_stage"
    )
    print("\n".join(lines), file=sys.stderr)
    sys.exit(2)


def _compute_metrics_for_k(
    ranked_ids_per_query: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    k: int,
) -> dict[str, float]:
    """
    对每个查询的 Top-K 候选计算 Recall@K、Precision@K、MRR。
    ranked_ids_per_query: {query_id: [candidate_id, ...]} 按精排得分降序
    ground_truth: {query_id: [positive_id, ...]}
    """
    n_queries = len(ranked_ids_per_query)
    if n_queries == 0:
        return {"recall_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0}

    recall_sum = 0.0
    precision_sum = 0.0
    mrr_sum = 0.0

    for query_id, ranked in ranked_ids_per_query.items():
        positives = set(ground_truth.get(query_id, []))
        top_k = ranked[:k]

        hits = sum(1 for cid in top_k if cid in positives)
        first_rank = None
        for rank, cid in enumerate(top_k, start=1):
            if cid in positives:
                first_rank = rank
                break

        recall_sum += 1.0 if hits > 0 else 0.0
        precision_sum += hits / k if k > 0 else 0.0
        mrr_sum += 1.0 / first_rank if first_rank is not None else 0.0

    return {
        "recall_at_k": recall_sum / n_queries,
        "precision_at_k": precision_sum / n_queries,
        "mrr": mrr_sum / n_queries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="两阶段流水线评估：Recall@K、Precision@K、MRR"
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="ground_truth.json 路径",
    )
    parser.add_argument(
        "--query-features",
        default=None,
        help="query_features.json 路径",
    )
    parser.add_argument(
        "--library-embeddings",
        default=None,
        help="library_safe_embeddings.json 路径",
    )
    parser.add_argument(
        "--library-features",
        default=None,
        help="library_features.json 路径",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(PROJECT_ROOT, "data", "two_stage"),
        help="两阶段数据目录（默认 data/two_stage）。冒烟请用 benchmarks/smoke/two_stage",
    )
    parser.add_argument(
        "--allow-large-inputs",
        action="store_true",
        help="允许单文件超过 256MiB 的 ground_truth/query/library JSON（全量加载，OOM 风险自负）",
    )
    parser.add_argument(
        "--max-input-bytes",
        type=int,
        default=_DEFAULT_MAX_INPUT_BYTES,
        help="单文件体积上限（默认 256MiB）；仅在不加 --allow-large-inputs 时生效",
    )
    parser.add_argument(
        "--coarse-k",
        type=int,
        default=100,
        help="粗筛 Top-K 候选数",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="精排模型路径（默认 output/best_model.pth）",
    )
    parser.add_argument(
        "-k",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="评估 K 值列表",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="结果输出路径（可选，JSON 格式）",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="最大评估查询数（用于快速验证，默认不限制）",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    gt_path = args.ground_truth or os.path.join(data_dir, "ground_truth.json")
    qf_path = args.query_features or os.path.join(data_dir, "query_features.json")
    le_path = args.library_embeddings or os.path.join(
        data_dir, "library_safe_embeddings.json"
    )
    lf_path = args.library_features or os.path.join(
        data_dir, "library_features.json"
    )

    inputs_meta = [
        (gt_path, "ground_truth.json"),
        (qf_path, "query_features.json"),
        (le_path, "library_safe_embeddings.json"),
        (lf_path, "library_features.json"),
    ]
    for p, name in inputs_meta:
        if not os.path.isfile(p):
            print(f"错误: 文件不存在 {p} ({name})", file=sys.stderr)
            sys.exit(1)

    max_b = max(1, int(args.max_input_bytes))
    _refuse_unbounded_json_loads(
        inputs_meta,
        max_bytes=max_b,
        allow_large=bool(args.allow_large_inputs),
    )

    with open(gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)
    if not isinstance(ground_truth, dict):
        print("错误: ground_truth 应为 {query_id: [positive_id, ...]}", file=sys.stderr)
        sys.exit(1)

    with open(qf_path, encoding="utf-8") as f:
        query_features = json.load(f)
    valid_query_ids = [
        qid for qid in ground_truth if qid in query_features
    ]
    if args.max_queries is not None and args.max_queries > 0:
        valid_query_ids = valid_query_ids[: args.max_queries]
    if not valid_query_ids:
        print("警告: 无有效查询（ground_truth 与 query_features 无交集）", file=sys.stderr)
        for k_val in args.k:
            print(f"k={k_val}: Recall@K=0.0000, Precision@K=0.0000, MRR=0.0000")
        sys.exit(0)

    model_path = args.model_path or os.path.join(PROJECT_ROOT, "output", "best_model.pth")
    from matcher.two_stage import TwoStagePipeline

    pipeline = TwoStagePipeline(
        library_safe_embeddings_path=le_path,
        library_features_path=lf_path,
        query_features_path=qf_path,
        coarse_k=args.coarse_k,
        rerank_model_path=model_path,
    )

    # ------------------------------------------------------------------
    # 逐查询执行：粗筛 + 精排分步，收集每查询诊断信息
    # ------------------------------------------------------------------
    ranked_ids_per_query: dict[str, list[str]] = {}
    total_queries = 0
    coarse_hit_count = 0        # 粗筛候选中包含正样本的查询数
    rerank_skipped_count = 0    # 精排被跳过的查询数（无候选 / 特征缺失）
    tied_top_count = 0          # Top-1 与 Top-2 精排得分并列的查询数
    rerank_ran_count = 0        # 精排实际执行的查询数

    for query_id in valid_query_ids:
        total_queries += 1
        positives = set(ground_truth.get(query_id) or [])

        # 1. 粗筛
        coarse_ids = pipeline.retrieve(query_id)
        if positives and any(cid in positives for cid in coarse_ids):
            coarse_hit_count += 1

        # 2. 精排
        if not coarse_ids:
            ranked_ids_per_query[query_id] = []
            rerank_skipped_count += 1
            continue

        reranked = pipeline.rerank(query_id, coarse_ids)
        rerank_ran_count += 1

        if not reranked:
            ranked_ids_per_query[query_id] = []
            rerank_skipped_count += 1
            continue

        ranked_ids_per_query[query_id] = [cid for cid, _ in reranked]

        # 3. 检测并列 Top
        if len(reranked) >= 2:
            s0, s1 = reranked[0][1], reranked[1][1]
            if abs(s0 - s1) < 1e-9:
                tied_top_count += 1

    # ------------------------------------------------------------------
    # 聚合指标
    # ------------------------------------------------------------------
    results: dict[str, dict[str, float]] = {}
    for k_val in args.k:
        if k_val < 1:
            continue
        metrics = _compute_metrics_for_k(
            ranked_ids_per_query, ground_truth, k_val
        )
        results[f"k={k_val}"] = metrics

    for k_label, m in results.items():
        print(
            f"{k_label}: Recall@K={m['recall_at_k']:.4f}, "
            f"Precision@K={m['precision_at_k']:.4f}, MRR={m['mrr']:.4f}"
        )

    # ------------------------------------------------------------------
    # 错误分析面板
    # ------------------------------------------------------------------
    coarse_hit_rate = (coarse_hit_count / total_queries) if total_queries > 0 else 0.0
    fallback_rate = (rerank_skipped_count / total_queries) if total_queries > 0 else 0.0
    tied_top_rate = (tied_top_count / rerank_ran_count) if rerank_ran_count > 0 else 0.0

    diagnostics = {
        "coarse_hit_rate": round(coarse_hit_rate, 4),
        "fallback_rate": round(fallback_rate, 4),
        "tied_top_rate": round(tied_top_rate, 4),
        "total_queries": total_queries,
        "coarse_hit_count": coarse_hit_count,
        "rerank_skipped_count": rerank_skipped_count,
        "tied_top_count": tied_top_count,
        "rerank_ran_count": rerank_ran_count,
    }

    # 终端诊断摘要
    print()
    print("--- 错误分析面板 ---")
    print(f"  coarse_hit_rate  = {coarse_hit_rate:.4f}  ({coarse_hit_count}/{total_queries} 个查询的正样本在粗筛 Top-{args.coarse_k} 中)")
    if coarse_hit_rate < 0.8:
        print(f"    ⚠ 粗筛漏召回高 → 增大 --coarse-k 或优化 SAFE 嵌入权重")
    print(f"  fallback_rate    = {fallback_rate:.4f}  ({rerank_skipped_count}/{total_queries} 个查询未经过精排)")
    if fallback_rate > 0.05:
        print(f"    ⚠ 精排跳过率高 → 检查候选特征是否缺失，或库特征文件过大触发内存降级")
    print(f"  tied_top_rate    = {tied_top_rate:.4f}  ({tied_top_count}/{rerank_ran_count} 个精排查询 Top-1/Top-2 并列)")
    if tied_top_rate > 0.1:
        print(f"    ⚠ 并列率高 → 精排模型区分度不足，考虑增加训练数据或调整模型架构")

    # ------------------------------------------------------------------
    # 输出 JSON（含诊断字段）
    # ------------------------------------------------------------------
    if args.output:
        from experiment_meta import collect_metadata
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        output_data = {
            "metrics": results,
            "diagnostics": diagnostics,
            "metadata": collect_metadata(args),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"结果已写入 {out_path}")


if __name__ == "__main__":
    main()
