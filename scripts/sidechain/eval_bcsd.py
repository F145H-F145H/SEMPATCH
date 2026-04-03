#!/usr/bin/env python3
"""
BCSD 评估脚本：给定固件嵌入与漏洞库嵌入，计算 Recall@K、Precision@K、MRR。
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
from scipy.spatial.distance import cdist


def normalize_cve_field(raw: object) -> list[str]:
    """
    将嵌入条目中的 cve 字段规范为字符串列表（与 EmbeddingItem.cve 一致）。
    缺失 / null → []；str → 非空则单元素列表；list → 非空字符串元素列表。
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    if isinstance(raw, list):
        out: list[str] = []
        for x in raw:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    s = str(raw).strip()
    return [s] if s else []


def load_embeddings(path: str) -> tuple[list[str], np.ndarray, list[list[str]]]:
    """
    加载嵌入 JSON 文件，返回 (names, vectors, cve_lists)。
    cve 为可选字段；cve_lists 与 names 一一对应、同长，每项为 List[str]。
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    functions = data.get("functions") or []
    names: list[str] = []
    vectors_list: list[list[float]] = []
    cve_lists: list[list[str]] = []

    for item in functions:
        name = item.get("name", "")
        vec = item.get("vector")
        if vec is None:
            continue
        cve_lists.append(normalize_cve_field(item.get("cve")))
        names.append(name)
        vectors_list.append(vec)

    if not vectors_list:
        return names, np.array([]).reshape(0, 0), cve_lists

    vectors = np.array(vectors_list, dtype=np.float64)
    return names, vectors, cve_lists


def compute_top_k(
    query_vectors: np.ndarray, db_vectors: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    对每个 query 向量与所有 db 向量计算余弦相似度，返回 Top-K 索引和分数。
    返回 indices (N_q, k), scores (N_q, k)，按相似度降序。
    """
    if query_vectors.size == 0 or db_vectors.size == 0:
        return np.array([]).reshape(0, k), np.array([]).reshape(0, k)

    # cdist(metric='cosine') 返回 1 - 余弦相似度，转为相似度
    dist = cdist(query_vectors, db_vectors, metric="cosine")
    # 处理可能的 NaN（零向量导致）
    dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=1.0)
    similarity = 1.0 - dist

    actual_k = min(k, db_vectors.shape[0])
    top_indices = np.argsort(-similarity, axis=1)[:, :actual_k]

    if actual_k < k:
        # 填充至 k 列
        pad = np.full((query_vectors.shape[0], k - actual_k), -1, dtype=np.int64)
        top_indices = np.hstack([top_indices, pad])

    top_scores = np.take_along_axis(similarity, top_indices, axis=1)
    # 填充位置的分数为 0
    if actual_k < k:
        top_scores[:, actual_k:] = 0.0

    return top_indices, top_scores


def build_relevant_pairs(
    query_names: list[str], db_names: list[str]
) -> set[tuple[int, int]]:
    """
    构建 relevant_pairs：(query_idx, db_idx) 同名即相关。
    """
    pairs: set[tuple[int, int]] = set()
    for qi, qn in enumerate(query_names):
        for di, dn in enumerate(db_names):
            if qn == dn:
                pairs.add((qi, di))
    return pairs


def build_relevant_pairs_by_cve(
    query_cves: list[list[str]], db_cves: list[list[str]]
) -> set[tuple[int, int]]:
    """
    构建 relevant_pairs：(query_idx, db_idx) 当 query 与 db 的 CVE 集合交集非空时相关。
    用于 1-day 漏洞评估；支持每条目多个 CVE。
    """
    pairs: set[tuple[int, int]] = set()
    for qi, qc in enumerate(query_cves):
        qset = {x for x in qc if x}
        if not qset:
            continue
        for di, dc in enumerate(db_cves):
            dset = {x for x in dc if x}
            if qset & dset:
                pairs.add((qi, di))
    return pairs


def compute_metrics(
    query_names: list[str],
    db_names: list[str],
    top_k_indices: np.ndarray,
    relevant_pairs: set[tuple[int, int]],
    k: int,
) -> dict[str, float]:
    """
    计算 Recall@K、Precision@K、MRR。
    """
    n_queries = len(query_names)
    if n_queries == 0:
        return {"recall_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0}

    recall_sum = 0.0
    precision_sum = 0.0
    mrr_sum = 0.0

    actual_k = min(k, top_k_indices.shape[1])

    for qi in range(n_queries):
        hits = 0
        first_rank = None
        row = top_k_indices[qi]

        for rank, db_idx in enumerate(row[:actual_k], start=1):
            if db_idx < 0:
                break
            if (qi, int(db_idx)) in relevant_pairs:
                hits += 1
                if first_rank is None:
                    first_rank = rank

        # Recall@K: 至少一个 relevant 则 1，否则 0
        recall_sum += 1.0 if hits > 0 else 0.0

        # Precision@K: relevant 占比
        precision_sum += hits / actual_k if actual_k > 0 else 0.0

        # MRR: 第一个 relevant 的排名倒数
        mrr_sum += 1.0 / first_rank if first_rank is not None else 0.0

    return {
        "recall_at_k": recall_sum / n_queries,
        "precision_at_k": precision_sum / n_queries,
        "mrr": mrr_sum / n_queries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BCSD 评估：固件嵌入 vs 漏洞库嵌入，计算 Recall@K、Precision@K、MRR"
    )
    parser.add_argument(
        "--firmware-emb",
        required=True,
        help="待查询的嵌入数据库路径（JSON 格式）",
    )
    parser.add_argument(
        "--db-emb",
        required=True,
        help="漏洞库/参考嵌入数据库路径（JSON 格式）",
    )
    parser.add_argument(
        "-k",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="K 值列表，如 1 5 10",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="评估结果输出路径（可选，JSON 格式）",
    )
    parser.add_argument(
        "--mode",
        choices=["name", "cve"],
        default="name",
        help="相关性模式：name=同名即相关，cve=同 CVE 即相关（1-day 漏洞评估）",
    )
    args = parser.parse_args()

    firmware_path = os.path.abspath(args.firmware_emb)
    db_path = os.path.abspath(args.db_emb)

    if not os.path.isfile(firmware_path):
        print(f"错误: 文件不存在 {firmware_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(db_path):
        print(f"错误: 文件不存在 {db_path}", file=sys.stderr)
        sys.exit(1)

    query_names, query_vectors, query_cves = load_embeddings(firmware_path)
    db_names, db_vectors, db_cves = load_embeddings(db_path)

    if len(query_names) == 0:
        print("警告: 固件嵌入为空", file=sys.stderr)
    if len(db_names) == 0:
        print("警告: 漏洞库嵌入为空", file=sys.stderr)

    if args.mode == "cve":
        if all(not c for c in query_cves) and all(not c for c in db_cves):
            print(
                "警告: --mode cve 但嵌入中无有效 CVE（列表均为空），relevant_pairs 将为空",
                file=sys.stderr,
            )
        relevant_pairs = build_relevant_pairs_by_cve(query_cves, db_cves)
    else:
        relevant_pairs = build_relevant_pairs(query_names, db_names)
    results: dict[str, dict[str, float]] = {}

    for k_val in args.k:
        if k_val < 1:
            continue
        top_k_indices, _ = compute_top_k(query_vectors, db_vectors, k_val)
        metrics = compute_metrics(
            query_names, db_names, top_k_indices, relevant_pairs, k_val
        )
        results[f"k={k_val}"] = metrics

    # 打印
    for k_label, m in results.items():
        print(f"{k_label}: Recall@K={m['recall_at_k']:.4f}, Precision@K={m['precision_at_k']:.4f}, MRR={m['mrr']:.4f}")

    if args.output:
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已写入 {out_path}")


if __name__ == "__main__":
    main()
