#!/usr/bin/env python3
"""
两阶段流水线评估（流式加载版，SQLite 特征数据库支持）
- 粗筛：FAISS 索引从 library_safe_embeddings.json 流式构建
- 精排：加载 MultiModalFusionModel，对候选重排序
- 特征库：支持从 SQLite 数据库按需读取（避免 OOM），或回退到全量 JSON
"""
import argparse
import json
import ijson
import os
import sys
import sqlite3
import numpy as np
import faiss
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

_DEFAULT_MAX_INPUT_BYTES = 256 * 1024 * 1024


# ----------------------------------------------------------------------
# SQLite 特征加载器
# ----------------------------------------------------------------------
class SQLiteFeatureLoader:
    """从 SQLite 数据库按需加载函数特征"""
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        # 快速测试连接
        self.conn.execute("SELECT 1 FROM features LIMIT 1")

    def get(self, func_id):
        """返回特征的 Python 字典，若不存在返回 None"""
        row = self.conn.execute(
            "SELECT features_json FROM features WHERE function_id = ?",
            (func_id,)
        ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def close(self):
        if self.conn:
            self.conn.close()


# ----------------------------------------------------------------------
# 流式构建 FAISS 索引（适配 {"functions": [...]} 格式）
# ----------------------------------------------------------------------
def build_faiss_index_streaming(embeddings_path, embed_dim=None):
    """
    从大型 JSON 文件流式构建 FAISS 索引。
    文件格式：{"functions": [{"function_id": "...", "vector": [...]}, ...]}
    若 embed_dim 为 None，则自动从第一个对象推断。
    返回 (faiss.Index, id_list)
    """
    index = None
    id_list = []
    
    with open(embeddings_path, 'rb') as f:
        parser = ijson.items(f, 'functions.item')
        for obj in parser:
            vec = obj['vector']
            if index is None:
                embed_dim = len(vec)
                index = faiss.IndexFlatIP(embed_dim)   # 内积索引（余弦相似度）
            emb = np.array(vec, dtype=np.float32).reshape(1, -1)
            index.add(emb)
            id_list.append(obj['function_id'])
    
    if index is None:
        raise ValueError(f"未在 {embeddings_path} 中找到任何嵌入向量")
    return index, id_list


# ----------------------------------------------------------------------
# 加载精排模型（根据你的实际模型修改）
# ----------------------------------------------------------------------
def load_rerank_model(model_path, device='cpu'):
    """加载 MultiModalFusionModel 精排模型"""
    try:
        from models.fusion import MultiModalFusionModel
    except ImportError:
        print("错误: 无法导入 models.fusion.MultiModalFusionModel，请检查路径", file=sys.stderr)
        sys.exit(1)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        model = MultiModalFusionModel(**config)
    elif 'model_state_dict' in checkpoint:
        model = MultiModalFusionModel(embed_dim=768, num_modalities=2)
    else:
        model = MultiModalFusionModel(embed_dim=768, num_modalities=2)
    
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ----------------------------------------------------------------------
# 流式提取查询的 SAFE 嵌入（适配 {"query_id": {"safe_embedding": [...]}} 格式）
# ----------------------------------------------------------------------
def get_query_embeddings_streaming(query_features_path, needed_ids):
    """
    流式读取 query_features.json，只返回 needed_ids 对应的 SAFE 嵌入。
    假设格式：{"query_id": {"safe_embedding": [...], ...}}
    返回 {query_id: np.array(embedding)}
    """
    result = {}
    with open(query_features_path, 'rb') as f:
        parser = ijson.kvitems(f, '')
        for qid, value in parser:
            if qid in needed_ids:
                emb = value.get('safe_embedding')
                if emb is None:
                    if isinstance(value, list):
                        emb = value
                    else:
                        print(f"警告: 查询 {qid} 缺少 safe_embedding 字段", file=sys.stderr)
                        continue
                result[qid] = np.array(emb, dtype=np.float32)
                if len(result) == len(needed_ids):
                    break
    missing = needed_ids - result.keys()
    if missing:
        print(f"警告: 以下查询在 query_features.json 中缺失 safe_embedding: {missing}", file=sys.stderr)
    return result


# ----------------------------------------------------------------------
# 精排：对候选函数重排序（占位实现，请根据实际模型替换）
# ----------------------------------------------------------------------
def rerank_candidates(query_embedding, candidate_ids, feature_loader, rerank_model, device='cpu'):
    """
    对粗筛候选进行精排。
    参数：
        query_embedding: np.array, 查询的 SAFE 嵌入
        candidate_ids: list of str, 候选函数 ID
        feature_loader: SQLiteFeatureLoader 或 dict（兼容旧内存字典）
        rerank_model: PyTorch 模型
    返回：[(id, score), ...] 按分数降序排列
    """
    if not candidate_ids:
        return []
    
    # TODO: 替换为实际模型推理代码
    # 占位实现：返回原始顺序，分数均为 0.0
    # 实际使用时，请根据 MultiModalFusionModel 的输入格式，从 feature_loader 获取特征并计算分数
    return [(cid, 0.0) for cid in candidate_ids]


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------
def _human_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(x)} B"
            return f"{x:.1f} {unit}"
        x /= 1024.0
    return f"{x:.1f} TiB"


def _refuse_unbounded_json_loads(paths, max_bytes, allow_large):
    if allow_large:
        return
    bad = []
    for path, label in paths:
        try:
            sz = os.path.getsize(path)
        except OSError:
            continue
        if sz > max_bytes:
            bad.append((label, path, sz))
    if not bad:
        return
    lines = ["错误: 以下输入文件过大，整文件加载极易 OOM："]
    for label, path, sz in bad:
        lines.append(f"  - {label}: {_human_bytes(sz)}  ({path})")
    lines.append(f"CLI 默认拒绝单文件大于 {_human_bytes(max_bytes)} 的输入。若确认内存充足，请添加 --allow-large-inputs。")
    print("\n".join(lines), file=sys.stderr)
    sys.exit(2)


def _compute_metrics_for_k(ranked_ids_per_query, ground_truth, k):
    n_queries = len(ranked_ids_per_query)
    if n_queries == 0:
        return {"recall_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0}
    recall_sum = 0.0
    precision_sum = 0.0
    mrr_sum = 0.0
    for qid, ranked in ranked_ids_per_query.items():
        positives = set(ground_truth.get(qid, []))
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


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="两阶段流水线评估（流式加载 + SQLite 特征库）")
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--query-features", default=None)
    parser.add_argument("--library-embeddings", default=None)
    parser.add_argument("--library-features", default=None, help="JSON 特征文件路径（若不使用 SQLite）")
    parser.add_argument("--library-features-db", default=None, help="SQLite 特征数据库路径（推荐）")
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data", "two_stage"))
    parser.add_argument("--allow-large-inputs", action="store_true")
    parser.add_argument("--max-input-bytes", type=int, default=_DEFAULT_MAX_INPUT_BYTES)
    parser.add_argument("--coarse-k", type=int, default=100)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--safe-model-path", default=None)  # 未使用，粗筛用 FAISS
    parser.add_argument("-k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir
    gt_path = args.ground_truth or os.path.join(data_dir, "ground_truth.json")
    qf_path = args.query_features or os.path.join(data_dir, "query_features.json")
    le_path = args.library_embeddings or os.path.join(data_dir, "library_safe_embeddings.json")
    lf_path = args.library_features or os.path.join(data_dir, "library_features.json")
    db_path = args.library_features_db or None

    # 检查必需文件
    required_files = [
        (gt_path, "ground_truth"),
        (qf_path, "query_features"),
        (le_path, "library_embeddings"),
    ]
    # 若未提供数据库，则 JSON 特征文件必须存在
    if not db_path:
        required_files.append((lf_path, "library_features"))

    for p, name in required_files:
        if not os.path.isfile(p):
            print(f"错误: 文件不存在 {p} ({name})", file=sys.stderr)
            sys.exit(1)

    # 体积检查
    files_to_check = [(gt_path, "ground_truth"), (qf_path, "query_features"), (le_path, "library_embeddings")]
    if not db_path:
        files_to_check.append((lf_path, "library_features"))
    _refuse_unbounded_json_loads(
        files_to_check,
        max_bytes=args.max_input_bytes,
        allow_large=args.allow_large_inputs,
    )

    # 1. 加载 ground_truth
    with open(gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)
    if not isinstance(ground_truth, dict):
        print("错误: ground_truth 应为字典", file=sys.stderr)
        sys.exit(1)

    all_query_ids = list(ground_truth.keys())
    if args.max_queries:
        all_query_ids = all_query_ids[:args.max_queries]
    needed_ids = set(all_query_ids)

    # 2. 流式构建 FAISS 索引
    print("正在流式构建 FAISS 索引（library_safe_embeddings.json）...")
    index, lib_ids = build_faiss_index_streaming(le_path)
    embed_dim = index.d
    print(f"FAISS 索引构建完成，共 {len(lib_ids)} 条，维度 {embed_dim}")

    # 3. 准备特征加载器
    use_sqlite = db_path and os.path.isfile(db_path)
    if use_sqlite:
        print(f"使用 SQLite 特征数据库: {db_path}")
        feature_loader = SQLiteFeatureLoader(db_path)
        total_features = feature_loader.conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        print(f"数据库包含 {total_features} 条特征记录")
    else:
        print("警告: 未提供 SQLite 数据库，将全量加载 JSON 文件（可能 OOM）")
        with open(lf_path, 'r', encoding='utf-8') as f:
            library_features = json.load(f)
        if isinstance(library_features, list):
            library_features = {item['function_id']: item.get('features', item) for item in library_features}
        elif not isinstance(library_features, dict):
            print("错误: library_features.json 格式不支持", file=sys.stderr)
            sys.exit(1)
        print(f"JSON 加载完成，共 {len(library_features)} 条")
        feature_loader = library_features  # 直接作为字典使用

    # 4. 加载精排模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = args.model_path or os.path.join(PROJECT_ROOT, "output", "best_model.pth")
    if not os.path.isfile(model_path):
        print(f"错误: 精排模型不存在 {model_path}", file=sys.stderr)
        sys.exit(1)
    print(f"加载精排模型: {model_path} (device={device})")
    rerank_model = load_rerank_model(model_path, device=device)

    # 5. 流式提取查询嵌入
    print("流式提取查询 SAFE 嵌入...")
    query_embeddings = get_query_embeddings_streaming(qf_path, needed_ids)
    print(f"成功提取 {len(query_embeddings)} 个查询嵌入")

    # 6. 逐查询评估
    ranked_ids_per_query = {}
    coarse_hit_count = 0
    rerank_skipped_count = 0
    tied_top_count = 0
    rerank_ran_count = 0

    for qid in tqdm(all_query_ids, desc="评估查询"):
        if qid not in query_embeddings:
            ranked_ids_per_query[qid] = []
            rerank_skipped_count += 1
            continue

        q_emb = query_embeddings[qid].reshape(1, -1).astype(np.float32)
        positives = set(ground_truth.get(qid, []))

        # 粗筛
        scores, indices = index.search(q_emb, args.coarse_k)
        coarse_ids = [lib_ids[idx] for idx in indices[0] if idx != -1]
        if positives and any(cid in positives for cid in coarse_ids):
            coarse_hit_count += 1

        if not coarse_ids:
            ranked_ids_per_query[qid] = []
            rerank_skipped_count += 1
            continue

        # 精排
        reranked = rerank_candidates(q_emb[0], coarse_ids, feature_loader, rerank_model, device)
        rerank_ran_count += 1
        if not reranked:
            ranked_ids_per_query[qid] = []
            rerank_skipped_count += 1
            continue

        ranked_ids_per_query[qid] = [cid for cid, _ in reranked]
        if len(reranked) >= 2 and abs(reranked[0][1] - reranked[1][1]) < 1e-9:
            tied_top_count += 1

    # 清理数据库连接
    if use_sqlite:
        feature_loader.close()

    # 计算指标
    total_queries = len(all_query_ids)
    for k_val in args.k:
        metrics = _compute_metrics_for_k(ranked_ids_per_query, ground_truth, k_val)
        print(f"k={k_val}: Recall@K={metrics['recall_at_k']:.4f}, Precision@K={metrics['precision_at_k']:.4f}, MRR={metrics['mrr']:.4f}")

    # 诊断面板
    coarse_hit_rate = coarse_hit_count / total_queries if total_queries else 0
    fallback_rate = rerank_skipped_count / total_queries if total_queries else 0
    tied_top_rate = tied_top_count / rerank_ran_count if rerank_ran_count else 0
    print("\n--- 错误分析面板 ---")
    print(f"  coarse_hit_rate  = {coarse_hit_rate:.4f}  ({coarse_hit_count}/{total_queries})")
    print(f"  fallback_rate    = {fallback_rate:.4f}  ({rerank_skipped_count}/{total_queries})")
    print(f"  tied_top_rate    = {tied_top_rate:.4f}  ({tied_top_count}/{rerank_ran_count})")

    # 输出 JSON
    if args.output:
        try:
            from experiment_meta import collect_metadata
        except ImportError:
            def collect_metadata(_):
                return {}
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        output_data = {
            "metrics": {f"k={k}": _compute_metrics_for_k(ranked_ids_per_query, ground_truth, k) for k in args.k},
            "diagnostics": {
                "coarse_hit_rate": coarse_hit_rate,
                "fallback_rate": fallback_rate,
                "tied_top_rate": tied_top_rate,
                "total_queries": total_queries,
                "coarse_hit_count": coarse_hit_count,
                "rerank_skipped_count": rerank_skipped_count,
                "tied_top_count": tied_top_count,
                "rerank_ran_count": rerank_ran_count,
            },
            "metadata": collect_metadata(args),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"结果已写入 {out_path}")


if __name__ == "__main__":
    main()