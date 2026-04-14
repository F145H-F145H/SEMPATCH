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
            # L2 归一化
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 0:
                emb = emb / emb_norm
            index.add(emb)
            id_list.append(obj['function_id'])
    
    if index is None:
        raise ValueError(f"未在 {embeddings_path} 中找到任何嵌入向量")
    return index, id_list


# ----------------------------------------------------------------------
# 加载精排模型（根据你的实际模型修改）
# ----------------------------------------------------------------------
def load_rerank_model(model_path, device='cpu'):
    from matcher.rerank import RerankModel
    return RerankModel(model_path=model_path, device=device, prefer_cuda=(device == 'cuda'))

def get_query_embeddings_streaming(query_features_path, needed_ids):
    """
    流式读取 query_features.json，只返回 needed_ids 对应的 SAFE 嵌入。
    支持两种格式：
      1. 预计算格式：{"query_id": {"safe_embedding": [...]}}
      2. 直接列表格式：{"query_id": [...]}
    返回 {query_id: np.array(embedding)}，value 为 None 表示需要后续实时计算
    """
    result = {}
    with open(query_features_path, 'rb') as f:
        parser = ijson.kvitems(f, '')
        for qid, value in parser:
            if qid in needed_ids:
                if isinstance(value, list):
                    # 直接就是 embedding 数组
                    result[qid] = np.array(value, dtype=np.float32)
                elif isinstance(value, dict):
                    emb = value.get('safe_embedding')
                    if emb is not None:
                        result[qid] = np.array(emb, dtype=np.float32)
                    else:
                        # 多模态特征格式，标记为 None 等待实时计算
                        result[qid] = None
                if len([v for v in result.values() if v is not None]) + len([v for v in result.values() if v is None]) >= len(needed_ids):
                    # 所有 needed_ids 都已发现（不管有无 embedding），提前结束
                    if set(result.keys()) >= needed_ids:
                        break
    return result


def _compute_missing_embeddings(result_dict, query_features_path, safe_model_path, device='cpu'):
    """对缺失 safe_embedding 的查询，用 SafeEmbedder 实时计算嵌入。"""
    from features.baselines.safe import SafeEmbedder

    missing_ids = [qid for qid, emb in result_dict.items() if emb is None]
    if not missing_ids:
        return
    print(f"正在为 {len(missing_ids)} 个缺失 safe_embedding 的查询实时计算嵌入（SafeEmbedder）...")

    embedder = SafeEmbedder(model_path=safe_model_path, device=device, prefer_cuda=(device == 'cuda'))

    # 如果 embedder 模型加载失败，退出
    if embedder._model is None:
        print(f"错误: SafeEmbedder 模型加载失败 ({safe_model_path})，无法计算嵌入", file=sys.stderr)
        sys.exit(1)

    with open(query_features_path, encoding='utf-8') as f:
        all_features = json.load(f)

    batch = []
    batch_ids = []
    BATCH_SIZE = 256
    for qid in missing_ids:
        mm = all_features.get(qid, {})
        batch.append(mm)
        batch_ids.append(qid)
        if len(batch) >= BATCH_SIZE:
            vecs = embedder.embed_many(batch, batch_size=BATCH_SIZE)
            for q, v in zip(batch_ids, vecs):
                result_dict[q] = np.array(v, dtype=np.float32)
            batch, batch_ids = [], []
    if batch:
        vecs = embedder.embed_many(batch, batch_size=BATCH_SIZE)
        for q, v in zip(batch_ids, vecs):
            result_dict[q] = np.array(v, dtype=np.float32)

    computed = sum(1 for v in result_dict.values() if v is not None)
    print(f"嵌入计算完成，共 {computed} 个查询拥有嵌入")


# ----------------------------------------------------------------------
# 精排：对候选函数重排序（占位实现，请根据实际模型替换）
# ----------------------------------------------------------------------
def rerank_candidates(query_embedding, candidate_ids, feature_loader, rerank_model, device='cpu', query_mm=None):
    """对粗筛候选进行精排。"""
    if not candidate_ids:
        return []
    if isinstance(feature_loader, dict):
        cand_features = [(cid, feature_loader.get(cid, {})) for cid in candidate_ids]
    else:
        try:
            cand_features = [(cid, feature_loader.get(cid)) for cid in candidate_ids]
        except Exception:
            cand_features = [(cid, {}) for cid in candidate_ids]
    cand_features = [(cid, mm) for cid, mm in cand_features if mm and isinstance(mm, dict)]
    if not cand_features:
        return [(cid, 0.0) for cid in candidate_ids]
    if not query_mm:
        return [(cid, 0.0) for cid in candidate_ids]
    try:
        return rerank_model.score(query_mm, cand_features)
    except Exception as e:
        print(f'  [rerank error] {e}')
        return [(cid, 0.0) for cid in candidate_ids]

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
    parser.add_argument("--safe-model-path", default=None, help="SAFE 模型路径，用于实时计算缺失的查询嵌入")
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

    # 对缺失 safe_embedding 的查询实时计算
    _missing = [qid for qid, emb in query_embeddings.items() if emb is None]
    if _missing:
        safe_model = args.safe_model_path or os.path.join(PROJECT_ROOT, "output", "best_model.pth")
        if not os.path.isfile(safe_model):
            print(f"错误: 需要 SAFE 模型实时计算嵌入，但模型文件不存在: {safe_model}", file=sys.stderr)
            print("请使用 --safe-model-path 指定 SAFE 模型路径", file=sys.stderr)
            sys.exit(1)
        _compute_missing_embeddings(query_embeddings, qf_path, safe_model, device)

    # 清除可能残留的 None 项
    query_embeddings = {k: v for k, v in query_embeddings.items() if v is not None}
    print(f"成功提取 {len(query_embeddings)} 个查询嵌入")

    # 6. 逐查询评估
    ranked_ids_per_query = {}
    coarse_hit_count = 0
    rerank_skipped_count = 0
    tied_top_count = 0
    rerank_ran_count = 0

    print('加载查询多模态特征...')
    import ijson as _ijson
    _query_mm_cache = {}
    with open(qf_path, 'rb') as _f:
        for _qid, _val in _ijson.kvitems(_f, ''):
            if _qid in all_query_ids and isinstance(_val, dict):
                _query_mm_cache[_qid] = _val
    print(f'  已加载 {len(_query_mm_cache)} 个查询的多模态特征')

    for qid in tqdm(all_query_ids, desc="评估查询"):
        if qid not in query_embeddings:
            ranked_ids_per_query[qid] = []
            rerank_skipped_count += 1
            continue

        q_emb = query_embeddings[qid].reshape(1, -1).astype(np.float32)
        positives = set(ground_truth.get(qid, []))

        # 粗筛
        # L2 归一化 query embedding
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm
        scores, indices = index.search(q_emb, args.coarse_k)
        coarse_ids = [lib_ids[idx] for idx in indices[0] if idx != -1]
        if positives and any(cid in positives for cid in coarse_ids):
            coarse_hit_count += 1

        if not coarse_ids:
            ranked_ids_per_query[qid] = []
            rerank_skipped_count += 1
            continue

        # 精排
        reranked = rerank_candidates(q_emb[0], coarse_ids, feature_loader, rerank_model, device, query_mm=_query_mm_cache.get(qid, {}))
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