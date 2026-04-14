#!/usr/bin/env python3
"""粗筛（FAISS）独立评估脚本，用于论文实验。"""

import json, os, sys, time
import numpy as np
import faiss
import ijson
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from features.baselines.safe import SafeEmbedder

def build_faiss_index(embeddings_path):
    lib_ids, lib_vecs = [], []
    with open(embeddings_path, 'rb') as f:
        for obj in ijson.items(f, 'functions.item'):
            lib_ids.append(obj['function_id'])
            lib_vecs.append(obj['vector'])
    mat = np.array(lib_vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.where(norms > 0, norms, 1)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    return index, lib_ids

def load_query_embeddings(qf_path, gt_keys, safe_model_path, device='cuda'):
    embedder = SafeEmbedder(model_path=safe_model_path, device=device, prefer_cuda=(device=='cuda'))
    query_mm = {}
    with open(qf_path, 'rb') as f:
        for qid, val in ijson.kvitems(f, ''):
            if qid in gt_keys and isinstance(val, dict):
                query_mm[qid] = val
    qids = list(query_mm.keys())
    mm_list = [query_mm[qid] for qid in qids]
    vecs = embedder.embed_many(mm_list, batch_size=256)
    return {qid: np.array(v, dtype=np.float32) for qid, v in zip(qids, vecs)}

def compute_metrics(ranked_per_query, ground_truth, k):
    n = len(ranked_per_query)
    if n == 0:
        return 0.0, 0.0, 0.0
    recall_sum = prec_sum = mrr_sum = 0.0
    for qid, ranked in ranked_per_query.items():
        positives = set(ground_truth.get(qid, []))
        top_k = ranked[:k]
        hits = sum(1 for cid in top_k if cid in positives)
        recall_sum += 1.0 if hits > 0 else 0.0
        prec_sum += hits / k
        for rank, cid in enumerate(top_k, 1):
            if cid in positives:
                mrr_sum += 1.0 / rank
                break
    return recall_sum / n, prec_sum / n, mrr_sum / n

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/two_stage")
    parser.add_argument("--library-embeddings", default=None)
    parser.add_argument("--safe-model-path", default="output/safe_best_model.pt")
    parser.add_argument("--coarse-k", type=int, default=100)
    parser.add_argument("-k", nargs="+", type=int, default=[1, 5, 10, 20, 50, 100])
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    le_path = args.library_embeddings or os.path.join(args.data_dir, "library_safe_embeddings.json")
    qf_path = os.path.join(args.data_dir, "query_features.json")
    gt_path = os.path.join(args.data_dir, "ground_truth.json")

    # 1. Build FAISS
    t0 = time.time()
    print("Building FAISS index...")
    index, lib_ids = build_faiss_index(le_path)
    lib_id_set = set(lib_ids)
    print(f"  {len(lib_ids)} vectors, dim={index.d}, {time.time()-t0:.1f}s")

    # 2. Load ground truth
    with open(gt_path) as f:
        gt = json.load(f)
    # Filter to queries whose targets exist in FAISS
    gt_filtered = {qid: targets for qid, targets in gt.items()
                   if set(targets) & lib_id_set}
    if args.max_queries:
        gt_filtered = dict(list(gt_filtered.items())[:args.max_queries])
    print(f"  {len(gt_filtered)} queries (targets in FAISS)")

    # 3. Compute query embeddings
    t0 = time.time()
    print("Computing query SAFE embeddings...")
    qid_keys = set(gt_filtered.keys())
    query_embs = load_query_embeddings(qf_path, qid_keys, args.safe_model_path)
    print(f"  {len(query_embs)} embeddings, {time.time()-t0:.1f}s")

    # 4. FAISS search
    t0 = time.time()
    max_k = max(args.k)
    ranked_per_query = {}
    for qid, q_vec in tqdm(query_embs.items(), desc="Searching"):
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm
        scores, indices = index.search(q_vec.reshape(1, -1), max_k)
        ranked_per_query[qid] = [lib_ids[j] for j in indices[0] if j != -1]
    print(f"  Search done, {time.time()-t0:.1f}s")

    # 5. Compute metrics
    print(f"\n{'='*60}")
    print(f"{'K':<6} {'Recall@K':<12} {'Precision@K':<12} {'MRR':<12}")
    print(f"{'='*60}")
    results = {}
    for k in sorted(args.k):
        r, p, m = compute_metrics(ranked_per_query, gt_filtered, k)
        results[f"k={k}"] = {"recall_at_k": r, "precision_at_k": p, "mrr": m}
        print(f"@{k:<4} {r:.4f}        {p:.4f}        {m:.4f}")

    # 6. Additional diagnostics
    total = len(gt_filtered)
    coarse_hits = sum(1 for qid in gt_filtered
                      if set(ranked_per_query.get(qid, [])) & set(gt_filtered[qid]))
    print(f"\n--- Diagnostics ---")
    print(f"  Queries:         {total}")
    print(f"  Coarse hit rate: {coarse_hits/total:.4f} ({coarse_hits}/{total})")

    # 7. Save
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        output = {
            "metrics": results,
            "diagnostics": {"coarse_hit_rate": coarse_hits/total, "total_queries": total},
            "params": {"coarse_k": args.coarse_k, "k_values": args.k},
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()
