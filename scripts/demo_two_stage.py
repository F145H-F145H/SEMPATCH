#!/usr/bin/env python3
"""
两阶段二进制相似度匹配 Benchmark Demo
=====================================
用 best_model.pth 对 binkit benchmark 做扩展正负对评估。

核心关注:
  - 所有 query×library 正负对的余弦相似度分布
  - AUC-ROC / AUC-PR / Recall@K / MRR
  - 粗筛 vs 精排的对比
  - 逐 query 精排排名明细
"""
import json
import os
import sys
import time

import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

DATA_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "dev_binkit", "split")
MODEL_PATH = os.path.join(PROJECT_ROOT, "output", "best_model.pth")


def sep(title: str):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def main():
    # ──────────────────────────────────────────────────────────────────
    # 1. 加载数据 & 模型
    # ──────────────────────────────────────────────────────────────────
    sep("1. 加载数据")

    with open(os.path.join(DATA_DIR, "ground_truth.json")) as f:
        ground_truth = json.load(f)
    with open(os.path.join(DATA_DIR, "query_features.json")) as f:
        query_features = json.load(f)
    with open(os.path.join(DATA_DIR, "library_features.json")) as f:
        library_features = json.load(f)
    with open(os.path.join(DATA_DIR, "library_safe_embeddings.json")) as f:
        library_embeddings = json.load(f)

    gt_qids = [qid for qid in ground_truth if qid in query_features and ground_truth[qid]]
    lid_list = sorted(library_features.keys())
    pos_set = {}
    for qid in gt_qids:
        pos_set[qid] = set(ground_truth[qid]) & set(lid_list)

    total_pos = sum(len(v) for v in pos_set.values())
    total_neg = len(gt_qids) * len(lid_list) - total_pos

    print(f"  queries: {len(gt_qids)}  library: {len(lid_list)}")
    print(f"  正对: {total_pos}   负对: {total_neg}   总计: {total_pos + total_neg}")

    # vocab
    from features.models.multimodal_fusion import get_default_vocab
    vocab = get_default_vocab()
    for mm in list(query_features.values()) + list(library_features.values()):
        seq = mm.get("sequence", {})
        for t in seq.get("pcode_tokens", []):
            if t and t not in vocab:
                vocab[t] = len(vocab)
        graph = mm.get("graph", {})
        for nf in graph.get("node_features", []):
            opcodes = nf if isinstance(nf, list) else nf.get("pcode_opcodes", [])
            for op in opcodes:
                if op and op not in vocab:
                    vocab[op] = len(vocab)
    pcode_vocab_size = max(len(vocab), 256)

    # model
    from features.models.multimodal_fusion import (
        MultiModalFusionModel,
        parse_multimodal_checkpoint,
        infer_use_dfg_from_state_dict,
    )

    raw = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    state_dict, meta = parse_multimodal_checkpoint(raw)
    use_dfg = meta.get("use_dfg", False) or infer_use_dfg_from_state_dict(state_dict)
    model = MultiModalFusionModel(pcode_vocab_size=pcode_vocab_size, use_dfg=use_dfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    epochs = meta.get("cli_args", {}).get("epochs", "?")
    ts = meta.get("timestamp", "?")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {total_params:,} params, dfg={use_dfg}, epochs={epochs}, saved={ts}")

    # ──────────────────────────────────────────────────────────────────
    # 2. 粗筛评估 (FAISS coarse)
    # ──────────────────────────────────────────────────────────────────
    sep("2. 粗筛评估  (SAFE embedding → FAISS Top-K)")

    from matcher.faiss_library import LibraryFaissIndex, retrieve_coarse

    faiss_index = LibraryFaissIndex(os.path.join(DATA_DIR, "library_safe_embeddings.json"))
    coarse_k = len(lid_list)  # 暴力全召回，看粗筛分数

    coarse_scores_pos = []
    coarse_scores_neg = []

    # 粗筛: 每个 query 通过 SAFE embedding 检索，得到所有 library 的排序
    # 记录每个 (query, library) 对的粗筛排名
    coarse_ranks = {}  # qid -> {lid: rank}
    for qid in gt_qids:
        mm = query_features[qid]
        coarse_ids = retrieve_coarse(mm, faiss_index, k=coarse_k)
        rank_map = {lid: i + 1 for i, lid in enumerate(coarse_ids)}
        coarse_ranks[qid] = rank_map

    # 按 rank 分档统计正/负命中
    rank_bins = [1, 5, 10]
    for k in rank_bins:
        pos_in = 0
        for qid in gt_qids:
            for lid in pos_set[qid]:
                if coarse_ranks[qid].get(lid, 999) <= k:
                    pos_in += 1
        print(f"  coarse Recall@{k}:  {pos_in}/{total_pos} = {pos_in/total_pos:.4f}")

    # ──────────────────────────────────────────────────────────────────
    # 3. 精排: 全量 query×library 对打分
    # ──────────────────────────────────────────────────────────────────
    sep("3. 精排评估  (MultiModalFusion → cosine similarity)")

    from features.models.multimodal_fusion import _tensorize_multimodal
    from matcher.similarity import cosine_similarity

    def embed_one(mm):
        tt, jt, nt, et, pm, dnt, det = _tensorize_multimodal(
            mm, vocab, device=None, pcode_vocab_size=pcode_vocab_size,
        )
        with torch.no_grad():
            vec = model(tt, jt, nt, et, padding_mask=pm, dfg_node_features=dnt, dfg_edge_index=det)
        return vec.tolist()

    # pre-embed all
    t0 = time.time()
    q_vecs = {qid: embed_one(query_features[qid]) for qid in gt_qids}
    l_vecs = {lid: embed_one(library_features[lid]) for lid in lid_list}
    print(f"  嵌入耗时: {time.time()-t0:.2f}s  (queries={len(q_vecs)}, library={len(l_vecs)})")

    # pairwise cosine
    all_labels = []
    all_scores = []
    per_query = {}  # qid -> [(lid, score, is_pos)]

    for qid in gt_qids:
        qv = q_vecs[qid]
        pairs = []
        for lid in lid_list:
            sim = cosine_similarity(qv, l_vecs[lid])
            is_pos = 1 if lid in pos_set[qid] else 0
            all_labels.append(is_pos)
            all_scores.append(sim)
            pairs.append((lid, sim, is_pos))
        pairs.sort(key=lambda x: x[1], reverse=True)
        per_query[qid] = pairs

    y_true = np.array(all_labels)
    y_score = np.array(all_scores)
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())

    print(f"\n  对比数据: {len(all_scores)} 对  (正={n_pos}, 负={n_neg})")

    # ──────────────────────────────────────────────────────────────────
    # 4. 指标
    # ──────────────────────────────────────────────────────────────────
    sep("4. 核心指标")

    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        precision_recall_curve,
        roc_curve,
    )

    auc_roc = roc_auc_score(y_true, y_score)
    auc_pr = average_precision_score(y_true, y_score)
    print(f"  AUC-ROC:  {auc_roc:.4f}")
    print(f"  AUC-PR:   {auc_pr:.4f}")

    # 分数分布
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    print(f"\n  正对分数:  mean={pos_scores.mean():.4f}  std={pos_scores.std():.4f}  "
          f"min={pos_scores.min():.4f}  max={pos_scores.max():.4f}")
    print(f"  负对分数:  mean={neg_scores.mean():.4f}  std={neg_scores.std():.4f}  "
          f"min={neg_scores.min():.4f}  max={neg_scores.max():.4f}")

    # 分数分布直方图 (文本)
    lo, hi = y_score.min(), y_score.max()
    nbins = 12
    edges = np.linspace(lo, hi, nbins + 1)
    print(f"\n  分数分布直方图 ({nbins} bins, range [{lo:.3f}, {hi:.3f}]):")
    print(f"  {'bin range':>18s}  {'pos':>4s}  {'neg':>4s}  bar")
    for i in range(nbins):
        mask = (y_score >= edges[i]) & (y_score < edges[i + 1])
        if i == nbins - 1:
            mask = (y_score >= edges[i]) & (y_score <= edges[i + 1])
        p = int((y_true[mask] == 1).sum())
        n = int((y_true[mask] == 0).sum())
        bar_p = "+" * min(p, 40)
        bar_n = "-" * min(n, 40)
        print(f"  [{edges[i]:+.3f},{edges[i+1]:+.3f})  {p:4d}  {n:4d}  {bar_p}{bar_n}")

    # Recall@K (精排)
    for k in [1, 5, 10]:
        hits = 0
        for qid in gt_qids:
            topk_lids = {lid for lid, _, _ in per_query[qid][:k]}
            if topk_lids & pos_set[qid]:
                hits += 1
        print(f"\n  精排 Recall@{k}: {hits}/{len(gt_qids)} = {hits/len(gt_qids):.4f}")

    # MRR
    mrr_sum = 0.0
    for qid in gt_qids:
        for rank, (lid, _, is_pos) in enumerate(per_query[qid], 1):
            if is_pos:
                mrr_sum += 1.0 / rank
                break
    mrr = mrr_sum / len(gt_qids)
    print(f"  精排 MRR:       {mrr:.4f}")

    # Best threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_stat = tpr - fpr
    best_idx = int(np.argmax(j_stat))
    best_thr = thresholds[best_idx]
    print(f"\n  最优阈值 (Youden J): {best_thr:.4f}  (TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f})")

    # ──────────────────────────────────────────────────────────────────
    # 5. 逐 Query 排名
    # ──────────────────────────────────────────────────────────────────
    sep("5. 逐 Query 精排排名")

    print(f"  {'query addr':>12s}  {'rank':>4s}  {'score':>7s}  {'hit':>3s}  library addr")
    print(f"  {'-'*12}  {'-'*4}  {'-'*7}  {'-'*3}  {'-'*30}")
    for qid in gt_qids:
        addr_q = qid.rsplit("|", 1)[-1] if "|" in qid else qid[-6:]
        for rank, (lid, score, is_pos) in enumerate(per_query[qid], 1):
            addr_l = lid.rsplit("|", 1)[-1] if "|" in lid else lid[-6:]
            mark = "  +" if is_pos else ""
            if is_pos or rank <= 3:
                print(f"  {addr_q:>12s}  #{rank:<3d}  {score:+.4f} {mark:3s}  {addr_l}")
            elif rank == 4 and any(x[2] for x in per_query[qid][3:]):
                # show ellipsis only if more positives below
                pass
        print()

    # ──────────────────────────────────────────────────────────────────
    # 6. 粗筛 vs 精排 对比
    # ──────────────────────────────────────────────────────────────────
    sep("6. 粗筛 vs 精排 排名变化")

    improved = 0
    degraded = 0
    same = 0
    for qid in gt_qids:
        for lid in pos_set[qid]:
            cr = coarse_ranks[qid].get(lid, 999)
            # find rerank position
            rr = 999
            for rank, (rid, _, _) in enumerate(per_query[qid], 1):
                if rid == lid:
                    rr = rank
                    break
            if rr < cr:
                improved += 1
            elif rr > cr:
                degraded += 1
            else:
                same += 1

    print(f"  正对总数: {total_pos}")
    print(f"  精排改善 (rank↓): {improved}  ({improved/total_pos:.1%})")
    print(f"  精排变差 (rank↑): {degraded}  ({degraded/total_pos:.1%})")
    print(f"  不变:             {same}  ({same/total_pos:.1%})")

    # ──────────────────────────────────────────────────────────────────
    # 汇总
    # ──────────────────────────────────────────────────────────────────
    sep("SUMMARY")
    print(f"""
  Model:  {total_params:,} params, dfg={use_dfg}, epochs={epochs}
  Data:   {len(gt_qids)} queries × {len(lid_list)} library = {total_pos+total_neg} pairs (pos={total_pos}, neg={total_neg})

  AUC-ROC:  {auc_roc:.4f}
  AUC-PR:   {auc_pr:.4f}
  MRR:      {mrr:.4f}

  Pos score: {pos_scores.mean():.4f} ± {pos_scores.std():.4f}
  Neg score: {neg_scores.mean():.4f} ± {neg_scores.std():.4f}
  Separation: {(pos_scores.mean() - neg_scores.mean()):.4f}
    """)


if __name__ == "__main__":
    main()
