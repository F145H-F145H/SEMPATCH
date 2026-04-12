"""
Multimodal 嵌入空间上的简化检索验证：Recall@1（查询 Top-1 是否在 ground_truth 正例集合中）。
供 train_multimodal 可选 epoch 末评估；与 train_safe 的两阶段管线分离。

优化版：库嵌入和 query 嵌入均使用 batched tensorize（tensorize_multimodal_many），
避免逐样本 _tensorize_multimodal 调用导致的 GPU 利用率低下。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


def _embed_batch_of_mm(
    model: torch.nn.Module,
    vocab: Dict[str, int],
    device: torch.device,
    mm_list: List[Dict[str, Any]],
    *,
    max_seq_len: int,
    max_graph_nodes: int,
    max_dfg_nodes: int,
    pcode_vocab_size: int = 256,
) -> Optional[torch.Tensor]:
    """批量嵌入多个 multimodal 特征，返回 (B, D) tensor 或 None（全部失败时）。"""
    from features.models.multimodal_fusion import tensorize_multimodal_many

    # 过滤空特征
    valid_mm = []
    valid_idx = []
    for i, mm in enumerate(mm_list):
        if mm and (mm.get("sequence", {}).get("pcode_tokens") or mm.get("graph", {}).get("node_features")):
            valid_mm.append(mm)
            valid_idx.append(i)

    if not valid_mm:
        return None

    try:
        batch = tensorize_multimodal_many(
            valid_mm, vocab, device=device,
            max_seq_len=max_seq_len,
            max_graph_nodes=max_graph_nodes,
            max_dfg_nodes=max_dfg_nodes,
            pcode_vocab_size=pcode_vocab_size,
        )
        with torch.no_grad():
            v = model(*batch)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        return v, valid_idx
    except Exception:
        return None


def _embed_one(
    model: torch.nn.Module,
    vocab: Dict[str, int],
    device: torch.device,
    mm: Dict[str, Any],
    *,
    max_seq_len: int,
    max_graph_nodes: int,
    max_dfg_nodes: int,
) -> Optional[torch.Tensor]:
    """单条嵌入 fallback。"""
    from features.models.multimodal_fusion import _tensorize_multimodal

    try:
        t, j, n, e, p, dn, de = _tensorize_multimodal(
            mm, vocab, device=device,
            max_seq_len=max_seq_len,
            max_graph_nodes=max_graph_nodes,
            max_dfg_nodes=max_dfg_nodes,
        )
        v = model(t, j, n, e, p, dfg_node_features=dn, dfg_edge_index=de)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        return v
    except Exception:
        return None


def multimodal_retrieval_recall_at_1(
    model: torch.nn.Module,
    vocab: Dict[str, int],
    device: torch.device,
    library_multimodal: Dict[str, Dict[str, Any]],
    query_multimodal: Dict[str, Dict[str, Any]],
    ground_truth: Dict[str, List[str]],
    *,
    max_seq_len: int = 512,
    max_graph_nodes: int = 128,
    max_dfg_nodes: int = 128,
    embed_batch_size: int = 16,
) -> Tuple[float, int, int]:
    """
    将全部库嵌入矩阵化（CPU 保存），再对 ground_truth 中出现的 query 批量嵌入，余弦相似度 Argmax。

    优化：使用 batched tensorize（tensorize_multimodal_many）代替逐样本 _tensorize_multimodal，
    在 embed_batch_size 范围内将多次 GPU kernel launch 合并为一次，显著提升 GPU 利用率。

    Returns:
        (recall_at_1, num_evaluated, num_correct)
    """
    model.eval()
    lib_ids = list(library_multimodal.keys())
    if not lib_ids:
        return 0.0, 0, 0

    # ── 库嵌入：batched tensorize，每 embed_batch_size 个函数一次 GPU forward ──
    vocab_size = max(len(vocab), 256)
    lib_rows_cpu: List[torch.Tensor] = []
    lib_ids_kept: List[str] = []
    with torch.no_grad():
        for i in range(0, len(lib_ids), embed_batch_size):
            chunk_ids = lib_ids[i : i + embed_batch_size]
            chunk_mm = [library_multimodal.get(fid) or {} for fid in chunk_ids]
            result = _embed_batch_of_mm(
                model, vocab, device, chunk_mm,
                max_seq_len=max_seq_len,
                max_graph_nodes=max_graph_nodes,
                max_dfg_nodes=max_dfg_nodes,
                pcode_vocab_size=vocab_size,
            )
            if result is not None:
                vecs, valid_idx = result
                for j, vi in enumerate(valid_idx):
                    lib_rows_cpu.append(vecs[j:j+1].cpu())
                    lib_ids_kept.append(chunk_ids[vi])

    if not lib_rows_cpu:
        return 0.0, 0, 0
    lib_mat_cpu = torch.cat(lib_rows_cpu, dim=0)
    lib_mat_cpu = torch.nn.functional.normalize(lib_mat_cpu, dim=1)
    lib_ids = lib_ids_kept

    eval_qids = [q for q in ground_truth if q in query_multimodal and ground_truth[q]]
    if not eval_qids:
        return 0.0, 0, 0

    # ── Query 嵌入：同样 batched ──
    correct = 0
    total = 0
    n_lib = lib_mat_cpu.size(0)
    with torch.no_grad():
        for qi in range(0, len(eval_qids), embed_batch_size):
            qid_chunk = eval_qids[qi : qi + embed_batch_size]
            qmm_chunk = [query_multimodal[qid] for qid in qid_chunk]
            result = _embed_batch_of_mm(
                model, vocab, device, qmm_chunk,
                max_seq_len=max_seq_len,
                max_graph_nodes=max_graph_nodes,
                max_dfg_nodes=max_dfg_nodes,
                pcode_vocab_size=vocab_size,
            )
            if result is None:
                continue
            q_vecs, q_valid_idx = result
            q_vecs = torch.nn.functional.normalize(q_vecs.to(device), dim=1)

            # Chunked similarity against library
            for j, vi in enumerate(q_valid_idx):
                qv = q_vecs[j:j+1]
                best_score = float("-inf")
                best_idx = 0
                for ci in range(0, n_lib, embed_batch_size):
                    chunk = lib_mat_cpu[ci : ci + embed_batch_size].to(device)
                    sims = qv @ chunk.T  # (1, chunk_size)
                    chunk_best_i = int(sims.argmax(dim=1).item())
                    chunk_score = sims[0, chunk_best_i].item()
                    if chunk_score > best_score:
                        best_score = chunk_score
                        best_idx = ci + chunk_best_i
                qid = qid_chunk[vi]
                total += 1
                if best_idx < len(lib_ids):
                    best_fid = lib_ids[best_idx]
                    positives = ground_truth.get(qid) or []
                    if best_fid in positives:
                        correct += 1

    recall = correct / total if total else 0.0
    return recall, total, correct
