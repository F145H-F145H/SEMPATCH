"""
Multimodal 嵌入空间上的简化检索验证：Recall@1（查询 Top-1 是否在 ground_truth 正例集合中）。
供 train_multimodal 可选 epoch 末评估；与 train_safe 的两阶段管线分离。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


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
    from features.models.multimodal_fusion import _tensorize_multimodal

    try:
        t, j, n, e, p, dn, de = _tensorize_multimodal(
            mm,
            vocab,
            device=device,
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

    Returns:
        (recall_at_1, num_evaluated, num_correct)
    """
    model.eval()
    lib_ids = list(library_multimodal.keys())
    if not lib_ids:
        return 0.0, 0, 0

    # Embed library on GPU but move to CPU immediately to save VRAM
    lib_rows_cpu: List[torch.Tensor] = []
    lib_ids_kept: List[str] = []
    with torch.no_grad():
        for i in range(0, len(lib_ids), embed_batch_size):
            chunk_ids = lib_ids[i : i + embed_batch_size]
            for fid in chunk_ids:
                mm = library_multimodal.get(fid) or {}
                v = _embed_one(
                    model,
                    vocab,
                    device,
                    mm,
                    max_seq_len=max_seq_len,
                    max_graph_nodes=max_graph_nodes,
                    max_dfg_nodes=max_dfg_nodes,
                )
                if v is not None:
                    lib_rows_cpu.append(v.cpu())
                    lib_ids_kept.append(fid)
    if not lib_rows_cpu:
        return 0.0, 0, 0
    lib_mat_cpu = torch.cat(lib_rows_cpu, dim=0)
    lib_mat_cpu = torch.nn.functional.normalize(lib_mat_cpu, dim=1)
    lib_ids = lib_ids_kept

    eval_qids = [q for q in ground_truth if q in query_multimodal and ground_truth[q]]
    if not eval_qids:
        return 0.0, 0, 0

    correct = 0
    total = 0
    n_lib = lib_mat_cpu.size(0)
    with torch.no_grad():
        for qid in eval_qids:
            qv = _embed_one(
                model,
                vocab,
                device,
                query_multimodal[qid],
                max_seq_len=max_seq_len,
                max_graph_nodes=max_graph_nodes,
                max_dfg_nodes=max_dfg_nodes,
            )
            if qv is None:
                continue
            qv = torch.nn.functional.normalize(qv.to(device), dim=1)
            # Chunked similarity to avoid loading full lib_mat onto GPU
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
            total += 1
            if best_idx >= len(lib_ids):
                continue
            best_fid = lib_ids[best_idx]
            positives = ground_truth.get(qid) or []
            if best_fid in positives:
                correct += 1

    recall = correct / total if total else 0.0
    return recall, total, correct
