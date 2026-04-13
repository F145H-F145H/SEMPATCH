"""
预计算 Tensor 格式的 collate_fn 和 step_fn。

配合 PrecomputedTensorDataset 使用，实现训练时零 Python dict 操作：
  - collate: 纯 numpy stack + edge 重建
  - step_fn: .to(device) + model forward
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _rebuild_edge_index(
    src_arrs: List[np.ndarray],
    dst_arrs: List[np.ndarray],
    n_edges_list: List[int],
    max_nodes: int,
) -> "torch.Tensor":
    """从 batch 内各 item 的 edge src/dst 重建带 offset 的 edge_index (2, total_edges)。"""
    edge_parts = []
    for batch_i, (src, dst, n_e) in enumerate(zip(src_arrs, dst_arrs, n_edges_list)):
        if n_e <= 0:
            continue
        offset = batch_i * max_nodes
        s = src[:n_e] + offset
        d = dst[:n_e] + offset
        edge_parts.append(np.stack([s, d], axis=0))

    if not edge_parts:
        return torch.zeros(2, 0, dtype=torch.long)
    combined = np.concatenate(edge_parts, axis=1)
    return torch.from_numpy(combined.astype(np.int64))


def collate_multimodal_precomputed(batch: List[Tuple]) -> Dict[str, Any]:
    """
    MultiModal 训练 collate_fn：从 PrecomputedTensorDataset 的 tuple 输出构建 batched tensors。

    输入 tuple 结构（25 元素）:
      a_side(12): token_ids, jump_mask, node_ids, edge_src, edge_dst, graph_n_edges,
                  dfg_node_ids, dfg_edge_src, dfg_edge_dst, dfg_n_edges, seq_len, node_count
      b_side(12): 同上
      label(1): float

    输出 dict: {"valid": True, "batch1": 7-tuple, "batch2": 7-tuple, "labels": tensor}
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")

    B = len(batch)
    if B == 0:
        return {"valid": False, "labels": torch.tensor([])}

    # 提取各 side 的数组和标量
    def _extract_side(side_offset: int):
        tokens = [item[side_offset + 0] for item in batch]
        jumps = [item[side_offset + 1] for item in batch]
        nodes = [item[side_offset + 2] for item in batch]
        edge_srcs = [item[side_offset + 3] for item in batch]
        edge_dsts = [item[side_offset + 4] for item in batch]
        n_edges = [item[side_offset + 5] for item in batch]
        dfg_nodes = [item[side_offset + 6] for item in batch]
        dfg_srcs = [item[side_offset + 7] for item in batch]
        dfg_dsts = [item[side_offset + 8] for item in batch]
        dfg_n_edges = [item[side_offset + 9] for item in batch]
        seq_lens = [item[side_offset + 10] for item in batch]
        node_counts = [item[side_offset + 11] for item in batch]
        return tokens, jumps, nodes, edge_srcs, edge_dsts, n_edges, \
               dfg_nodes, dfg_srcs, dfg_dsts, dfg_n_edges, seq_lens, node_counts

    a_tok, a_jmp, a_nod, a_es, a_ed, a_ne, \
        a_dn, a_ds, a_dd, a_dne, a_sl, a_nc = _extract_side(0)
    b_tok, b_jmp, b_nod, b_es, b_ed, b_ne, \
        b_dn, b_ds, b_dd, b_dne, b_sl, b_nc = _extract_side(12)

    labels = torch.tensor([item[24] for item in batch], dtype=torch.float32)

    def _build_side(tokens, jumps, nodes, edge_srcs, edge_dsts, n_edges,
                    dfg_nodes, dfg_srcs, dfg_dsts, dfg_n_edges, seq_lens, node_counts):
        max_seq = len(tokens[0])
        max_nod = len(nodes[0])
        max_dfg = len(dfg_nodes[0])

        token_t = torch.from_numpy(np.stack(tokens).astype(np.int64))
        jump_t = torch.from_numpy(np.stack(jumps).astype(np.int64))

        # padding_mask: True 表示 pad
        pad_mask = torch.zeros(B, max_seq, dtype=torch.bool)
        for i, sl in enumerate(seq_lens):
            if sl < max_seq:
                pad_mask[i, sl:] = True

        node_t = torch.from_numpy(np.stack(nodes).astype(np.int64))

        # rebuild graph edge_index with batch offsets
        edge_t = _rebuild_edge_index(edge_srcs, edge_dsts, n_edges, max_nod)

        dfg_node_t = torch.from_numpy(np.stack(dfg_nodes).astype(np.int64))

        # rebuild dfg edge_index with batch offsets
        dfg_edge_t = _rebuild_edge_index(dfg_srcs, dfg_dsts, dfg_n_edges, max_dfg)

        return token_t, jump_t, node_t, edge_t, pad_mask, dfg_node_t, dfg_edge_t

    batch1 = _build_side(a_tok, a_jmp, a_nod, a_es, a_ed, a_ne,
                         a_dn, a_ds, a_dd, a_dne, a_sl, a_nc)
    batch2 = _build_side(b_tok, b_jmp, b_nod, b_es, b_ed, b_ne,
                         b_dn, b_ds, b_dd, b_dne, b_sl, b_nc)

    return {"valid": True, "batch1": batch1, "batch2": batch2, "labels": labels}


def collate_safe_precomputed(batch: List[Tuple]) -> Dict[str, Any]:
    """
    SAFE 训练 collate_fn：从 PrecomputedTensorDataset.get_safe_token_arrays() 构建 batch。

    输入 tuple: (token_ids, jump_mask, a_token_ids, a_jump_mask, label)
    输出 dict: {"valid": True, "t1", "p1", "t2", "p2", "labels"}
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")

    B = len(batch)
    if B == 0:
        return {"valid": False, "labels": torch.tensor([])}

    # batch 中每个元素由 _make_safe_pair_collate 返回
    a_tokens = [item[0] for item in batch]
    a_jumps = [item[1] for item in batch]
    b_tokens = [item[2] for item in batch]
    b_jumps = [item[3] for item in batch]
    labels = torch.tensor([item[4] for item in batch], dtype=torch.float32)

    max_len = len(a_tokens[0])

    def _make_pad_mask(jumps):
        # jump mask 中 0 表示有效 token（或 padding），通过 seq_len 推断
        # 简化：假设非零 jump 区域是有效的，最后一个非零位置之后为 pad
        pm = torch.zeros(B, max_len, dtype=torch.bool)
        return pm

    t1 = torch.from_numpy(np.stack(a_tokens).astype(np.int64))
    t2 = torch.from_numpy(np.stack(b_tokens).astype(np.int64))
    p1 = torch.from_numpy(np.stack(a_jumps).astype(np.int64))
    p2 = torch.from_numpy(np.stack(b_jumps).astype(np.int64))

    # 构建 pad_mask: token_id == 0 的位置为 pad
    pad1 = (t1 == 0)
    pad2 = (t2 == 0)

    return {
        "valid": True,
        "t1": t1, "p1": pad1,
        "t2": t2, "p2": pad2,
        "labels": labels,
    }


def make_safe_precomputed_pairs(dataset, num_pairs, positive_ratio, seed):
    """
    为 SAFE 训练预生成 pair 列表，返回 collate_fn 可用的 tuple 数据集。
    用法：在 train_safe.py 中调用，生成固定 pair 列表供 DataLoader 使用。
    """
    import random as _random
    rng = _random.Random(seed)

    pairs = []
    for _ in range(num_pairs):
        is_pos = rng.random() < positive_ratio
        if is_pos and dataset._positive_candidates:
            for _ in range(20):
                _name, idxs = rng.choice(dataset._positive_candidates)
                if len(idxs) < 2:
                    continue
                a, b = rng.sample(idxs, 2)
                break
            else:
                a, b = rng.sample(dataset._all_idx, 2)
            label = 1.0
        else:
            a, b = rng.sample(dataset._all_idx, 2)
            label = 0.0

        at, aj, _asl = dataset.get_safe_token_arrays(a)
        bt, bj, _bsl = dataset.get_safe_token_arrays(b)
        pairs.append((at, aj, bt, bj, label))

    return pairs
