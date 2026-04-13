"""
预计算 multimodal 特征为固定形状 NumPy 数组，供训练时零 dict 操作加载。

每个函数的 multimodal 特征预转为：
  token_ids:    (max_seq_len,)      int16
  jump_mask:    (max_seq_len,)      int8
  node_ids:     (max_graph_nodes,)  int16
  edge_src:     (max_edges,)        int32   graph edge sources (0-padded)
  edge_dst:     (max_edges,)        int32   graph edge destinations (0-padded)
  graph_n_edges: int16              实际 graph edge 数
  dfg_node_ids: (max_dfg_nodes,)    int16
  dfg_edge_src: (max_dfg_edges,)    int32
  dfg_edge_dst: (max_dfg_edges,)    int32
  dfg_n_edges:  int16               实际 DFG edge 数
  seq_len:      int16
  node_count:   int16
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_log = logging.getLogger(__name__)

# 默认最大边数（单函数 graph/dfg 边数极少超过此值）
_DEFAULT_MAX_EDGES = 512
_DEFAULT_MAX_DFG_EDGES = 256


def multimodal_to_arrays(
    mm: Dict[str, Any],
    vocab: Dict[str, int],
    *,
    max_seq_len: int = 512,
    max_graph_nodes: int = 128,
    max_dfg_nodes: int = 64,
    pcode_vocab_size: int = 256,
    max_edges: int = _DEFAULT_MAX_EDGES,
    max_dfg_edges: int = _DEFAULT_MAX_DFG_EDGES,
) -> Dict[str, np.ndarray]:
    """将单个 multimodal dict 转为固定形状 numpy 数组。"""
    unk = 1
    seq = mm.get("sequence") or {}
    graph = mm.get("graph") or {}
    dfg = mm.get("dfg") or {}

    # ── sequence tokens ──
    precomp_ids = seq.get("pcode_token_ids")
    if precomp_ids and isinstance(precomp_ids, list):
        t_raw = list(precomp_ids[:max_seq_len])
    else:
        tokens = seq.get("pcode_tokens") or []
        t_raw = [vocab.get(t, unk) for t in tokens[:max_seq_len]]

    token_ids = np.zeros(max_seq_len, dtype=np.int16)
    for i, v in enumerate(t_raw):
        token_ids[i] = max(0, min(int(v), pcode_vocab_size - 1))

    jump_mask = np.zeros(max_seq_len, dtype=np.int8)
    jm = seq.get("jump_mask") or []
    for i, v in enumerate(jm[:max_seq_len]):
        jump_mask[i] = int(v)

    actual_seq_len = min(len(t_raw), max_seq_len)

    # ── graph nodes ──
    node_feats = graph.get("node_features") or []
    node_ids = np.zeros(max_graph_nodes, dtype=np.int16)
    actual_node_count = min(len(node_feats), max_graph_nodes)
    for i, nf in enumerate(node_feats[:max_graph_nodes]):
        if isinstance(nf, dict):
            po = nf.get("opcode_id")
            if po is not None:
                node_ids[i] = max(0, min(int(po), pcode_vocab_size - 1))
                continue
            opcodes = nf.get("pcode_opcodes") or []
        else:
            opcodes = nf if isinstance(nf, list) else []
        idx = vocab.get(opcodes[0], unk) if opcodes else 0
        node_ids[i] = max(0, min(idx, pcode_vocab_size - 1))

    # ── graph edges ──
    ei = graph.get("edge_index") or [[], []]
    src_list, dst_list = [], []
    if ei and len(ei) >= 2:
        for s, d in zip(ei[0], ei[1]):
            if 0 <= s < actual_node_count and 0 <= d < actual_node_count:
                src_list.append(int(s))
                dst_list.append(int(d))
    graph_n_edges = min(len(src_list), max_edges)
    edge_src = np.zeros(max_edges, dtype=np.int32)
    edge_dst = np.zeros(max_edges, dtype=np.int32)
    if graph_n_edges > 0:
        edge_src[:graph_n_edges] = src_list[:graph_n_edges]
        edge_dst[:graph_n_edges] = dst_list[:graph_n_edges]

    # ── dfg nodes ──
    dfg_nf = dfg.get("node_features") or []
    dfg_node_ids = np.zeros(max_dfg_nodes, dtype=np.int16)
    for i, x in enumerate(dfg_nf[:max_dfg_nodes]):
        if isinstance(x, int):
            dfg_node_ids[i] = int(x) % 512

    # ── dfg edges ──
    dei = dfg.get("edge_index") or [[], []]
    dfg_src_list, dfg_dst_list = [], []
    actual_dfg_nodes = min(len(dfg_nf), max_dfg_nodes)
    if dei and len(dei) >= 2:
        for s, d in zip(dei[0], dei[1]):
            if 0 <= s < actual_dfg_nodes and 0 <= d < actual_dfg_nodes:
                dfg_src_list.append(int(s))
                dfg_dst_list.append(int(d))
    dfg_n_edges = min(len(dfg_src_list), max_dfg_edges)
    dfg_edge_src = np.zeros(max_dfg_edges, dtype=np.int32)
    dfg_edge_dst = np.zeros(max_dfg_edges, dtype=np.int32)
    if dfg_n_edges > 0:
        dfg_edge_src[:dfg_n_edges] = dfg_src_list[:dfg_n_edges]
        dfg_edge_dst[:dfg_n_edges] = dfg_dst_list[:dfg_n_edges]

    return {
        "token_ids": token_ids,
        "jump_mask": jump_mask,
        "node_ids": node_ids,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "graph_n_edges": np.int16(graph_n_edges),
        "dfg_node_ids": dfg_node_ids,
        "dfg_edge_src": dfg_edge_src,
        "dfg_edge_dst": dfg_edge_dst,
        "dfg_n_edges": np.int16(dfg_n_edges),
        "seq_len": np.int16(actual_seq_len),
        "node_count": np.int16(actual_node_count),
    }


def build_precomputed_npz(
    multimodals_with_ids: List[Tuple[str, Dict[str, Any]]],
    vocab: Dict[str, int],
    output_path: str,
    *,
    max_seq_len: int = 512,
    max_graph_nodes: int = 128,
    max_dfg_nodes: int = 64,
    pcode_vocab_size: int = 256,
    max_edges: int = _DEFAULT_MAX_EDGES,
    max_dfg_edges: int = _DEFAULT_MAX_DFG_EDGES,
) -> Dict[str, int]:
    """
    批量转换 multimodal 列表为 npz 文件。

    Args:
        multimodals_with_ids: [(function_id, multimodal_dict), ...]
        vocab: pcode vocab
        output_path: 输出 .npz 路径
        max_*: 各维度上限

    Returns:
        {"function_id": array_index, ...} 映射字典
    """
    N = len(multimodals_with_ids)
    if N == 0:
        raise ValueError("multimodals_with_ids is empty")

    _log.info("build_precomputed_npz: 转换 %d 个函数 → %s", N, output_path)

    # 预分配
    token_ids = np.zeros((N, max_seq_len), dtype=np.int16)
    jump_mask = np.zeros((N, max_seq_len), dtype=np.int8)
    node_ids_arr = np.zeros((N, max_graph_nodes), dtype=np.int16)
    edge_src = np.zeros((N, max_edges), dtype=np.int32)
    edge_dst = np.zeros((N, max_edges), dtype=np.int32)
    graph_n_edges = np.zeros(N, dtype=np.int16)
    dfg_node_ids = np.zeros((N, max_dfg_nodes), dtype=np.int16)
    dfg_edge_src = np.zeros((N, max_dfg_edges), dtype=np.int32)
    dfg_edge_dst = np.zeros((N, max_dfg_edges), dtype=np.int32)
    dfg_n_edges_arr = np.zeros(N, dtype=np.int16)
    seq_lens = np.zeros(N, dtype=np.int16)
    node_counts = np.zeros(N, dtype=np.int16)

    fid_to_idx: Dict[str, int] = {}

    t0 = time.perf_counter()
    LOG_EVERY = max(1, N // 20)

    for i, (fid, mm) in enumerate(multimodals_with_ids):
        fid_to_idx[fid] = i
        arrays = multimodal_to_arrays(
            mm, vocab,
            max_seq_len=max_seq_len,
            max_graph_nodes=max_graph_nodes,
            max_dfg_nodes=max_dfg_nodes,
            pcode_vocab_size=pcode_vocab_size,
            max_edges=max_edges,
            max_dfg_edges=max_dfg_edges,
        )
        token_ids[i] = arrays["token_ids"]
        jump_mask[i] = arrays["jump_mask"]
        node_ids_arr[i] = arrays["node_ids"]
        edge_src[i] = arrays["edge_src"]
        edge_dst[i] = arrays["edge_dst"]
        graph_n_edges[i] = arrays["graph_n_edges"]
        dfg_node_ids[i] = arrays["dfg_node_ids"]
        dfg_edge_src[i] = arrays["dfg_edge_src"]
        dfg_edge_dst[i] = arrays["dfg_edge_dst"]
        dfg_n_edges_arr[i] = arrays["dfg_n_edges"]
        seq_lens[i] = arrays["seq_len"]
        node_counts[i] = arrays["node_count"]

        if (i + 1) % LOG_EVERY == 0 or i == N - 1:
            elapsed = time.perf_counter() - t0
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            _log.info("  转换进度: %d/%d (%.0f 条/s, %.1fs)", i + 1, N, speed, elapsed)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        token_ids=token_ids,
        jump_mask=jump_mask,
        node_ids=node_ids_arr,
        edge_src=edge_src,
        edge_dst=edge_dst,
        graph_n_edges=graph_n_edges,
        dfg_node_ids=dfg_node_ids,
        dfg_edge_src=dfg_edge_src,
        dfg_edge_dst=dfg_edge_dst,
        dfg_n_edges=dfg_n_edges_arr,
        seq_lens=seq_lens,
        node_counts=node_counts,
    )
    elapsed = time.perf_counter() - t0
    fsize_mb = os.path.getsize(output_path) / (1024 * 1024)
    _log.info(
        "build_precomputed_npz: 完成，%d 函数，%.1fs，文件大小 %.1fMB",
        N, elapsed, fsize_mb,
    )
    return fid_to_idx


# ---------------------------------------------------------------------------
# 流式 memmap 版本（两遍扫描，峰值内存极低）
# ---------------------------------------------------------------------------

# memmap 子文件名后缀
_MEMMAP_SUFFIXES = {
    "token_ids":    (".token_ids.npy",    np.int16),
    "jump_mask":    (".jump_mask.npy",    np.int8),
    "node_ids":     (".node_ids.npy",     np.int16),
    "edge_src":     (".edge_src.npy",     np.int32),
    "edge_dst":     (".edge_dst.npy",     np.int32),
    "graph_n_edges":(".graph_n_edges.npy",np.int16),
    "dfg_node_ids": (".dfg_node_ids.npy", np.int16),
    "dfg_edge_src": (".dfg_edge_src.npy", np.int32),
    "dfg_edge_dst": (".dfg_edge_dst.npy", np.int32),
    "dfg_n_edges":  (".dfg_n_edges.npy", np.int16),
    "seq_lens":     (".seq_lens.npy",    np.int16),
    "node_counts":  (".node_counts.npy", np.int16),
}


def _memmap_paths(output_path: str) -> Dict[str, str]:
    """返回 {key: memmap_file_path}。"""
    base = os.path.splitext(output_path)[0]
    return {k: base + suf for k, (suf, _) in _MEMMAP_SUFFIXES.items()}


def create_memmap_arrays(
    N: int,
    output_path: str,
    *,
    max_seq_len: int = 512,
    max_graph_nodes: int = 128,
    max_dfg_nodes: int = 64,
    max_edges: int = _DEFAULT_MAX_EDGES,
    max_dfg_edges: int = _DEFAULT_MAX_DFG_EDGES,
) -> Dict[str, np.ndarray]:
    """
    在磁盘上创建 memmap 数组并返回 dict。
    caller 逐行填充后调用 flush_memmap_arrays()。
    """
    shapes = {
        "token_ids":     (N, max_seq_len),
        "jump_mask":     (N, max_seq_len),
        "node_ids":      (N, max_graph_nodes),
        "edge_src":      (N, max_edges),
        "edge_dst":      (N, max_edges),
        "graph_n_edges": (N,),
        "dfg_node_ids":  (N, max_dfg_nodes),
        "dfg_edge_src":  (N, max_dfg_edges),
        "dfg_edge_dst":  (N, max_dfg_edges),
        "dfg_n_edges":   (N,),
        "seq_lens":      (N,),
        "node_counts":   (N,),
    }
    paths = _memmap_paths(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    arrays: Dict[str, np.ndarray] = {}
    for key, (suffix, dtype) in _MEMMAP_SUFFIXES.items():
        arrays[key] = np.memmap(paths[key], dtype=dtype, mode="w+", shape=shapes[key])
    return arrays


def flush_memmap_arrays(
    arrays: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    """Flush 所有 memmap，打包为 .npz，删除临时 .npy。"""
    for arr in arrays.values():
        arr.flush()
    # 将所有 memmap 打包为单一 npz
    np.savez_compressed(output_path, **arrays)
    # 释放 memmap 引用
    arrays.clear()
    # 删除临时 .npy 文件
    for path in _memmap_paths(output_path).values():
        try:
            os.remove(path)
        except OSError:
            pass


def build_precomputed_npz_streaming(
    jsonl_path: str,
    needed_set: set,
    vocab: Dict[str, int],
    output_path: str,
    *,
    max_seq_len: int = 512,
    max_graph_nodes: int = 128,
    max_dfg_nodes: int = 64,
    pcode_vocab_size: int = 256,
    max_edges: int = _DEFAULT_MAX_EDGES,
    max_dfg_edges: int = _DEFAULT_MAX_DFG_EDGES,
    log_every: int = 10000,
) -> Dict[str, int]:
    """
    两遍扫描 + memmap 流式构建，峰值内存仅保留当前一条样本。

    第一遍：统计 N（样本总数），收集 vocab（调用方已完成时可跳过）。
    第二遍：逐条填充 memmap 数组。

    Returns:
        {"function_id": array_index, ...} 映射字典
    """
    from utils.precomputed_multimodal_io import iter_jsonl_sidecar

    # ── 第一遍：统计 N ──
    _log.info("build_precomputed_npz_streaming: 第一遍扫描 — 统计样本数…")
    N = 0
    for fid, _ in iter_jsonl_sidecar(jsonl_path):
        if fid in needed_set:
            N += 1
    _log.info("  样本总数: %d", N)
    if N == 0:
        raise ValueError("没有匹配到任何样本")

    # ── 创建 memmap ──
    _log.info("  创建 memmap 文件…")
    arrays = create_memmap_arrays(
        N, output_path,
        max_seq_len=max_seq_len,
        max_graph_nodes=max_graph_nodes,
        max_dfg_nodes=max_dfg_nodes,
        max_edges=max_edges,
        max_dfg_edges=max_dfg_edges,
    )

    # ── 第二遍：逐条填充 ──
    _log.info("build_precomputed_npz_streaming: 第二遍扫描 — 逐条填充…")
    fid_to_idx: Dict[str, int] = {}
    idx = 0
    t0 = time.perf_counter()

    for fid, mm in iter_jsonl_sidecar(jsonl_path):
        if fid not in needed_set:
            continue

        sample = multimodal_to_arrays(
            mm, vocab,
            max_seq_len=max_seq_len,
            max_graph_nodes=max_graph_nodes,
            max_dfg_nodes=max_dfg_nodes,
            pcode_vocab_size=pcode_vocab_size,
            max_edges=max_edges,
            max_dfg_edges=max_dfg_edges,
        )

        arrays["token_ids"][idx]     = sample["token_ids"]
        arrays["jump_mask"][idx]     = sample["jump_mask"]
        arrays["node_ids"][idx]      = sample["node_ids"]
        arrays["edge_src"][idx]      = sample["edge_src"]
        arrays["edge_dst"][idx]      = sample["edge_dst"]
        arrays["graph_n_edges"][idx] = sample["graph_n_edges"]
        arrays["dfg_node_ids"][idx]  = sample["dfg_node_ids"]
        arrays["dfg_edge_src"][idx]  = sample["dfg_edge_src"]
        arrays["dfg_edge_dst"][idx]  = sample["dfg_edge_dst"]
        arrays["dfg_n_edges"][idx]   = sample["dfg_n_edges"]
        arrays["seq_lens"][idx]      = sample["seq_len"]
        arrays["node_counts"][idx]   = sample["node_count"]

        fid_to_idx[fid] = idx
        idx += 1

        if idx % log_every == 0:
            elapsed = time.perf_counter() - t0
            speed = idx / elapsed if elapsed > 0 else 0
            _log.info("  已处理 %d/%d (%.0f 条/s)", idx, N, speed)

    # ── 保存 ──
    _log.info("  Flush memmap → npz …")
    flush_memmap_arrays(arrays, output_path)
    elapsed = time.perf_counter() - t0
    fsize_mb = os.path.getsize(output_path) / (1024 * 1024)
    _log.info(
        "build_precomputed_npz_streaming: 完成，%d 函数，%.1fs，%.1fMB",
        idx, elapsed, fsize_mb,
    )
    return fid_to_idx


def build_edge_index_batch(
    src_arr: np.ndarray,
    dst_arr: np.ndarray,
    n_edges_arr: np.ndarray,
    node_counts: np.ndarray,
    indices: np.ndarray,
    max_nodes: int,
) -> Tuple[np.ndarray, int]:
    """
    从预计算的 edge src/dst 数组重建带 batch offset 的 edge_index。

    Args:
        src_arr: (N, max_edges) source node indices
        dst_arr: (N, max_edges) dest node indices
        n_edges_arr: (N,) actual edge count per item
        node_counts: (N,) actual node count per item
        indices: (B,) batch item indices
        max_nodes: per-item max nodes (for offset calculation)

    Returns:
        (edge_index, total_edges) — edge_index shape (2, total_edges)
    """
    edge_lists = []
    for batch_i, idx in enumerate(indices):
        n_e = int(n_edges_arr[idx])
        if n_e <= 0:
            continue
        offset = batch_i * max_nodes
        s = src_arr[idx, :n_e] + offset
        d = dst_arr[idx, :n_e] + offset
        edge_lists.append(np.stack([s, d], axis=0))

    if not edge_lists:
        return np.zeros((2, 0), dtype=np.int64), 0

    combined = np.concatenate(edge_lists, axis=1)
    return combined, combined.shape[1]