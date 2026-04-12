"""
MultiModalFusionModel（survey 5.1）：图分支 + 序列分支 + 跨模态注意力。
可选 DFG 独立图分支（阶段 H）：CFG 与 DFG 图嵌入拼接后融合，再与序列跨模态注意力。
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _build_vocab(pcode_tokens: List[str]) -> Dict[str, int]:
    """从 token 列表构建 vocab，=0, [UNK]=1。"""
    vocab: Dict[str, int] = {"": 0, "[UNK]": 1}
    for t in pcode_tokens:
        if t and t not in vocab:
            vocab[t] = len(vocab)
    return vocab


def infer_use_dfg_from_state_dict(state_dict: Dict[str, Any]) -> bool:
    """根据权重键推断是否为带 DFG 分支的检查点。"""
    return any(k.startswith("dfg_node_embed.") for k in state_dict.keys())


def parse_multimodal_checkpoint(
    raw: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    解析训练保存的检查点：支持 {state_dict, meta} 或裸 state_dict。
    返回 (state_dict, meta)。
    """
    if isinstance(raw, dict) and "state_dict" in raw:
        meta = raw.get("meta")
        return raw["state_dict"], dict(meta) if isinstance(meta, dict) else {}
    if isinstance(raw, dict):
        return raw, {}
    return {}, {}


class MultiModalFusionModel(nn.Module if TORCH_AVAILABLE else object):
    """
    多模态融合模型：图分支 + 序列分支 + 跨模态注意力。
    use_dfg=True 时增加 DFG 图分支，与 CFG 图嵌入拼接后压回 output_dim。
    """

    def __init__(
        self,
        pcode_vocab_size: int = 256,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 128,
        max_seq_len: int = 512,
        max_graph_nodes: int = 128,
        num_gnn_layers: int = 2,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        *,
        use_dfg: bool = False,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for MultiModalFusionModel")
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.use_dfg = use_dfg

        # 序列分支：P-code token embedding + 跳转位置编码 + Transformer
        self.seq_embed = nn.Embedding(pcode_vocab_size, embed_dim, padding_idx=0)
        self.jump_proj = nn.Linear(1, embed_dim)  # jump mask -> 位置编码增量
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        # 关闭嵌套张量快速路径，避免 PyTorch 2.x 在 src_key_padding_mask 下打印 prototype 警告
        _te_kw: Dict[str, Any] = {}
        if "enable_nested_tensor" in inspect.signature(nn.TransformerEncoder.__init__).parameters:
            _te_kw["enable_nested_tensor"] = False
        self.transformer = nn.TransformerEncoder(encoder_layer, num_transformer_layers, **_te_kw)
        self.seq_proj = nn.Linear(embed_dim, output_dim)

        # 图分支：简化的 GNN（消息传递）
        self.node_embed = nn.Embedding(pcode_vocab_size, embed_dim)  # 节点 id 或 pcode 聚合
        self.gnn_layers = nn.ModuleList(
            [
                nn.Linear(embed_dim * 2, hidden_dim),
                nn.Linear(hidden_dim, embed_dim),
            ]
        )
        self.gnn_proj = nn.Linear(embed_dim, output_dim)

        if use_dfg:
            self.dfg_node_embed = nn.Embedding(512, embed_dim, padding_idx=0)
            self.dfg_gnn_proj = nn.Linear(embed_dim, output_dim)
            self.graph_fuse = nn.Linear(output_dim * 2, output_dim)
        else:
            self.dfg_node_embed = None  # type: ignore[assignment]
            self.dfg_gnn_proj = None  # type: ignore[assignment]
            self.graph_fuse = None  # type: ignore[assignment]

        # 跨模态注意力：图嵌入 attend to 序列嵌入
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_proj = nn.Linear(output_dim * 2, output_dim)

    def _seq_forward(
        self,
        token_ids: "torch.Tensor",
        jump_mask: "torch.Tensor",
        padding_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        B, L = token_ids.shape
        x = self.seq_embed(token_ids)
        jump_enc = self.jump_proj(jump_mask.float().unsqueeze(-1))
        x = x + jump_enc
        if padding_mask is not None:
            x = self.transformer(x, src_key_padding_mask=padding_mask)
        else:
            x = self.transformer(x)
        x = x.mean(dim=1)
        return self.seq_proj(x)

    def _graph_forward(
        self,
        node_features: "torch.Tensor",
        edge_index: "torch.Tensor",
    ) -> "torch.Tensor":
        h = self.node_embed(node_features)  # (B, N, E)
        B, N, E = h.shape
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            # 批量稀疏聚合：一次 scatter_add 替代 B 次 Python 循环
            h_flat = h.reshape(B * N, E)
            agg = torch.zeros_like(h_flat)
            agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, E), h_flat[src])
            deg = torch.zeros(B * N, device=h.device)
            deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
            deg = deg.clamp(min=1).unsqueeze(-1)
            agg = agg / deg
            h_flat = torch.cat([h_flat, agg], dim=-1)
            h_flat = self.gnn_layers[0](h_flat)
            h_flat = torch.relu(h_flat)
            h_flat = self.gnn_layers[1](h_flat)
            h = h_flat.reshape(B, N, E)
        h = h.mean(dim=1)
        return self.gnn_proj(h)

    def _dfg_graph_forward(
        self,
        node_features: "torch.Tensor",
        edge_index: "torch.Tensor",
    ) -> "torch.Tensor":
        if not self.use_dfg or self.dfg_node_embed is None or self.dfg_gnn_proj is None:
            raise RuntimeError("DFG branch not enabled")
        h = self.dfg_node_embed(node_features)  # (B, N, E)
        B, N, E = h.shape
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            # 批量稀疏聚合：一次 scatter_add 替代 B 次 Python 循环
            h_flat = h.reshape(B * N, E)
            agg = torch.zeros_like(h_flat)
            agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, E), h_flat[src])
            deg = torch.zeros(B * N, device=h.device)
            deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
            deg = deg.clamp(min=1).unsqueeze(-1)
            agg = agg / deg
            h_flat = h_flat + agg  # residual for DFG
            h = h_flat.reshape(B, N, E)
        h = h.mean(dim=1)
        return self.dfg_gnn_proj(h)

    def forward(
        self,
        token_ids: "torch.Tensor",
        jump_mask: "torch.Tensor",
        graph_node_features: "torch.Tensor",
        edge_index: "torch.Tensor",
        padding_mask: Optional["torch.Tensor"] = None,
        dfg_node_features: Optional["torch.Tensor"] = None,
        dfg_edge_index: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        seq_emb = self._seq_forward(token_ids, jump_mask, padding_mask)
        graph_emb = self._graph_forward(graph_node_features, edge_index)

        if self.use_dfg:
            if dfg_node_features is None:
                dfg_node_features = torch.zeros_like(graph_node_features)
            if dfg_edge_index is None:
                dfg_edge_index = torch.zeros(
                    2, 0, dtype=torch.long, device=graph_node_features.device
                )
            dfg_emb = self._dfg_graph_forward(dfg_node_features, dfg_edge_index)
            if self.graph_fuse is not None:
                graph_emb = self.graph_fuse(torch.cat([graph_emb, dfg_emb], dim=-1))

        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)
        if seq_emb.dim() == 1:
            seq_emb = seq_emb.unsqueeze(0)
        graph_q = graph_emb.unsqueeze(1)
        seq_kv = seq_emb.unsqueeze(1)
        attn_out, _ = self.cross_attn(graph_q, seq_kv, seq_kv)
        attn_out = attn_out.squeeze(1)
        fused = torch.cat([graph_emb, attn_out], dim=-1)
        out = self.fusion_proj(fused)
        out = F.normalize(out, dim=-1)
        if out.shape[0] == 1:
            return out.squeeze(0)
        return out


def _clamp_edge_index(edge_idx: List[List[int]], num_nodes: int) -> List[List[int]]:
    """Filter out-of-range or negative edge indices."""
    if not edge_idx or len(edge_idx) < 2 or not edge_idx[0]:
        return edge_idx if edge_idx else [[], []]
    src, dst = edge_idx[0], edge_idx[1]
    good = [(s, d) for s, d in zip(src, dst) if 0 <= s < num_nodes and 0 <= d < num_nodes]
    if not good:
        return [[], []]
    return [[s for s, _ in good], [d for _, d in good]]


def _clamp_ids(ids: List[int], max_val: int, unk: int = 1) -> List[int]:
    """Clamp index values to [0, max_val), replacing out-of-range with unk."""
    return [unk if v < 0 or v >= max_val else v for v in ids]


def _tensorize_multimodal(
    multimodal: Dict[str, Any],
    vocab: Dict[str, int],
    device: Optional["torch.device"] = None,
    max_seq_len: int = 512,
    max_graph_nodes: int = 128,
    max_dfg_nodes: int = 128,
    pcode_vocab_size: int = 256,
) -> Tuple[
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
]:
    """将 multimodal 特征转为 tensor。

    返回 7 元组，位置映射到 forward() 参数：
        (token_t, jump_t, node_t, edge_t, pad_mask, dfg_node_t, dfg_edge_t)
         ↓          ↓        ↓        ↓        ↓         ↓          ↓
      token_ids  jump_mask graph_node_ edge_index padding_ dfg_node_ dfg_edge_
                            features             mask     features   index
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")
    seq = multimodal.get("sequence") or {}
    graph = multimodal.get("graph") or {}
    jump_mask = seq.get("jump_mask") or []

    # ── 序列 tokens：优先预计算 pcode_token_ids ──
    precomp_ids = seq.get("pcode_token_ids")
    if precomp_ids and isinstance(precomp_ids, list):
        token_ids_raw = list(precomp_ids[:max_seq_len])
    else:
        tokens = seq.get("pcode_tokens") or []
        token_ids_raw = [vocab.get(t, 1) for t in tokens[:max_seq_len]]
    token_ids = _clamp_ids(token_ids_raw, pcode_vocab_size)
    jump = list(jump_mask[:max_seq_len])
    if not token_ids:
        token_ids = [1]
        jump = [0]
    pad_len = max_seq_len - len(token_ids)
    token_ids = token_ids + [0] * pad_len
    jump = jump + [0] * pad_len
    token_t = torch.tensor([token_ids], dtype=torch.long)
    jump_t = torch.tensor([jump], dtype=torch.long)
    pad_mask = torch.zeros(1, max_seq_len, dtype=torch.bool)
    if pad_len > 0:
        pad_mask[0, -pad_len:] = True

    # ── Graph nodes：优先预计算 opcode_id ──
    node_feats = graph.get("node_features") or []
    nf_flat: List[int] = []
    for nf in node_feats[:max_graph_nodes]:
        if isinstance(nf, dict):
            precomp_op = nf.get("opcode_id")
            if precomp_op is not None:
                nf_flat.append(int(precomp_op))
                continue
            opcodes = nf.get("pcode_opcodes", []) or []
        else:
            opcodes = nf
        idx = vocab.get(opcodes[0], 1) if opcodes else 0
        nf_flat.append(idx)
    if not nf_flat:
        nf_flat = [0]
    nf_flat = _clamp_ids(nf_flat, pcode_vocab_size)
    node_t = torch.tensor([nf_flat], dtype=torch.long)
    edge_idx = graph.get("edge_index") or [[], []]
    edge_idx = _clamp_edge_index(edge_idx, len(nf_flat))
    edge_t = (
        torch.tensor(edge_idx, dtype=torch.long)
        if edge_idx[0]
        else torch.zeros(2, 0, dtype=torch.long)
    )

    dfg = multimodal.get("dfg") or {}
    dfg_nf = dfg.get("node_features") or []
    dfg_ids: List[int] = []
    for x in dfg_nf[:max_dfg_nodes]:
        if isinstance(x, int):
            dfg_ids.append(int(x) % 512)
        else:
            dfg_ids.append(0)
    if not dfg_ids:
        dfg_ids = [0]
    dfg_node_t = torch.tensor([dfg_ids], dtype=torch.long)
    dfg_e = dfg.get("edge_index") or [[], []]
    dfg_e = _clamp_edge_index(dfg_e, len(dfg_ids))
    dfg_edge_t = (
        torch.tensor(dfg_e, dtype=torch.long) if dfg_e[0] else torch.zeros(2, 0, dtype=torch.long)
    )

    if device:
        token_t = token_t.to(device)
        jump_t = jump_t.to(device)
        pad_mask = pad_mask.to(device)
        node_t = node_t.to(device)
        edge_t = edge_t.to(device)
        dfg_node_t = dfg_node_t.to(device)
        dfg_edge_t = dfg_edge_t.to(device)
    return token_t, jump_t, node_t, edge_t, pad_mask, dfg_node_t, dfg_edge_t


def tensorize_multimodal_many(
    multimodals: List[Dict[str, Any]],
    vocab: Dict[str, int],
    device: Optional["torch.device"] = None,
    max_seq_len: int = 512,
    max_graph_nodes: int = 128,
    max_dfg_nodes: int = 128,
    pcode_vocab_size: int = 256,
) -> Tuple[
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
]:
    """批量将 multimodal 特征转为 batched tensor（向量化版）。

    相比旧版改进：
    - Pass 1 用 list comprehension + pre-allocated lists 替代逐元素 append
    - Pass 3 用 numpy 一次性填充序列/节点 batch，避免逐样本 torch.tensor 构造
    - GPU 传输合并为单次（7 个 tensor 一次性 .to(device)）
    - edge_index 先收集为 Python list 再一次 torch.cat

    返回与 _tensorize_multimodal 相同的 7 元组，但第一维均为 B（批量大小）。
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")
    if not multimodals:
        raise ValueError("multimodals must be non-empty")

    import numpy as np

    B = len(multimodals)
    _UNK = 1

    # -- Pass 1: extract per-item token/node lists --
    # 优化：优先读预计算的 pcode_token_ids / opcode_id（避免 vocab.get() 循环）
    per_item_tokens: List[List[int]] = []
    per_item_jumps: List[List[int]] = []
    per_item_nodes: List[List[int]] = []
    per_item_edge_src: List[List[int]] = []
    per_item_edge_dst: List[List[int]] = []
    per_item_dfg_nodes: List[List[int]] = []
    per_item_dfg_edge_src: List[List[int]] = []
    per_item_dfg_edge_dst: List[List[int]] = []

    _vocab_get = vocab.get
    _empty_list: List[int] = []

    for mm in multimodals:
        seq = mm.get("sequence") or {}
        graph = mm.get("graph") or {}
        jump_mask = seq.get("jump_mask") or []

        # ── 序列 tokens：优先预计算 pcode_token_ids ──
        precomp_ids = seq.get("pcode_token_ids")
        if precomp_ids and isinstance(precomp_ids, list):
            # 已有 int 列表，直接切片 + clamp（C 级速度，无 dict 查表）
            t_raw = precomp_ids[:max_seq_len]
            t_ids = [unk if v < 0 or v >= pcode_vocab_size else v for v in t_raw]
        else:
            tokens = seq.get("pcode_tokens") or []
            t_raw = [_vocab_get(t, _UNK) for t in tokens[:max_seq_len]]
            t_ids = _clamp_ids(t_raw, pcode_vocab_size)

        jmp = list(jump_mask[:max_seq_len])
        if not t_ids:
            t_ids = [_UNK]
            jmp = [0]
        per_item_tokens.append(t_ids)
        per_item_jumps.append(jmp)

        # ── Graph nodes：优先预计算 opcode_id ──
        node_feats = graph.get("node_features") or []
        nf: List[int] = []
        append_nf = nf.append
        for n_feat in node_feats[:max_graph_nodes]:
            if isinstance(n_feat, dict):
                precomp_op = n_feat.get("opcode_id")
                if precomp_op is not None:
                    append_nf(int(precomp_op))
                    continue
                opcodes = n_feat.get("pcode_opcodes") or []
            else:
                opcodes = n_feat
            append_nf(_vocab_get(opcodes[0], _UNK) if opcodes else 0)
        if not nf:
            nf = [0]
        nf = _clamp_ids(nf, pcode_vocab_size)
        per_item_nodes.append(nf)

        ei = graph.get("edge_index") or [[], []]
        ei = _clamp_edge_index(ei, len(nf))
        per_item_edge_src.append(ei[0] if ei and ei[0] else _empty_list)
        per_item_edge_dst.append(ei[1] if ei and len(ei) > 1 and ei[1] else _empty_list)

        dfg = mm.get("dfg") or {}
        dfg_nf = dfg.get("node_features") or []
        dfg_ids = [(int(x) % 512) if isinstance(x, int) else 0 for x in dfg_nf[:max_dfg_nodes]]
        if not dfg_ids:
            dfg_ids = [0]
        per_item_dfg_nodes.append(dfg_ids)

        dei = dfg.get("edge_index") or [[], []]
        dei = _clamp_edge_index(dei, len(dfg_ids))
        per_item_dfg_edge_src.append(dei[0] if dei and dei[0] else _empty_list)
        per_item_dfg_edge_dst.append(dei[1] if dei and len(dei) > 1 and dei[1] else _empty_list)

    # -- Pass 2: compute batch maxima --
    max_actual_seq = max(len(t) for t in per_item_tokens)
    max_actual_nodes = max(len(n) for n in per_item_nodes)
    max_actual_dfg_nodes = max(len(d) for d in per_item_dfg_nodes)

    # -- Pass 3: build batched tensors via numpy for sequence/node data --
    # Sequence tensors
    token_np = np.zeros((B, max_actual_seq), dtype=np.int64)
    jump_np = np.zeros((B, max_actual_seq), dtype=np.int64)
    pad_mask_np = np.ones((B, max_actual_seq), dtype=np.bool_)

    for i in range(B):
        tok = per_item_tokens[i]
        jmp = per_item_jumps[i]
        L = len(tok)
        token_np[i, :L] = tok
        jump_np[i, :L] = jmp
        pad_mask_np[i, :L] = False

    # Graph node tensors
    node_np = np.zeros((B, max_actual_nodes), dtype=np.int64)
    for i in range(B):
        nodes = per_item_nodes[i]
        N = len(nodes)
        node_np[i, :N] = nodes

    # DFG node tensors
    dfg_node_np = np.zeros((B, max_actual_dfg_nodes), dtype=np.int64)
    for i in range(B):
        dnodes = per_item_dfg_nodes[i]
        DN = len(dnodes)
        dfg_node_np[i, :DN] = dnodes

    # Graph edge indices: collect then single concat
    all_edge_src: List[int] = []
    all_edge_dst: List[int] = []
    for i in range(B):
        es = per_item_edge_src[i]
        ed = per_item_edge_dst[i]
        if es:
            offset = i * max_actual_nodes
            all_edge_src.extend(s + offset for s in es)
            all_edge_dst.extend(d + offset for d in ed)

    # DFG edge indices
    all_dfg_src: List[int] = []
    all_dfg_dst: List[int] = []
    for i in range(B):
        ds = per_item_dfg_edge_src[i]
        dd = per_item_dfg_edge_dst[i]
        if ds:
            offset = i * max_actual_dfg_nodes
            all_dfg_src.extend(s + offset for s in ds)
            all_dfg_dst.extend(d + offset for d in dd)

    # -- Single GPU transfer: build all tensors on CPU, then one .to(device) --
    token_batch = torch.from_numpy(token_np)
    jump_batch = torch.from_numpy(jump_np)
    pad_mask_batch = torch.from_numpy(pad_mask_np)
    node_batch = torch.from_numpy(node_np)
    dfg_node_batch = torch.from_numpy(dfg_node_np)

    if all_edge_src:
        edge_t = torch.stack([
            torch.tensor(all_edge_src, dtype=torch.long),
            torch.tensor(all_edge_dst, dtype=torch.long),
        ])
    else:
        edge_t = torch.zeros(2, 0, dtype=torch.long)

    if all_dfg_src:
        dfg_edge_t = torch.stack([
            torch.tensor(all_dfg_src, dtype=torch.long),
            torch.tensor(all_dfg_dst, dtype=torch.long),
        ])
    else:
        dfg_edge_t = torch.zeros(2, 0, dtype=torch.long)

    if device:
        # 单次传输：对同一 device 的 tensor 合并传输
        token_batch = token_batch.to(device, non_blocking=True)
        jump_batch = jump_batch.to(device, non_blocking=True)
        pad_mask_batch = pad_mask_batch.to(device, non_blocking=True)
        node_batch = node_batch.to(device, non_blocking=True)
        edge_t = edge_t.to(device, non_blocking=True)
        dfg_node_batch = dfg_node_batch.to(device, non_blocking=True)
        dfg_edge_t = dfg_edge_t.to(device, non_blocking=True)

    return token_batch, jump_batch, node_batch, edge_t, pad_mask_batch, dfg_node_batch, dfg_edge_t


def get_default_vocab() -> Dict[str, int]:
    """返回常见 P-code opcode 的默认 vocab。"""
    common_ops = [
        "",
        "[UNK]",
        "COPY",
        "LOAD",
        "STORE",
        "BRANCH",
        "CBRANCH",
        "BRANCHIND",
        "CALL",
        "CALLIND",
        "RETURN",
        "INT_ADD",
        "INT_SUB",
        "INT_AND",
        "INT_OR",
        "INT_XOR",
        "INT_MULT",
        "INT_DIV",
        "INT_EQUAL",
        "INT_NOTEQUAL",
        "INT_LESS",
        "INT_SLESS",
        "INT_NEGATE",
        "INT_ZEXT",
        "INT_SEXT",
        "INT_2COMP",
        "INT_LEFT",
        "INT_RIGHT",
        "INT_SRIGHT",
        "INT_CARRY",
        "INT_SCARRY",
        "INT_SBORROW",
        "POPCOUNT",
    ]
    return {op: i for i, op in enumerate(common_ops)}
