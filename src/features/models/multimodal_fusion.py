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
            agg = torch.zeros_like(h)
            for b in range(B):
                offset = b * N
                mask = (src >= offset) & (src < offset + N)
                s = src[mask] - offset
                d = dst[mask] - offset
                if s.numel() == 0:
                    continue
                agg[b].index_add_(0, d, h[b][s])
                deg = torch.zeros(N, device=h.device)
                deg.index_add_(0, d, torch.ones_like(s, dtype=torch.float))
                deg = deg.clamp(min=1).unsqueeze(-1)
                agg[b] = agg[b] / deg
            h = torch.cat([h, agg], dim=-1)
            h = self.gnn_layers[0](h)
            h = torch.relu(h)
            h = self.gnn_layers[1](h)
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
            agg = torch.zeros_like(h)
            for b in range(B):
                offset = b * N
                mask = (src >= offset) & (src < offset + N)
                s = src[mask] - offset
                d = dst[mask] - offset
                if s.numel() == 0:
                    continue
                agg[b].index_add_(0, d, h[b][s])
                deg = torch.zeros(N, device=h.device)
                deg.index_add_(0, d, torch.ones_like(s, dtype=torch.float))
                deg = deg.clamp(min=1).unsqueeze(-1)
                agg[b] = agg[b] / deg
            h = h + agg  # residual for DFG
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
        if out.shape[0] == 1:
            return out.squeeze(0)
        return out


def _tensorize_multimodal(
    multimodal: Dict[str, Any],
    vocab: Dict[str, int],
    device: Optional["torch.device"] = None,
    max_seq_len: int = 512,
    max_graph_nodes: int = 128,
    max_dfg_nodes: int = 128,
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
    tokens = seq.get("pcode_tokens") or []
    jump_mask = seq.get("jump_mask") or []
    token_ids = [vocab.get(t, 1) for t in tokens[:max_seq_len]]
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
    node_feats = graph.get("node_features") or []
    nf_flat: List[int] = []
    for nf in node_feats[:max_graph_nodes]:
        opcodes = nf if isinstance(nf, list) else nf.get("pcode_opcodes", []) or []
        idx = vocab.get(opcodes[0], 1) if opcodes else 0
        nf_flat.append(idx)
    if not nf_flat:
        nf_flat = [0]
    node_t = torch.tensor([nf_flat], dtype=torch.long)
    edge_idx = graph.get("edge_index") or [[], []]
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
) -> Tuple[
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
]:
    """批量将 multimodal 特征转为 batched tensor。

    返回与 _tensorize_multimodal 相同的 7 元组，但第一维均为 B（批量大小）。
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")
    if not multimodals:
        raise ValueError("multimodals must be non-empty")

    B = len(multimodals)

    # -- Pass 1: compute per-item sizes --
    per_item_tokens: List[List[int]] = []
    per_item_jumps: List[List[int]] = []
    per_item_nodes: List[List[int]] = []
    per_item_edges: List[List[List[int]]] = []
    per_item_dfg_nodes: List[List[int]] = []
    per_item_dfg_edges: List[List[List[int]]] = []

    for mm in multimodals:
        seq = mm.get("sequence") or {}
        graph = mm.get("graph") or {}
        tokens = seq.get("pcode_tokens") or []
        jump_mask = seq.get("jump_mask") or []
        token_ids = [vocab.get(t, 1) for t in tokens[:max_seq_len]]
        jump = list(jump_mask[:max_seq_len])
        if not token_ids:
            token_ids = [1]
            jump = [0]
        per_item_tokens.append(token_ids)
        per_item_jumps.append(jump)

        node_feats = graph.get("node_features") or []
        nf_flat: List[int] = []
        for nf in node_feats[:max_graph_nodes]:
            opcodes = nf if isinstance(nf, list) else nf.get("pcode_opcodes", []) or []
            idx = vocab.get(opcodes[0], 1) if opcodes else 0
            nf_flat.append(idx)
        if not nf_flat:
            nf_flat = [0]
        per_item_nodes.append(nf_flat)

        edge_idx = graph.get("edge_index") or [[], []]
        per_item_edges.append(edge_idx)

        dfg = mm.get("dfg") or {}
        dfg_nf = dfg.get("node_features") or []
        dfg_ids: List[int] = []
        for x in dfg_nf[:max_dfg_nodes]:
            if isinstance(x, int):
                dfg_ids.append(int(x) % 512)
            else:
                dfg_ids.append(0)
        if not dfg_ids:
            dfg_ids = [0]
        per_item_dfg_nodes.append(dfg_ids)

        dfg_e = dfg.get("edge_index") or [[], []]
        per_item_dfg_edges.append(dfg_e)

    # -- Pass 2: compute batch maxima --
    max_actual_seq = max(len(t) for t in per_item_tokens)
    max_actual_nodes = max(len(n) for n in per_item_nodes)
    max_actual_edges = max(len(e[0]) for e in per_item_edges)
    max_actual_dfg_nodes = max(len(d) for d in per_item_dfg_nodes)
    max_actual_dfg_edges = max(len(e[0]) for e in per_item_dfg_edges)

    seq_pad = max_actual_seq
    node_pad = max_actual_nodes
    dfg_node_pad = max_actual_dfg_nodes

    # -- Pass 3: build batched tensors --
    token_batch = torch.zeros(B, seq_pad, dtype=torch.long)
    jump_batch = torch.zeros(B, seq_pad, dtype=torch.long)
    pad_mask_batch = torch.ones(B, seq_pad, dtype=torch.bool)
    node_batch = torch.zeros(B, node_pad, dtype=torch.long)
    edge_src_list: List[torch.Tensor] = []
    edge_dst_list: List[torch.Tensor] = []
    dfg_node_batch = torch.zeros(B, dfg_node_pad, dtype=torch.long)
    dfg_edge_src_list: List[torch.Tensor] = []
    dfg_edge_dst_list: List[torch.Tensor] = []

    for i in range(B):
        # Sequence
        tok = per_item_tokens[i]
        jmp = per_item_jumps[i]
        L = len(tok)
        token_batch[i, :L] = torch.tensor(tok, dtype=torch.long)
        jump_batch[i, :L] = torch.tensor(jmp, dtype=torch.long)
        pad_mask_batch[i, :L] = False

        # Graph nodes
        nodes = per_item_nodes[i]
        N = len(nodes)
        node_batch[i, :N] = torch.tensor(nodes, dtype=torch.long)

        # Graph edges: offset by i * node_pad
        ei = per_item_edges[i]
        if ei and ei[0]:
            src = torch.tensor(ei[0], dtype=torch.long) + i * node_pad
            dst = torch.tensor(ei[1], dtype=torch.long) + i * node_pad
        else:
            src = torch.zeros(0, dtype=torch.long)
            dst = torch.zeros(0, dtype=torch.long)
        edge_src_list.append(src)
        edge_dst_list.append(dst)

        # DFG nodes
        dnodes = per_item_dfg_nodes[i]
        DN = len(dnodes)
        dfg_node_batch[i, :DN] = torch.tensor(dnodes, dtype=torch.long)

        # DFG edges: offset by i * dfg_node_pad
        dei = per_item_dfg_edges[i]
        if dei and dei[0]:
            dsrc = torch.tensor(dei[0], dtype=torch.long) + i * dfg_node_pad
            ddst = torch.tensor(dei[1], dtype=torch.long) + i * dfg_node_pad
        else:
            dsrc = torch.zeros(0, dtype=torch.long)
            ddst = torch.zeros(0, dtype=torch.long)
        dfg_edge_src_list.append(dsrc)
        dfg_edge_dst_list.append(ddst)

    if max_actual_edges > 0:
        edge_t = torch.stack(
            [torch.cat(edge_src_list), torch.cat(edge_dst_list)]
        )
    else:
        edge_t = torch.zeros(2, 0, dtype=torch.long)

    if max_actual_dfg_edges > 0:
        dfg_edge_t = torch.stack(
            [torch.cat(dfg_edge_src_list), torch.cat(dfg_edge_dst_list)]
        )
    else:
        dfg_edge_t = torch.zeros(2, 0, dtype=torch.long)

    if device:
        token_batch = token_batch.to(device)
        jump_batch = jump_batch.to(device)
        pad_mask_batch = pad_mask_batch.to(device)
        node_batch = node_batch.to(device)
        edge_t = edge_t.to(device)
        dfg_node_batch = dfg_node_batch.to(device)
        dfg_edge_t = dfg_edge_t.to(device)

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
