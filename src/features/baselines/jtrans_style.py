"""
jTrans 风格基线（项目内近似实现）：以 **基本块（CFG 节点）顺序** 上的 P-code opcode 为序列，
经轻量嵌入 + 聚合得到函数向量；**非** 官方 vul337/jTrans 仓库的预训练权重。

与 SAFE（全函数 pcode 线性序列）区分：本基线在 token 前加块下标前缀 `@<i>:<opcode>`，
强调块结构，便于与论文中「块级 Transformer」叙事对齐；输出维度与 SAFE 一致（128），
可直接写入同一 EmbeddingItem / eval_bcsd 流程。

外部完整 jTrans：见 docs/BASELINE_AND_EVAL.md 中的复现指引。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

_OUTPUT_DIM = 128
_EMBED_DIM = 64


def _opcodes_from_node_feature(nf: Any) -> List[str]:
    if isinstance(nf, list):
        return [str(x) for x in nf if x]
    if isinstance(nf, dict):
        return [str(x) for x in (nf.get("pcode_opcodes") or []) if x]
    return []


def _collect_vocab_from_multimodal(mm: Dict[str, Any], vocab: Dict[str, int]) -> None:
    """块序 opcode → @块索引:操作码 形式的 token。"""
    graph = mm.get("graph") or {}
    nfs = graph.get("node_features") or []
    for i, nf in enumerate(nfs):
        for op in _opcodes_from_node_feature(nf):
            tok = f"@{i}:{op}"
            if tok not in vocab:
                vocab[tok] = len(vocab)


def _collect_vocab(features: Dict[str, Any]) -> Dict[str, int]:
    vocab: Dict[str, int] = {"": 0, "[UNK]": 1}
    for item in features.get("functions") or []:
        feats = item.get("features") or {}
        mm = feats.get("multimodal") or {}
        _collect_vocab_from_multimodal(mm, vocab)
    return vocab


def jtrans_style_tokenize(
    multimodal: Dict[str, Any],
    vocab: Dict[str, int],
    max_len: int = 512,
) -> Tuple[List[int], List[bool]]:
    graph = multimodal.get("graph") or {}
    nfs = graph.get("node_features") or []
    tokens: List[str] = []
    for i, nf in enumerate(nfs):
        for op in _opcodes_from_node_feature(nf):
            tokens.append(f"@{i}:{op}")
            if len(tokens) >= max_len:
                break
        if len(tokens) >= max_len:
            break
    ids = [vocab.get(t, 1) for t in tokens[:max_len]]
    pad_len = max_len - len(ids)
    ids = ids + [0] * pad_len
    pad_mask = [False] * (max_len - pad_len) + [True] * pad_len
    return ids, pad_mask


def jtrans_style_tokenize_many(
    multimodals: Sequence[Dict[str, Any]],
    vocab: Dict[str, int],
    *,
    max_len: int = 512,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for jtrans_style_tokenize_many")
    ids_list: List[List[int]] = []
    pad_list: List[List[bool]] = []
    for mm in multimodals:
        ids, pad = jtrans_style_tokenize(mm, vocab, max_len=max_len)
        ids_list.append(ids)
        pad_list.append(pad)
    return (
        torch.tensor(ids_list, dtype=torch.long),
        torch.tensor(pad_list, dtype=torch.bool),
    )


class _JTransStyleEncoder(nn.Module if TORCH_AVAILABLE else object):
    """与 SAFE 同构的序列编码器（不同 token 来源）。"""

    def __init__(self, vocab_size: int, embed_dim: int = _EMBED_DIM, output_dim: int = _OUTPUT_DIM):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for jTrans-style baseline")
        super().__init__()
        self.embed = nn.Embedding(max(vocab_size, 16), embed_dim, padding_idx=0)
        self.proj = nn.Linear(embed_dim, output_dim)

    def forward(self, token_ids: "torch.Tensor", pad_mask: "torch.Tensor | None" = None) -> "torch.Tensor":
        x = self.embed(token_ids)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            lengths = (~pad_mask).float().sum(dim=1, keepdim=True).clamp(min=1)
            x = x.sum(dim=1) / lengths
        else:
            x = x.mean(dim=1)
        return self.proj(x)


def jtrans_style_save_model(
    model: "_JTransStyleEncoder",
    vocab: Dict[str, int],
    path: str,
    embed_dim: int = _EMBED_DIM,
    output_dim: int = _OUTPUT_DIM,
) -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "embed_dim": embed_dim,
            "output_dim": output_dim,
            "baseline": "jtrans_style",
        },
        path,
    )


def jtrans_style_load_model(path: str) -> Tuple["_JTransStyleEncoder", Dict[str, int]]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError(f"无效的 jTrans-style 模型格式: {path}")
    state = data.get("state_dict")
    vocab = data.get("vocab")
    embed_dim = data.get("embed_dim", _EMBED_DIM)
    output_dim = data.get("output_dim", _OUTPUT_DIM)
    if state is None or vocab is None:
        raise ValueError(f"jTrans-style 模型缺少 state_dict 或 vocab: {path}")
    if isinstance(state, dict) and "embed.weight" in state:
        vocab_size = int(state["embed.weight"].shape[0])
    else:
        vocab_size = max(len(vocab), 256)
    model = _JTransStyleEncoder(vocab_size=vocab_size, embed_dim=embed_dim, output_dim=output_dim)
    model.load_state_dict(state)
    model.eval()
    return model, vocab


def embed_batch_jtrans_style(
    features: Dict[str, Any],
    model_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    jTrans 风格嵌入：输出与 embed_batch / embed_batch_safe 相同的 List[{name, vector}]（128 维）。
    """
    funcs = features.get("functions") or []
    if not funcs:
        return []

    if not TORCH_AVAILABLE:
        return [{"name": f.get("name", ""), "vector": [0.0] * _OUTPUT_DIM} for f in funcs]

    if model_path and os.path.isfile(model_path):
        model, vocab = jtrans_style_load_model(model_path)
    else:
        vocab = _collect_vocab(features)
        vocab_size = max(len(vocab), 256)
        model = _JTransStyleEncoder(vocab_size=vocab_size)
        model.eval()

    out: List[Dict[str, Any]] = []
    max_len = 512
    with torch.no_grad():
        for item in funcs:
            name = item.get("name", "")
            feats = item.get("features") or {}
            mm = feats.get("multimodal") or {}
            ids, pad_mask_list = jtrans_style_tokenize(mm, vocab, max_len=max_len)
            token_t = torch.tensor([ids], dtype=torch.long)
            pad_mask_t = torch.tensor([pad_mask_list], dtype=torch.bool)
            vec = model(token_t, pad_mask_t)
            out.append({"name": name, "vector": vec.squeeze(0).tolist()})
    return out
