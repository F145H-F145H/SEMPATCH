"""
SAFE 风格基线：轻量序列编码器（token embedding + 聚合），用于与 SemPatch 对比。

SAFE 原文：汇编指令 → Instruction2Vec → 自注意力 RNN 聚合 → 函数嵌入。
本实现采用 P-code token embedding + mean 聚合，输入与 embed_batch 兼容的 FeaturesDict。
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


def _collect_vocab_from_multimodal(mm: Dict[str, Any], vocab: Dict[str, int]) -> None:
    """从单个 multimodal 字典收集 pcode token，追加到 vocab。"""
    seq = mm.get("sequence") or {}
    for t in seq.get("pcode_tokens") or []:
        if t and t not in vocab:
            vocab[t] = len(vocab)
    graph = mm.get("graph") or {}
    for nf in graph.get("node_features") or []:
        opcodes = nf if isinstance(nf, list) else (nf.get("pcode_opcodes") or [])
        for op in opcodes:
            if op and op not in vocab:
                vocab[op] = len(vocab)


def _collect_vocab(features: Dict[str, Any]) -> Dict[str, int]:
    """从 FeaturesDict 收集 pcode token 构建 vocab。"""
    vocab: Dict[str, int] = {"[PAD]": 0, "[UNK]": 1}
    funcs = features.get("functions") or []
    for item in funcs:
        feats = item.get("features") or {}
        mm = feats.get("multimodal") or {}
        _collect_vocab_from_multimodal(mm, vocab)
    return vocab


def collect_vocab_from_features_file(features_path: str) -> Dict[str, int]:
    """
    从 library_features.json 格式（{function_id: multimodal_dict}）构建 vocab。
    用于训练时统一 vocab，保证 tokenize 与模型 embed 层一致。

    注意：大库会整文件 json.load，易 OOM；大侧车请改用 collect_vocab_from_features_jsonl。
    """
    vocab: Dict[str, int] = {"[PAD]": 0, "[UNK]": 1}
    with open(features_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return vocab
    for mm in data.values():
        if isinstance(mm, dict):
            _collect_vocab_from_multimodal(mm, vocab)
    return vocab


def collect_vocab_from_features_jsonl(features_path: str) -> Dict[str, int]:
    """
    从 JSONL 侧车（每行 {"function_id","multimodal"}）流式扫描构建 vocab。
    不保留整库 multimodal，峰值内存远低于对 library_features.json 的 json.load。
    """
    vocab: Dict[str, int] = {"[PAD]": 0, "[UNK]": 1}
    from utils.precomputed_multimodal_io import iter_jsonl_sidecar

    for _fid, mm in iter_jsonl_sidecar(features_path):
        if isinstance(mm, dict):
            _collect_vocab_from_multimodal(mm, vocab)
    return vocab


def safe_tokenize(
    multimodal: Dict[str, Any],
    vocab: Dict[str, int],
    max_len: int = 512,
) -> Tuple[List[int], List[bool]]:
    """
    将 multimodal 特征转为 token_ids 与 pad_mask，供 _SafeEncoder 训练使用。
    从 sequence.pcode_tokens 取 token，按 vocab 映射，填充至 max_len。
    返回 (token_ids, pad_mask)，pad_mask 中 True 表示 pad 位置。
    """
    seq = multimodal.get("sequence") or {}
    tokens = seq.get("pcode_tokens") or []
    ids = [vocab.get(t, 1) for t in tokens[:max_len]]
    pad_len = max_len - len(ids)
    ids = ids + [0] * pad_len
    pad_mask = [False] * (max_len - pad_len) + [True] * pad_len
    return ids, pad_mask


def safe_tokenize_many(
    multimodals: Sequence[Dict[str, Any]],
    vocab: Dict[str, int],
    *,
    max_len: int = 512,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    批量 tokenize，返回 (token_ids, pad_mask)。

    - token_ids: (B, max_len) long
    - pad_mask: (B, max_len) bool, True 表示 pad
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for safe_tokenize_many")
    ids_list: List[List[int]] = []
    pad_list: List[List[bool]] = []
    for mm in multimodals:
        ids, pad = safe_tokenize(mm, vocab, max_len=max_len)
        ids_list.append(ids)
        pad_list.append(pad)
    token_t = torch.tensor(ids_list, dtype=torch.long)
    pad_mask_t = torch.tensor(pad_list, dtype=torch.bool)
    return token_t, pad_mask_t


class _SafeEncoder(nn.Module if TORCH_AVAILABLE else object):
    """SAFE 风格：token embedding + mean 聚合。"""

    def __init__(self, vocab_size: int, embed_dim: int = _EMBED_DIM, output_dim: int = _OUTPUT_DIM):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for SAFE baseline")
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


def safe_save_model(
    model: "_SafeEncoder",
    vocab: Dict[str, int],
    path: str,
    embed_dim: int = _EMBED_DIM,
    output_dim: int = _OUTPUT_DIM,
) -> None:
    """保存 SAFE 模型与 vocab 至单一 .pt 文件，供 embed_batch_safe 加载。"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for safe_save_model")
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "embed_dim": embed_dim,
            "output_dim": output_dim,
        },
        path,
    )


def safe_load_model(
    path: str,
) -> Tuple["_SafeEncoder", Dict[str, int]]:
    """加载 SAFE 模型与 vocab，返回 (model, vocab)。"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for safe_load_model")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError(f"无效的 SAFE 模型格式: {path}")
    state = data.get("state_dict")
    vocab = data.get("vocab")
    embed_dim = data.get("embed_dim", _EMBED_DIM)
    output_dim = data.get("output_dim", _OUTPUT_DIM)
    if state is None or vocab is None:
        raise ValueError(f"SAFE 模型缺少 state_dict 或 vocab: {path}")
    vocab_size = max(len(vocab), 256)
    model = _SafeEncoder(vocab_size=vocab_size, embed_dim=embed_dim, output_dim=output_dim)
    model.load_state_dict(state)
    model.eval()
    return model, vocab


class SafeEmbedder:
    """
    SAFE 推理器：一次加载模型与 vocab，支持批量嵌入。

    说明：粗筛阶段频繁调用时，使用此类避免反复从磁盘加载模型。
    """

    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        device: Optional["object"] = None,
        prefer_cuda: bool = True,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for SafeEmbedder")
        self._max_len = 512
        self._device = self._resolve_device(device, prefer_cuda=prefer_cuda)

        if model_path and os.path.isfile(model_path):
            model, vocab = safe_load_model(model_path)
            self._model = model.to(self._device)
            self._vocab = vocab
        else:
            # 未提供 model_path 时，调用方应避免在热路径中使用（因为 vocab 难以一致）
            self._model = None
            self._vocab = {"[PAD]": 0, "[UNK]": 1}

    @staticmethod
    def _resolve_device(device: Optional["object"], *, prefer_cuda: bool) -> "torch.device":
        if device is not None:
            return torch.device(device)
        if prefer_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def embed_many(
        self,
        multimodals: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 256,
    ) -> List[List[float]]:
        if not multimodals:
            return []
        if self._model is None:
            # 无权重时退化为零向量（避免在验证/评估阶段产生随机噪声）
            return [[0.0] * _OUTPUT_DIM for _ in multimodals]

        out: List[List[float]] = []
        self._model.eval()
        with torch.no_grad():
            for i in range(0, len(multimodals), batch_size):
                chunk = multimodals[i : i + batch_size]
                token_t, pad_mask_t = safe_tokenize_many(
                    chunk, self._vocab, max_len=self._max_len
                )
                token_t = token_t.to(self._device)
                pad_mask_t = pad_mask_t.to(self._device)
                vec = self._model(token_t, pad_mask_t)
                out.extend(vec.detach().cpu().tolist())
        return out


def embed_batch_safe(
    features: Dict[str, Any],
    model_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    SAFE 风格嵌入：仅使用序列 token，输出与 embed_batch 兼容的 List[{name, vector}]。
    model_path: 训练权重路径，指定时加载 saved model+vocab；否则从 features 收集 vocab 并随机初始化。
    """
    funcs = features.get("functions") or []
    if not funcs:
        return []

    if not TORCH_AVAILABLE:
        return _fallback_embeddings(funcs)

    if model_path and os.path.isfile(model_path):
        model, vocab = safe_load_model(model_path)
    else:
        vocab = _collect_vocab(features)
        vocab_size = max(len(vocab), 256)
        model = _SafeEncoder(vocab_size=vocab_size)
        model.eval()

    out: List[Dict[str, Any]] = []
    max_len = 512
    with torch.no_grad():
        for item in funcs:
            name = item.get("name", "")
            feats = item.get("features") or {}
            mm = feats.get("multimodal") or {}
            ids, pad_mask_list = safe_tokenize(mm, vocab, max_len=max_len)
            token_t = torch.tensor([ids], dtype=torch.long)
            pad_mask_t = torch.tensor([pad_mask_list], dtype=torch.bool)
            vec = model(token_t, pad_mask_t)
            out.append({"name": name, "vector": vec.squeeze(0).tolist()})
    return out


def _fallback_embeddings(funcs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """无 PyTorch 时的回退：零向量。"""
    return [{"name": f.get("name", ""), "vector": [0.0] * _OUTPUT_DIM} for f in funcs]
