"""
SAFE 风格基线：轻量序列编码器（token embedding + 聚合），用于与 SemPatch 对比。

SAFE 原文：汇编指令 → Instruction2Vec → 自注意力 RNN 聚合 → 函数嵌入。
本实现采用 P-code token embedding + mean 聚合，输入与 embed_batch 兼容的 FeaturesDict。
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

_OUTPUT_DIM = 128
_EMBED_DIM = 64
_log = logging.getLogger(__name__)


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
    使用流式 JSON 解析，不将整个文件加载到内存，避免大库 OOM。
    """
    from utils.precomputed_multimodal_io import is_jsonl_sidecar_path, iter_jsonl_sidecar

    vocab: Dict[str, int] = {"[PAD]": 0, "[UNK]": 1}
    n = 0
    t0 = time.monotonic()
    LOG_EVERY = 20000

    if is_jsonl_sidecar_path(features_path):
        _log.info("正在从 JSONL 流式构建词表: %s", features_path)
        for _fid, mm in iter_jsonl_sidecar(features_path):
            if isinstance(mm, dict):
                _collect_vocab_from_multimodal(mm, vocab)
            n += 1
            if n % LOG_EVERY == 0:
                elapsed = time.monotonic() - t0
                _log.info("  词表扫描: %d 条, 词表 %d tokens (%.0fs)", n, len(vocab), elapsed)
        elapsed = time.monotonic() - t0
        _log.info("词表构建完成: %d 条扫描, %d tokens (%.1fs)", n, len(vocab), elapsed)
        return vocab

    # 流式解析 JSON 对象，避免 json.load OOM
    try:
        from scripts.sidechain.build_embeddings_db import _iter_json_object_records
    except ImportError:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "..", "scripts", "sidechain"))
        from build_embeddings_db import _iter_json_object_records  # type: ignore[no-redef]

    file_size_mb = os.path.getsize(features_path) / (1024 * 1024)
    _log.info("正在从 JSON 流式构建词表: %s (%.1f MB)", features_path, file_size_mb)
    with open(features_path, "rb") as fp:
        for _fid, mm in _iter_json_object_records(fp):
            if isinstance(mm, dict):
                _collect_vocab_from_multimodal(mm, vocab)
            n += 1
            if n % LOG_EVERY == 0:
                elapsed = time.monotonic() - t0
                speed = n / elapsed if elapsed > 0 else 0
                _log.info(
                    "  词表扫描: %d 条, 词表 %d tokens (%.0f 条/s, %.0fs)",
                    n, len(vocab), speed, elapsed,
                )
    elapsed = time.monotonic() - t0
    _log.info("词表构建完成: %d 条扫描, %d tokens (%.1fs)", n, len(vocab), elapsed)
    return vocab


def collect_vocab_from_features_jsonl(features_path: str) -> Dict[str, int]:
    """
    从 JSONL 侧车（每行 {"function_id","multimodal"}）流式扫描构建 vocab。
    不保留整库 multimodal，峰值内存远低于对 library_features.json 的 json.load。
    """
    vocab: Dict[str, int] = {"[PAD]": 0, "[UNK]": 1}
    from utils.precomputed_multimodal_io import iter_jsonl_sidecar

    n = 0
    t0 = time.monotonic()
    LOG_EVERY = 20000
    _log.info("正在从 JSONL 流式构建词表: %s", features_path)
    for _fid, mm in iter_jsonl_sidecar(features_path):
        if isinstance(mm, dict):
            _collect_vocab_from_multimodal(mm, vocab)
        n += 1
        if n % LOG_EVERY == 0:
            elapsed = time.monotonic() - t0
            speed = n / elapsed if elapsed > 0 else 0
            _log.info(
                "  词表扫描: %d 条, 词表 %d tokens (%.0f 条/s, %.0fs)",
                n, len(vocab), speed, elapsed,
            )
    elapsed = time.monotonic() - t0
    _log.info("词表构建完成: %d 条扫描, %d tokens (%.1fs)", n, len(vocab), elapsed)
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

    def forward(
        self, token_ids: "torch.Tensor", pad_mask: "torch.Tensor | None" = None
    ) -> "torch.Tensor":
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
            import logging

            logging.getLogger(__name__).warning(
                "SafeEmbedder: model_path 未提供或文件不存在，embed_many 将返回零向量"
            )
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
            return [[0.0] * _OUTPUT_DIM for _ in multimodals]

        out: List[List[float]] = []
        self._model.eval()
        with torch.no_grad():
            for i in range(0, len(multimodals), batch_size):
                chunk = multimodals[i : i + batch_size]
                token_t, pad_mask_t = safe_tokenize_many(chunk, self._vocab, max_len=self._max_len)
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

    names = [item.get("name", "") for item in funcs]
    multimodals = [(item.get("features") or {}).get("multimodal") or {} for item in funcs]

    with torch.no_grad():
        chunk_size = 256
        total_chunks = (len(multimodals) + chunk_size - 1) // chunk_size
        LOG_CHUNKS_EVERY = max(1, total_chunks // 10)
        t0 = time.monotonic()
        if total_chunks > 1:
            _log.info(
                "SAFE 嵌入: %d 条记录, %d 批 (batch=%d)",
                len(multimodals), total_chunks, chunk_size,
            )
        for ci, start in enumerate(range(0, len(multimodals), chunk_size)):
            chunk_mm = multimodals[start : start + chunk_size]
            chunk_names = names[start : start + chunk_size]
            token_t, pad_mask_t = safe_tokenize_many(chunk_mm, vocab, max_len=max_len)
            vecs = model(token_t, pad_mask_t)  # (B, output_dim)
            for j, name in enumerate(chunk_names):
                out.append({"name": name, "vector": vecs[j].tolist()})
            if total_chunks > 1 and (ci + 1) % LOG_CHUNKS_EVERY == 0:
                elapsed = time.monotonic() - t0
                done = min(start + chunk_size, len(multimodals))
                speed = done / elapsed if elapsed > 0 else 0
                eta = (len(multimodals) - done) / speed if speed > 0 else 0
                _log.info(
                    "  SAFE 批 %d/%d: %d/%d (%.0f 条/s, ETA %.0fs)",
                    ci + 1, total_chunks, done, len(multimodals), speed, eta,
                )
    return out


def _fallback_embeddings(funcs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """无 PyTorch 时的回退：零向量。"""
    return [{"name": f.get("name", ""), "vector": [0.0] * _OUTPUT_DIM} for f in funcs]
