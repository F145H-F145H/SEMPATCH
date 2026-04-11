"""
嵌入推断接口。接入 MultiModalFusionModel（5.1），支持批量嵌入。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _project_root() -> str:
    """src/features/inference.py → 项目根目录。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_inference_device(device: Optional[Any] = None, *, prefer_cuda: bool = True) -> Any:
    """解析推理设备：显式 device 优先，否则 cuda（若可用且 prefer_cuda）否则 cpu。"""
    import torch

    if device is not None:
        return torch.device(device) if isinstance(device, str) else device
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_model_path(model_path: Optional[str]) -> Optional[str]:
    """
    解析可加载的权重路径：参数 > SEMPATCH_MODEL_PATH > 存在的 output/best_model.pth。
    均不存在则返回 None（使用随机初始化权重）。
    """
    if model_path and os.path.isfile(model_path):
        return os.path.abspath(model_path)
    env = os.environ.get("SEMPATCH_MODEL_PATH", "").strip()
    if env and os.path.isfile(env):
        return os.path.abspath(env)
    default = os.path.join(_project_root(), "output", "best_model.pth")
    if os.path.isfile(default):
        return os.path.abspath(default)
    return None


def run_with_cuda_oom_fallback(
    compute_fn: Callable[[Any], T],
    chosen_device: Any,
    *,
    context: str = "",
) -> T:
    """在 CUDA OOM 时清空缓存并重试 CPU（精排等大批量场景）。"""
    import torch

    from exceptions import EmbeddingError

    try:
        return compute_fn(chosen_device)
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" not in msg:
            raise
        dev_type = getattr(chosen_device, "type", None)
        if dev_type != "cuda" and str(chosen_device) != "cuda":
            raise
        logger.warning("CUDA OOM%s，回退 CPU", f" ({context})" if context else "")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            return compute_fn(torch.device("cpu"))
        except RuntimeError as cpu_e:
            raise EmbeddingError(
                f"CUDA OOM 后 CPU 回退也失败{' (' + context + ')' if context else ''}: {cpu_e}"
            ) from cpu_e


def _collect_vocab_from_features(features: Dict[str, Any]) -> Dict[str, int]:
    """从 features 中收集所有 pcode token 构建 vocab。"""
    from features.models.multimodal_fusion import get_default_vocab

    vocab = get_default_vocab()
    funcs = features.get("functions") or []
    for item in funcs:
        feats = item.get("features") or {}
        mm = feats.get("multimodal") or {}
        seq = mm.get("sequence") or {}
        tokens = seq.get("pcode_tokens") or []
        for t in tokens:
            if t and t not in vocab:
                vocab[t] = len(vocab)
        graph = mm.get("graph") or {}
        nf = graph.get("node_features") or []
        for n in nf:
            opcodes = n if isinstance(n, list) else n.get("pcode_opcodes", []) or []
            for op in opcodes:
                if op and op not in vocab:
                    vocab[op] = len(vocab)
    return vocab


def embed_batch(
    features: Dict[str, Any],
    *,
    model_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    将 features 转为向量嵌入。
    若 PyTorch 可用且特征含 multimodal，使用 MultiModalFusionModel；
    否则回退到简单基线（随机/零向量）。

    model_path: 可选，训练得到的 MultiModalFusionModel state_dict；与 _resolve_model_path 联用。
    """
    funcs = features.get("functions") or []
    if not funcs:
        return []

    try:
        import torch
        from features.models.multimodal_fusion import (
            MultiModalFusionModel,
            get_default_vocab,
            infer_use_dfg_from_state_dict,
            parse_multimodal_checkpoint,
            tensorize_multimodal_many,
        )

        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False

    has_multimodal = any((item.get("features") or {}).get("multimodal") for item in funcs)
    if not TORCH_AVAILABLE or not has_multimodal:
        return _embed_baseline(features)

    from features.models.multimodal_fusion import tensorize_multimodal_many

    vocab = _collect_vocab_from_features(features)
    vocab_size = max(len(vocab), 256)
    resolved = _resolve_model_path(model_path)
    use_dfg = False
    state_dict: Dict[str, Any] = {}
    if resolved:
        try:
            raw = torch.load(resolved, map_location="cpu", weights_only=True)
            state_dict, meta = parse_multimodal_checkpoint(raw)
            if "use_dfg" in meta:
                use_dfg = bool(meta.get("use_dfg"))
            elif infer_use_dfg_from_state_dict(state_dict):
                use_dfg = True
        except Exception as e:
            logger.warning("MultiModalFusion 权重解析失败，使用随机初始化: %s", e)
            state_dict = {}
    model = MultiModalFusionModel(pcode_vocab_size=vocab_size, use_dfg=use_dfg)
    if state_dict:
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.warning("MultiModalFusion 权重加载失败，使用当前 batch 对应随机初始化: %s", e)
    model.eval()

    # Separate functions with multimodal features from those without
    names = [item.get("name", "") for item in funcs]
    multimodals_list = [(item.get("features") or {}).get("multimodal") for item in funcs]

    embeddings: List[Dict[str, Any]] = [{}] * len(funcs)  # placeholder
    zero_vec = [0.0] * 128

    # Indices that have multimodal features
    valid_indices = [i for i, mm in enumerate(multimodals_list) if mm]
    if not valid_indices:
        return [{"name": n, "vector": zero_vec} for n in names]

    with torch.no_grad():
        # Process in chunks to bound memory
        chunk_size = 256
        for start in range(0, len(valid_indices), chunk_size):
            chunk_idx = valid_indices[start : start + chunk_size]
            chunk_mm = [multimodals_list[i] for i in chunk_idx]
            try:
                batched = tensorize_multimodal_many(
                    chunk_mm, vocab, device=None, max_seq_len=512, max_graph_nodes=128, max_dfg_nodes=128
                )
                token_t, jump_t, node_t, edge_t, pad_mask, dfg_nt, dfg_et = batched
                vecs = model(
                    token_t,
                    jump_t,
                    node_t,
                    edge_t,
                    padding_mask=pad_mask,
                    dfg_node_features=dfg_nt,
                    dfg_edge_index=dfg_et,
                )  # (B_chunk, output_dim)
                for j, orig_i in enumerate(chunk_idx):
                    embeddings[orig_i] = {"name": names[orig_i], "vector": vecs[j].numpy().tolist()}
            except Exception:
                for orig_i in chunk_idx:
                    embeddings[orig_i] = {"name": names[orig_i], "vector": zero_vec}

    # Fill any remaining slots (functions without multimodal)
    for i in range(len(funcs)):
        if not embeddings[i]:
            embeddings[i] = {"name": names[i], "vector": zero_vec}

    return embeddings


def _embed_baseline(features: Dict[str, Any]) -> List[Dict[str, Any]]:
    """无 PyTorch 或多模态时的基线：基于 pcode 序列或 ACFG 的确定性哈希向量。"""
    import hashlib

    funcs = features.get("functions") or []
    out = []
    for item in funcs:
        name = item.get("name", "")
        feats = item.get("features") or item.get("acfg") or {}
        tokens = []
        if isinstance(feats, dict):
            mm = feats.get("multimodal") or {}
            seq = mm.get("sequence") or {}
            tokens = seq.get("pcode_tokens") or []
            if not tokens:
                nf = (
                    feats.get("node_features")
                    or (feats.get("graph") or {}).get("node_features")
                    or []
                )
                for n in nf:
                    opcodes = n if isinstance(n, list) else n.get("pcode_opcodes", []) or []
                    tokens.extend(opcodes)
        token_str = "|".join(str(t) for t in tokens)
        if not token_str:
            out.append({"name": name, "vector": [0.0] * 128})
            continue
        digest = hashlib.md5(token_str.encode("utf-8")).digest()
        vec = [0.0] * 128
        for i in range(128):
            byte_idx = i // 8
            bit_idx = i % 8
            vec[i] = ((digest[byte_idx] >> bit_idx) & 1) * 1.0 - 0.5
        out.append({"name": name, "vector": vec})
    return out
