"""两阶段流水线精排模块：候选特征查找与多模态精排得分。"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .similarity import cosine_similarity

logger = logging.getLogger(__name__)


def _collect_vocab_from_multimodals(
    query_multimodal: dict,
    candidate_multimodals: List[dict],
) -> Dict[str, int]:
    """从 query 与所有 candidate 的 multimodal 特征收集 pcode token 构建 vocab。"""
    from features.models.multimodal_fusion import get_default_vocab

    vocab = get_default_vocab()
    all_mm = [query_multimodal] + [m for _, m in candidate_multimodals if m]

    for mm in all_mm:
        if not mm:
            continue
        seq = mm.get("sequence") or {}
        for t in seq.get("pcode_tokens") or []:
            if t and t not in vocab:
                vocab[t] = len(vocab)
        graph = mm.get("graph") or {}
        for nf in graph.get("node_features") or []:
            opcodes = nf if isinstance(nf, list) else nf.get("pcode_opcodes", []) or []
            for op in opcodes:
                if op and op not in vocab:
                    vocab[op] = len(vocab)
    return vocab


def load_candidate_features(
    candidate_ids: List[str],
    library_features_path: str,
) -> List[Tuple[str, dict]]:
    """
    从 library_features.json 按 function_id 加载 multimodal 特征。

    library_features.json 格式：{function_id: multimodal_dict}
    缺失的 id 会跳过并记录 warning，不静默失败。

    返回: [(candidate_id, multimodal_dict), ...]，仅包含存在的 id。
    """
    if not os.path.isfile(library_features_path):
        raise FileNotFoundError(f"库特征文件不存在: {library_features_path}")

    with open(library_features_path, encoding="utf-8") as f:
        lib_features = json.load(f)

    if not isinstance(lib_features, dict):
        raise ValueError(
            f"library_features 格式应为 {{function_id: multimodal}}，"
            f"实际得到 {type(lib_features).__name__}"
        )

    result: List[Tuple[str, dict]] = []
    for cid in candidate_ids:
        if cid not in lib_features:
            logger.warning("候选 function_id 不在库中，已跳过: %s", cid)
            continue
        mm = lib_features[cid]
        if not mm or not isinstance(mm, dict):
            logger.warning("候选 %s 的特征无效，已跳过", cid)
            continue
        result.append((cid, mm))
    return result


def load_candidate_features_from_dict(
    candidate_ids: Sequence[str],
    lib_features: Any,
) -> List[Tuple[str, dict]]:
    """从已加载的库特征 dict（或 dict-like 映射）中按 id 获取 multimodal 特征。"""
    if not hasattr(lib_features, "get"):
        raise ValueError("lib_features 必须支持 .get() (dict-like)")
    result: List[Tuple[str, dict]] = []
    for cid in candidate_ids:
        mm = lib_features.get(cid)
        if not mm or not isinstance(mm, dict):
            continue
        result.append((str(cid), mm))
    return result


class RerankModel:
    """
    精排模型封装：一次加载 MultiModalFusionModel，支持对候选批量 forward。
    use_dfg_model: None 时按检查点 meta / 权重键推断；True/False 强制构造对应结构。
    """

    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        device: Optional["object"] = None,
        prefer_cuda: bool = True,
        use_dfg_model: Optional[bool] = None,
    ) -> None:
        try:
            import torch
            from features.inference import _resolve_model_path, resolve_inference_device
            from features.models.multimodal_fusion import (
                MultiModalFusionModel,
                get_default_vocab,
                infer_use_dfg_from_state_dict,
                parse_multimodal_checkpoint,
            )
        except ImportError as e:
            raise RuntimeError("精排需要 PyTorch 与 features 模块") from e

        self._torch = torch
        self._device = resolve_inference_device(device, prefer_cuda=prefer_cuda)
        self._vocab = get_default_vocab()
        self._vocab_size = max(len(self._vocab), 256)
        self._max_seq_len = 512
        self._max_graph_nodes = 128
        self._max_dfg_nodes = 128

        resolved = _resolve_model_path(model_path)
        state_dict: Dict[str, Any] = {}
        meta: Dict[str, Any] = {}
        if resolved:
            raw = torch.load(resolved, map_location=self._device, weights_only=True)
            state_dict, meta = parse_multimodal_checkpoint(raw)
            if meta.get("pcode_vocab_size"):
                self._vocab_size = max(int(meta["pcode_vocab_size"]), self._vocab_size)
            if meta.get("max_graph_nodes"):
                self._max_graph_nodes = max(1, int(meta["max_graph_nodes"]))
            if meta.get("max_dfg_nodes"):
                self._max_dfg_nodes = max(1, int(meta["max_dfg_nodes"]))
            if meta.get("max_seq_len"):
                self._max_seq_len = max(1, int(meta["max_seq_len"]))

        use_dfg = False
        if use_dfg_model is True:
            use_dfg = True
        elif use_dfg_model is False:
            use_dfg = False
        elif "use_dfg" in meta:
            use_dfg = bool(meta.get("use_dfg"))
        elif infer_use_dfg_from_state_dict(state_dict):
            use_dfg = True

        self._model = MultiModalFusionModel(
            pcode_vocab_size=self._vocab_size,
            use_dfg=use_dfg,
        ).to(self._device)
        if state_dict:
            missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.debug("精排 load_state_dict missing keys: %s", list(missing)[:8])
            if unexpected:
                logger.debug("精排 load_state_dict unexpected keys: %s", list(unexpected)[:8])
        self._model.eval()

    def _tensorize_many(self, multimodals: Sequence[dict]) -> Tuple[
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
    ]:
        """
        批量 tensorize：返回 (token_ids, jump_mask, node_ids, padding_mask, dfg_node_ids, dfg_edge_index)。
        CFG/DFG 的 edge_index 当前 GNN 实现中未用消息传递，传空或占位。
        """
        torch = self._torch
        B = len(multimodals)
        L = self._max_seq_len
        N = self._max_graph_nodes
        Nd = self._max_dfg_nodes
        token_ids = torch.zeros((B, L), dtype=torch.long, device=self._device)
        jump_mask = torch.zeros((B, L), dtype=torch.long, device=self._device)
        pad_mask = torch.ones((B, L), dtype=torch.bool, device=self._device)
        node_ids = torch.zeros((B, N), dtype=torch.long, device=self._device)
        dfg_nodes = torch.zeros((B, Nd), dtype=torch.long, device=self._device)

        for i, mm in enumerate(multimodals):
            seq = (mm or {}).get("sequence") or {}
            tokens = seq.get("pcode_tokens") or []
            jumps = seq.get("jump_mask") or []
            t = [self._vocab.get(x, 1) for x in tokens[:L]]
            j = list(jumps[:L])
            llen = len(t)
            if llen > 0:
                token_ids[i, :llen] = torch.tensor(t, dtype=torch.long, device=self._device)
                jump_mask[i, :llen] = torch.tensor(
                    j + [0] * max(0, llen - len(j)), dtype=torch.long, device=self._device
                )
                pad_mask[i, :llen] = False

            graph = (mm or {}).get("graph") or {}
            nfs = graph.get("node_features") or []
            nf_ids: List[int] = []
            for nf in nfs[:N]:
                opcodes = nf if isinstance(nf, list) else (nf.get("pcode_opcodes") or [])
                nf_ids.append(self._vocab.get(opcodes[0], 1) if opcodes else 0)
            if nf_ids:
                node_ids[i, : len(nf_ids)] = torch.tensor(
                    nf_ids, dtype=torch.long, device=self._device
                )

            dfg = (mm or {}).get("dfg") or {}
            dfg_nf = dfg.get("node_features") or []
            did: List[int] = []
            for x in dfg_nf[:Nd]:
                if isinstance(x, int):
                    did.append(int(x) % 512)
            if did:
                dfg_nodes[i, : len(did)] = torch.tensor(did, dtype=torch.long, device=self._device)

        # 空 pcode 序列时 padding_mask 全 True，Transformer 嵌套张量会触发 to_padded_tensor 报错
        for i in range(B):
            if bool(pad_mask[i].all()):
                token_ids[i, 0] = 1
                jump_mask[i, 0] = 0
                pad_mask[i, 0] = False

        dfg_edge_t = torch.zeros((2, 0), dtype=torch.long, device=self._device)
        return token_ids, jump_mask, node_ids, pad_mask, dfg_nodes, dfg_edge_t

    def _embed_many(self, multimodals: Sequence[dict], *, batch_size: int) -> List[List[float]]:
        torch = self._torch
        out: List[List[float]] = []
        self._model.eval()
        with torch.no_grad():
            for i in range(0, len(multimodals), batch_size):
                chunk = multimodals[i : i + batch_size]
                tt, jt, nt, pm, dnt, det = self._tensorize_many(chunk)
                edge_t = torch.zeros((2, 0), dtype=torch.long, device=self._device)
                vec = self._model(
                    tt,
                    jt,
                    nt,
                    edge_t,
                    padding_mask=pm,
                    dfg_node_features=dnt,
                    dfg_edge_index=det,
                )
                if vec.dim() == 1:
                    vec = vec.unsqueeze(0)
                out.extend(vec.detach().cpu().tolist())
        return out

    def score(
        self,
        query_multimodal: dict,
        candidate_features_list: List[Tuple[str, dict]],
        *,
        batch_size: int = 1024,
    ) -> List[Tuple[str, float]]:
        """对候选批量打分并按 score 降序返回。"""
        if not candidate_features_list:
            return []

        cand_ids = [cid for cid, _ in candidate_features_list]
        cand_mms = [mm for _, mm in candidate_features_list]

        try:
            q_vec = self._embed_many([query_multimodal], batch_size=1)[0]
        except Exception as e:
            logger.warning("查询特征 forward 失败: %s", e)
            return [(cid, 0.0) for cid in cand_ids]

        try:
            cand_vecs = self._embed_many(cand_mms, batch_size=batch_size)
        except Exception as e:
            logger.warning("候选特征 forward 失败: %s", e)
            cand_vecs = [[0.0] * len(q_vec) for _ in cand_ids]

        scores = [(cid, cosine_similarity(q_vec, v)) for cid, v in zip(cand_ids, cand_vecs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


def compute_rerank_scores(
    query_features: dict,
    candidate_features_list: List[Tuple[str, dict]],
    model_path: Optional[str] = None,
    *,
    device: Optional["object"] = None,
    prefer_cuda: bool = True,
    use_dfg_model: Optional[bool] = None,
) -> List[Tuple[str, float]]:
    """
    .. deprecated:: 0.2
        请使用 :class:`RerankModel` 的 :meth:`~RerankModel.score` 方法代替。

    批量精排：对每个候选计算与查询的余弦相似度得分。

    query_features: 单查询的 multimodal 特征 {graph, sequence}
    candidate_features_list: [(candidate_id, multimodal_dict), ...]
    model_path: 与 embed_batch 相同优先级（参数 > SEMPATCH_MODEL_PATH > 默认未训练）

    返回: [(candidate_id, score), ...]，按 score 降序排列。
    """
    import warnings

    warnings.warn(
        "compute_rerank_scores 已弃用，请使用 RerankModel.score()",
        DeprecationWarning,
        stacklevel=2,
    )
    if not candidate_features_list:
        return []

    try:
        import torch
        from features.inference import (
            _resolve_model_path,
            resolve_inference_device,
            run_with_cuda_oom_fallback,
        )
        from features.models.multimodal_fusion import (
            MultiModalFusionModel,
            _tensorize_multimodal,
            infer_use_dfg_from_state_dict,
            parse_multimodal_checkpoint,
        )
    except ImportError:
        raise RuntimeError("精排需要 PyTorch 与 features 模块")

    chosen_device = resolve_inference_device(device, prefer_cuda=prefer_cuda)

    def _compute_on_device(dev: "object") -> List[Tuple[str, float]]:
        vocab = _collect_vocab_from_multimodals(query_features, candidate_features_list)
        vocab_size = max(len(vocab), 256)
        resolved = _resolve_model_path(model_path)
        state_dict: Dict[str, Any] = {}
        meta: Dict[str, Any] = {}
        if resolved:
            raw = torch.load(resolved, map_location=dev, weights_only=True)
            state_dict, meta = parse_multimodal_checkpoint(raw)
            if meta.get("pcode_vocab_size"):
                vocab_size = max(vocab_size, int(meta["pcode_vocab_size"]))

        use_dfg = False
        if use_dfg_model is True:
            use_dfg = True
        elif use_dfg_model is False:
            use_dfg = False
        elif "use_dfg" in meta:
            use_dfg = bool(meta.get("use_dfg"))
        elif infer_use_dfg_from_state_dict(state_dict):
            use_dfg = True

        max_seq = max(512, int(meta.get("max_seq_len", 512) or 512))
        max_gn = max(128, int(meta.get("max_graph_nodes", 128) or 128))
        max_dn = max(128, int(meta.get("max_dfg_nodes", 128) or 128))

        model = MultiModalFusionModel(pcode_vocab_size=vocab_size, use_dfg=use_dfg).to(dev)
        if state_dict:
            model.load_state_dict(state_dict, strict=False)
        model.eval()

        scores: List[Tuple[str, float]] = []
        with torch.no_grad():
            try:
                qt, qj, qn, qe, qp, qdn, qde = _tensorize_multimodal(
                    query_features,
                    vocab,
                    device=dev,
                    max_seq_len=max_seq,
                    max_graph_nodes=max_gn,
                    max_dfg_nodes=max_dn,
                )
                query_vec = model(
                    qt,
                    qj,
                    qn,
                    qe,
                    padding_mask=qp,
                    dfg_node_features=qdn,
                    dfg_edge_index=qde,
                )
                query_list = query_vec.detach().cpu().numpy().tolist()
            except Exception as e:
                logger.warning("查询特征 tensorize/forward 失败: %s", e)
                return [(cid, 0.0) for cid, _ in candidate_features_list]

            for cid, cand_mm in candidate_features_list:
                try:
                    ct, cj, cn, ce, cp, cdn, cde = _tensorize_multimodal(
                        cand_mm,
                        vocab,
                        device=dev,
                        max_seq_len=max_seq,
                        max_graph_nodes=max_gn,
                        max_dfg_nodes=max_dn,
                    )
                    cand_vec = model(
                        ct,
                        cj,
                        cn,
                        ce,
                        padding_mask=cp,
                        dfg_node_features=cdn,
                        dfg_edge_index=cde,
                    )
                    cand_list = cand_vec.detach().cpu().numpy().tolist()
                    score = cosine_similarity(query_list, cand_list)
                except Exception:
                    score = 0.0
                scores.append((cid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    return run_with_cuda_oom_fallback(
        _compute_on_device,
        chosen_device,
        context=f"compute_rerank_scores(candidates={len(candidate_features_list)})",
    )
