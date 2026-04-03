"""
两阶段「粗筛-精排」流水线：独立可调用，不依赖 DAG。

整合 LibraryFaissIndex（粗筛）、compute_rerank_scores（精排），
对外提供 retrieve、rerank、retrieve_and_rerank 接口。
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .faiss_library import LibraryFaissIndex, retrieve_coarse, retrieve_coarse_many
from .rerank import RerankModel, load_candidate_features_from_dict


def _default_rerank_model_path() -> str:
    """项目根目录下的 output/best_model.pth。"""
    # src/matcher/two_stage.py -> 上三级 -> 项目根
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "output", "best_model.pth")


class TwoStagePipeline:
    """
    两阶段流水线：SAFE 粗筛 + 多模态精排。

    构造函数注入所有依赖路径，内部构建 LibraryFaissIndex。
    不依赖 DAG 或 ctx，可单独测试与评估。
    """

    def __init__(
        self,
        library_safe_embeddings_path: str,
        library_features_path: str,
        query_features_path: str,
        coarse_k: int = 100,
        rerank_model_path: Optional[str] = None,
        safe_model_path: Optional[str] = None,
        rerank_device: Optional[object] = None,
        prefer_cuda: bool = True,
        rerank_use_dfg: Optional[bool] = None,
    ) -> None:
        """
        Args:
            library_safe_embeddings_path: 库函数 SAFE 嵌入 JSON 路径
            library_features_path: 库函数 multimodal 特征 JSON 路径（按 function_id 索引）
            query_features_path: 查询函数 multimodal 特征 JSON 路径
            coarse_k: 粗筛返回的候选数量
            rerank_model_path: 精排模型权重路径，None 时使用 output/best_model.pth
            safe_model_path: 训练后的 SAFE 权重路径，粗筛查询时使用，与库嵌入一致
            rerank_device: 精排推理设备（如 \"cuda\" / \"cpu\" / torch.device），None 时按 prefer_cuda 自动选择
            prefer_cuda: 在未显式指定 rerank_device 时，是否优先使用 CUDA
            rerank_use_dfg: None=按检查点推断；True/False=强制启用或禁用 DFG 精排分支
        """
        self._library_embeddings_path = library_safe_embeddings_path
        self._library_features_path = library_features_path
        self._query_features_path = query_features_path
        self._coarse_k = coarse_k
        self._rerank_model_path = rerank_model_path or _default_rerank_model_path()
        self._safe_model_path = safe_model_path
        self._rerank_device = rerank_device
        self._prefer_cuda = prefer_cuda

        self._faiss_index = LibraryFaissIndex(library_safe_embeddings_path)

        # query features 常驻内存
        with open(query_features_path, encoding="utf-8") as f:
            self._query_features: dict = json.load(f)
        if not isinstance(self._query_features, dict):
            raise ValueError(
                f"query_features 格式应为 {{function_id: multimodal}}，"
                f"实际得到 {type(self._query_features).__name__}"
            )

        # library features 常驻内存（避免每个 query 读盘）
        with open(library_features_path, encoding="utf-8") as f:
            self._library_features: dict = json.load(f)
        if not isinstance(self._library_features, dict):
            raise ValueError(
                f"library_features 格式应为 {{function_id: multimodal}}，"
                f"实际得到 {type(self._library_features).__name__}"
            )

        # 模型缓存：SAFE 粗筛 embedder + 精排模型
        from features.baselines.safe import SafeEmbedder

        self._safe_embedder = SafeEmbedder(
            model_path=self._safe_model_path,
            device="cuda" if prefer_cuda else "cpu",
            prefer_cuda=prefer_cuda,
        )
        self._rerank_model = RerankModel(
            model_path=self._rerank_model_path,
            device=self._rerank_device,
            prefer_cuda=self._prefer_cuda,
            use_dfg_model=rerank_use_dfg,
        )

    def retrieve(self, query_func_id: str) -> List[str]:
        """
        粗筛：从 query_features 加载特征，返回 Top-K 候选 function_id 列表。
        """
        if query_func_id not in self._query_features:
            raise KeyError(f"查询 function_id 不存在: {query_func_id}")
        mm = self._query_features[query_func_id]
        return retrieve_coarse(
            mm, self._faiss_index, k=self._coarse_k,
            safe_model_path=self._safe_model_path,
        )

    def rerank(
        self, query_func_id: str, candidate_ids: List[str]
    ) -> List[Tuple[str, float]]:
        """
        精排：加载 query 与 candidate 特征，返回按得分降序的 [(candidate_id, score), ...]。
        空候选返回空列表。
        """
        if query_func_id not in self._query_features:
            raise KeyError(f"查询 function_id 不存在: {query_func_id}")
        if not candidate_ids:
            return []
        query_mm = self._query_features[query_func_id]
        cand_features = load_candidate_features_from_dict(
            candidate_ids, self._library_features
        )
        return self._rerank_model.score(query_mm, cand_features)

    def retrieve_and_rerank(self, query_func_id: str) -> List[Tuple[str, float]]:
        """
        粗筛 + 精排：先 retrieve 得到 Top-K 候选，再 rerank 精排返回完整列表。
        """
        candidates = self.retrieve(query_func_id)
        return self.rerank(query_func_id, candidates)

    def evaluate(
        self,
        valid_ids: Sequence[str],
        ground_truth: Dict[str, Sequence[str]],
        *,
        batch_size: int = 128,
        rerank_batch_size: int = 1024,
        subsample: Optional[int] = None,
        rerank_k: Optional[int] = None,
        seed: int = 42,
        progress_every: int = 10,
    ) -> Tuple[float, float]:
        """
        批量两阶段评估，返回 (coarse_recall, recall_at_1)。

        - subsample: 可选，随机抽样 N 个 query 做近似验证（训练期加速）。
        - rerank_k: 可选，仅对前 rerank_k 个 coarse 候选做精排（≤coarse_k）。
        """
        ids = [qid for qid in valid_ids if qid in self._query_features and qid in ground_truth]
        if not ids:
            return 0.0, 0.0
        if subsample is not None and subsample > 0 and subsample < len(ids):
            rnd = random.Random(seed)
            ids = rnd.sample(ids, subsample)

        coarse_hits = 0
        r1_hits = 0
        n_total = 0

        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            q_multis = [self._query_features[qid] for qid in batch_ids]
            coarse_lists = retrieve_coarse_many(
                q_multis,
                self._faiss_index,
                k=self._coarse_k,
                safe_embedder=self._safe_embedder,
            )

            # 统计 coarse recall，并准备精排输入
            rerank_inputs: List[Tuple[str, dict, List[str]]] = []
            for qid, q_mm, coarse_ids in zip(batch_ids, q_multis, coarse_lists):
                positives = set(ground_truth.get(qid) or [])
                if positives and any(cid in positives for cid in coarse_ids):
                    coarse_hits += 1
                if rerank_k is not None and rerank_k > 0:
                    coarse_ids = coarse_ids[: min(rerank_k, len(coarse_ids))]
                rerank_inputs.append((qid, q_mm, coarse_ids))

            # 批量精排：按 query 分块
            for i in range(0, len(rerank_inputs), 1):
                qid, q_mm, cand_ids = rerank_inputs[i]
                if not cand_ids:
                    n_total += 1
                    continue
                cand_feats = load_candidate_features_from_dict(cand_ids, self._library_features)
                ranked = self._rerank_model.score(
                    q_mm, cand_feats, batch_size=rerank_batch_size
                )
                positives = set(ground_truth.get(qid) or [])
                if ranked and ranked[0][0] in positives:
                    r1_hits += 1
                n_total += 1

            if progress_every > 0:
                done = min(start + len(batch_ids), len(ids))
                if done % progress_every == 0:
                    print(f"[validation] {done}/{len(ids)}", flush=True)

        if n_total == 0:
            return 0.0, 0.0
        return coarse_hits / n_total, r1_hits / n_total
