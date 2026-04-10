"""
两阶段「粗筛-精排」流水线：独立可调用，不依赖 DAG。

整合 LibraryFaissIndex（粗筛）、compute_rerank_scores（精排），
对外提供 retrieve、rerank、retrieve_and_rerank 接口。
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .faiss_library import LibraryFaissIndex, retrieve_coarse, retrieve_coarse_many
from .rerank import RerankModel, load_candidate_features_from_dict

logger = logging.getLogger(__name__)

# 小于此阈值的 library_features.json 全量加载（避免 seek 索引开销）；
# 超过此阈值则构建 key→byte-offset 索引，精排时按需读取。
_EAGER_LOAD_THRESHOLD_BYTES = 100 * 1024 * 1024  # 100 MiB


def _default_rerank_model_path() -> str:
    """项目根目录下的 output/best_model.pth。"""
    # src/matcher/two_stage.py -> 上三级 -> 项目根
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "output", "best_model.pth")


def _find_json_value_end(data: bytes, start: int) -> int:
    """
    从 data[start:] 开始，找到与 start 位置 JSON 值配对的结束位置。
    支持嵌套对象/数组、字符串、数字、布尔、null。
    返回值结束后第一个非空白字符的字节偏移。
    """
    i = start
    n = len(data)
    # 跳过前导空白
    while i < n and data[i] in (0x20, 0x09, 0x0A, 0x0D):
        i += 1
    if i >= n:
        return n

    ch = data[i]
    if ch == 0x7B:  # '{'
        depth = 1
        i += 1
        in_str = False
        esc = False
        while i < n and depth > 0:
            c = data[i]
            if esc:
                esc = False
            elif c == 0x5C and in_str:  # backslash
                esc = True
            elif c == 0x22 and not in_str:  # '"'
                in_str = True
            elif c == 0x22 and in_str:  # '"'
                in_str = False
            elif not in_str:
                if c == 0x7B:
                    depth += 1
                elif c == 0x7D:
                    depth -= 1
            i += 1
        return i
    elif ch == 0x5B:  # '['
        depth = 1
        i += 1
        in_str = False
        esc = False
        while i < n and depth > 0:
            c = data[i]
            if esc:
                esc = False
            elif c == 0x5C and in_str:
                esc = True
            elif c == 0x22 and not in_str:
                in_str = True
            elif c == 0x22 and in_str:
                in_str = False
            elif not in_str:
                if c == 0x5B:
                    depth += 1
                elif c == 0x5D:
                    depth -= 1
            i += 1
        return i
    elif ch == 0x22:  # '"'
        i += 1
        esc = False
        while i < n:
            c = data[i]
            if esc:
                esc = False
            elif c == 0x5C:
                esc = True
            elif c == 0x22:
                i += 1
                return i
            i += 1
        return n
    else:
        # number / true / false / null — 扫到 , } ] 或空白
        while i < n and data[i] not in (0x2C, 0x7D, 0x5D, 0x20, 0x09, 0x0A, 0x0D):
            i += 1
        return i


class _LibraryFeaturesLazy:
    """
    惰性加载的 library_features 索引。

    小文件：一次性 json.load，行为等价 dict。
    大文件：首扫构建 key→(start, end) 字节偏移索引，
            get() 时只读取并解析目标 key 的 JSON 切片。
    """

    def __init__(self, path: str, *, eager_threshold: int = _EAGER_LOAD_THRESHOLD_BYTES) -> None:
        self._path = path
        self._eager: Optional[Dict[str, Any]] = None
        self._index: Optional[Dict[str, Tuple[int, int]]] = None
        self._fd: Optional[Any] = None

        try:
            file_size = os.path.getsize(path)
        except OSError:
            file_size = 0

        if file_size <= eager_threshold:
            self._load_eager()
        else:
            self._build_index()

    # ------------------------------------------------------------------
    # Eager path (small files)
    # ------------------------------------------------------------------

    def _load_eager(self) -> None:
        with open(self._path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"library_features 格式应为 {{function_id: multimodal}}，"
                f"实际得到 {type(data).__name__}"
            )
        self._eager = data

    # ------------------------------------------------------------------
    # Lazy path (large files)
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """
        扫描 JSON 文件，建立 top-level key → 字节偏移映射。
        文件格式假定为 {"key1": value1, "key2": value2, ...}，
        key 均为字符串，value 为任意 JSON 值。

        使用 mmap 避免将整个文件读入进程内存。
        """
        import mmap as _mmap

        self._index = {}
        self._fd = open(self._path, "rb")
        self._mm = _mmap.mmap(self._fd.fileno(), 0, access=_mmap.ACCESS_READ)
        buf = self._mm
        n = len(buf)

        # 定位到第一个 '{'
        i = 0
        while i < n and buf[i] != 0x7B:
            i += 1
        i += 1  # skip '{'

        while i < n:
            # 跳过空白和逗号
            while i < n and buf[i] in (0x20, 0x09, 0x0A, 0x0D, 0x2C):
                i += 1
            if i >= n or buf[i] == 0x7D:  # '}'
                break

            # 解析 key: "... "
            if buf[i] != 0x22:  # '"'
                break
            key_start = i + 1
            i += 1
            esc = False
            while i < n:
                c = buf[i]
                if esc:
                    esc = False
                elif c == 0x5C:
                    esc = True
                elif c == 0x22:
                    break
                i += 1
            key_end = i
            key = buf[key_start:key_end].decode("utf-8", errors="replace")
            i += 1  # skip closing '"'

            # 跳过 ':'
            while i < n and buf[i] in (0x20, 0x09, 0x0A, 0x0D, 0x3A):
                i += 1

            # 记录 value 的起始位置
            val_start = i
            val_end = _find_json_value_end(buf, i)
            self._index[key] = (val_start, val_end)
            i = val_end

        logger.info(
            "惰性 library_features 索引构建完成: %s, %d 个 key, 文件 %.1f MiB",
            self._path,
            len(self._index),
            n / (1024 * 1024),
        )

    # ------------------------------------------------------------------
    # dict-like interface
    # ------------------------------------------------------------------

    def __contains__(self, key: str) -> bool:
        if self._eager is not None:
            return key in self._eager
        return self._index is not None and key in self._index

    def __getitem__(self, key: str) -> Any:
        if self._eager is not None:
            return self._eager[key]
        if self._index is None or key not in self._index:
            raise KeyError(key)
        start, end = self._index[key]
        assert self._mm is not None
        raw = bytes(self._mm[start:end])
        return json.loads(raw)

    def get(self, key: str, default: Any = None) -> Any:
        if self._eager is not None:
            return self._eager.get(key, default)
        if self._index is None or key not in self._index:
            return default
        start, end = self._index[key]
        assert self._mm is not None
        raw = bytes(self._mm[start:end])
        return json.loads(raw)

    def __len__(self) -> int:
        if self._eager is not None:
            return len(self._eager)
        return len(self._index) if self._index is not None else 0

    def keys(self):  # noqa: ANN201
        if self._eager is not None:
            return self._eager.keys()
        return self._index.keys() if self._index is not None else {}.keys()

    def close(self) -> None:
        if hasattr(self, "_mm") and self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            self._fd.close()
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        mode = "eager" if self._eager is not None else "lazy"
        n = len(self)
        return f"<_LibraryFeaturesLazy {mode} keys={n} path={self._path!r}>"


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

        # library features：小文件全量加载，大文件构建 key→偏移索引惰性读取
        self._library_features = _LibraryFeaturesLazy(library_features_path)

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
            mm,
            self._faiss_index,
            k=self._coarse_k,
            safe_model_path=self._safe_model_path,
        )

    def rerank(self, query_func_id: str, candidate_ids: List[str]) -> List[Tuple[str, float]]:
        """
        精排：加载 query 与 candidate 特征，返回按得分降序的 [(candidate_id, score), ...]。
        空候选返回空列表。
        """
        if query_func_id not in self._query_features:
            raise KeyError(f"查询 function_id 不存在: {query_func_id}")
        if not candidate_ids:
            return []
        query_mm = self._query_features[query_func_id]
        cand_features = load_candidate_features_from_dict(candidate_ids, self._library_features)
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

            # 逐 query 精排（每个 query 的候选集不同，rerank_batch_size 控制候选内批大小）
            for i in range(len(rerank_inputs)):
                qid, q_mm, cand_ids = rerank_inputs[i]
                if not cand_ids:
                    n_total += 1
                    continue
                cand_feats = load_candidate_features_from_dict(cand_ids, self._library_features)
                ranked = self._rerank_model.score(q_mm, cand_feats, batch_size=rerank_batch_size)
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
