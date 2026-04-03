"""
训练数据集：成对函数特征，用于孪生/对比学习。
支持从 BinKit 索引动态提取，以及基于内存/磁盘的特征缓存。
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # type: ignore


def _normalize_entry(entry: str) -> str:
    """统一 entry 格式便于匹配：转为小写，确保 0x 前缀。"""
    s = (entry or "").strip().lower()
    if s.startswith("0x"):
        return s
    return f"0x{s}" if s else ""


def _entry_matches(a: str, b: str) -> bool:
    """判断两个 entry 是否表示同一地址。"""
    na = _normalize_entry(a)
    nb = _normalize_entry(b)
    if na == nb:
        return True
    try:
        return int(na, 16) == int(nb, 16)
    except ValueError:
        return False


def _cache_key(binary_path: str, entry: str) -> str:
    """生成缓存键的哈希。"""
    raw = f"{os.path.abspath(binary_path)}|{_normalize_entry(entry)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _function_id(binary_path: str, entry: str) -> str:
    return f"{binary_path}|{_normalize_entry(entry)}"


def _iter_with_prebuild_progress(
    iterable: Iterable[Any],
    *,
    total: Optional[int],
    desc: str,
    log: logging.Logger,
    unit: str = "条",
    log_info_every: Optional[int] = None,
) -> Iterator[Any]:
    """
    预扫描侧车时的进度反馈：优先 tqdm（不在此路径打与进度重复的 log.info，避免与进度条交错刷屏）；
    无 tqdm 时由 log_info_every 或默认间隔打 INFO。
    """
    est = total if total is not None and total > 0 else None
    try:
        from tqdm import tqdm

        with tqdm(
            iterable,
            total=est,
            desc=desc,
            unit=unit,
            dynamic_ncols=True,
            mininterval=0.15,
            miniters=1,
        ) as pbar:
            for item in pbar:
                yield item
    except ImportError:
        n = 0
        every = (
            log_info_every
            if log_info_every is not None
            else (max(1, min(5000, est // 40)) if est else 2000)
        )
        for item in iterable:
            n += 1
            if est is not None:
                if n == 1 or n % every == 0 or n >= est:
                    log.info("%s: %d / %d", desc, min(n, est), est)
            elif n == 1 or n % every == 0:
                log.info("%s: 已处理 %d 条…", desc, n)
            yield item
        log.info("%s: 完成，共 %d 条", desc, n)


def _normalize_mix_weights(d: Optional[Dict[str, float]], defaults: Dict[str, float]) -> Tuple[List[str], List[float]]:
    base = dict(defaults)
    if d:
        for k, v in d.items():
            if k in base:
                base[k] = float(v)
    keys = list(base.keys())
    w = [max(0.0, base[k]) for k in keys]
    s = sum(w) or 1.0
    return keys, [x / s for x in w]


class PairwiseFunctionDataset(Dataset):
    """
    成对函数数据集，用于孪生网络对比学习。
    正对：默认「不同二进制中的同名」；pairing_mode=binkit_refined 时为同源（启发式 project_id）
    + 同名，并可优先跨编译变体 / 按 pair_mix 分层。
    负对：默认随机不同函数；binkit_refined 时可混合同源异名（硬负）与图规模相近的硬负。
    """

    def __init__(
        self,
        index_path: str,
        project_root: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_disk_cache: bool = False,
        precomputed_features_path: Optional[str] = None,
        memory_cache_max_items: int = 8192,
        lsir_cache_max_binaries: int = 32,
        num_pairs: int = 1000,
        positive_ratio: float = 0.5,
        seed: Optional[int] = None,
        pairing_mode: str = "legacy",
        max_cfg_node_ratio: float = 0.0,
        prefer_cross_variant_positive: bool = True,
        pair_mix: Optional[Dict[str, float]] = None,
        negative_weights: Optional[Dict[str, float]] = None,
        graph_similar_max_delta: int = 4,
        *,
        precomputed_lazy_reuse_read_file_handle: bool = True,
        precomputed_lazy_log_first_n: int = 0,
        fixed_pairs_per_epoch: bool = False,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for PairwiseFunctionDataset")
        self.project_root = project_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.cache_dir = cache_dir or os.path.join(self.project_root, "data", "features_cache")
        self.use_disk_cache = use_disk_cache
        self.precomputed_features_path = precomputed_features_path
        self.memory_cache_max_items = max(0, int(memory_cache_max_items))
        self.lsir_cache_max_binaries = max(0, int(lsir_cache_max_binaries))
        self.num_pairs = num_pairs
        self.positive_ratio = positive_ratio
        self._rng = random.Random(seed)
        self.pairing_mode = (pairing_mode or "legacy").strip().lower()
        self.max_cfg_node_ratio = float(max_cfg_node_ratio)
        self.prefer_cross_variant_positive = bool(prefer_cross_variant_positive)
        self.graph_similar_max_delta = max(0, int(graph_similar_max_delta))
        self._pair_mix_keys, self._pair_mix_probs = _normalize_mix_weights(
            pair_mix,
            {"cross_arch": 1.0, "same_arch_cross_compiler": 1.0, "same_arch_same_toolchain": 1.0, "any": 1.0},
        )
        self._neg_keys, self._neg_probs = _normalize_mix_weights(
            negative_weights,
            {"hard_same_project": 0.35, "hard_similar_graph": 0.25, "random": 0.4},
        )

        os.makedirs(self.cache_dir, exist_ok=True)

        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._lsir_raw_cache: Dict[str, Dict[str, Any]] = {}
        self._binary_abs_to_rel: Dict[str, str] = {}
        self._precomputed_features: Dict[str, Dict[str, Any]] = {}
        self._precomputed_lazy_index: Optional[Any] = None
        self._precomputed_lazy_log_remaining = max(0, int(precomputed_lazy_log_first_n))
        self._precomputed_lazy_reuse_read_file_handle = bool(precomputed_lazy_reuse_read_file_handle)
        self._fixed_pairs_per_epoch = bool(fixed_pairs_per_epoch)
        self._epoch_pair_specs: Optional[List[Optional[Tuple[Tuple[str, str], Tuple[str, str], int]]]] = None

        self._index: List[Dict[str, Any]] = []
        self._name_to_sites: Dict[str, List[Tuple[str, str]]] = {}
        self._all_sites: List[Tuple[str, str, str]] = []
        self._load_index(index_path)

        needed_ids: Optional[Set[str]] = None
        if precomputed_features_path:
            needed_ids = {
                self._function_id_for_precomputed(ba, e) for ba, e, _name in self._all_sites
            }
        try:
            from utils.precomputed_multimodal_io import (
                build_jsonl_sidecar_lazy_index,
                is_jsonl_sidecar_path,
                load_precomputed_multimodal_map,
            )

            if precomputed_features_path and is_jsonl_sidecar_path(precomputed_features_path):
                self._precomputed_lazy_index = build_jsonl_sidecar_lazy_index(
                    precomputed_features_path,
                    needed_ids or set(),
                    reuse_read_file_handle=self._precomputed_lazy_reuse_read_file_handle,
                )
            else:
                self._precomputed_features = load_precomputed_multimodal_map(
                    precomputed_features_path,
                    needed_ids,
                )
        except Exception:
            self._precomputed_features = {}
            self._precomputed_lazy_index = None

        self._binary_meta: Dict[str, Tuple[str, Any]] = {}
        self._refined_positive_candidates: List[Tuple[str, str, List[Tuple[str, str]]]] = []
        self._project_name_to_sites: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
        self._projects_multi_name: List[str] = []
        self._sites_with_nodes: Optional[List[Tuple[str, str, int]]] = None
        # num_nodes -> 站点列表；hard_similar 负采样用，避免每次 O(N) 扫全表
        self._sites_by_num_nodes: Dict[int, List[Tuple[str, str]]] = {}
        self._build_binary_meta_and_refined()
        self._prebuild_sites_with_nodes_if_needed()

    def _neg_weight(self, key: str) -> float:
        for k, p in zip(self._neg_keys, self._neg_probs):
            if k == key:
                return float(p)
        return 0.0

    def _prebuild_sites_with_nodes_if_needed(self) -> None:
        """
        hard_similar_graph 负采样需要「站点 -> graph.num_nodes」列表。
        若在首个 __getitem__ 里惰性扫描全索引 + JSONL 随机读，首个 batch 会极慢甚至像卡死；
        故在初始化时一次性构建；JSONL 侧车走 bulk_get_iter 顺序读。
        """
        log = logging.getLogger(__name__)
        if self.pairing_mode != "binkit_refined" or self._neg_weight("hard_similar_graph") <= 0.0:
            return
        if self._sites_with_nodes is not None:
            return
        if not self._precomputed_features and self._precomputed_lazy_index is None:
            self._sites_with_nodes = []
            self._sites_by_num_nodes = {}
            return

        site_by_fid: Dict[str, Tuple[str, str]] = {}
        for ba, e, _name in self._all_sites:
            fid = self._function_id_for_precomputed(ba, e)
            site_by_fid.setdefault(fid, (ba, e))
        ordered_fids = list(site_by_fid.keys())
        acc: List[Tuple[str, str, int]] = []

        lazy = self._precomputed_lazy_index
        if lazy is not None and hasattr(lazy, "bulk_get_iter"):
            idx_map = getattr(lazy, "_index", None)
            bulk_total: Optional[int] = None
            if isinstance(idx_map, dict):
                bulk_total = sum(1 for fid in ordered_fids if fid in idx_map)
            log.info(
                "PairwiseFunctionDataset: 预构建 hard_similar 图节点表（顺序读 JSONL，约 %d 行）…",
                bulk_total if bulk_total is not None else len(ordered_fids),
            )
            try:
                raw_it = lazy.bulk_get_iter(ordered_fids)
                for fid, mm in _iter_with_prebuild_progress(
                    raw_it,
                    total=bulk_total,
                    desc="预扫描 graph.num_nodes",
                    log=log,
                    unit="行",
                ):
                    if not isinstance(mm, dict):
                        continue
                    g = mm.get("graph") or {}
                    try:
                        nn = int(g.get("num_nodes") or 0)
                    except (TypeError, ValueError):
                        nn = 0
                    if nn > 0:
                        ba, e = site_by_fid[fid]
                        acc.append((ba, e, nn))
            except Exception as ex:
                log.warning("bulk_get_iter 失败，回退逐条读取: %s", ex)
                acc = []
                for ba, e, _n in _iter_with_prebuild_progress(
                    self._all_sites,
                    total=len(self._all_sites),
                    desc="预扫描 graph.num_nodes（回退）",
                    log=log,
                    unit="fn",
                ):
                    nn = self._graph_nodes_precomputed(ba, e)
                    if nn is not None and nn > 0:
                        acc.append((ba, e, nn))
        else:
            sites_it2 = _iter_with_prebuild_progress(
                self._all_sites,
                total=len(self._all_sites),
                desc="预扫描 graph.num_nodes",
                log=log,
                unit="fn",
            )
            for ba, e, _n in sites_it2:
                nn = self._graph_nodes_precomputed(ba, e)
                if nn is not None and nn > 0:
                    acc.append((ba, e, nn))

        self._sites_with_nodes = acc
        self._rebuild_sites_by_num_nodes_from_acc(acc)
        log.info(
            "PairwiseFunctionDataset: hard_similar 有效站点 %d（有 graph.num_nodes），%d 个不同节点数分桶",
            len(acc),
            len(self._sites_by_num_nodes),
        )

    def _rebuild_sites_by_num_nodes_from_acc(self, acc: List[Tuple[str, str, int]]) -> None:
        m: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
        for ba, e, nn in acc:
            if nn > 0:
                m[int(nn)].append((ba, e))
        self._sites_by_num_nodes = {k: v for k, v in m.items()}

    def _build_binary_meta_and_refined(self) -> None:
        from utils.binkit_provenance import parse_binary_provenance
        from utils.training_function_filter import strip_linker_suffix

        for binary_abs, rel in self._binary_abs_to_rel.items():
            pid, hints = parse_binary_provenance(rel)
            self._binary_meta[binary_abs] = (pid, hints)

        if self.pairing_mode != "binkit_refined":
            return

        grp: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
        proj_names: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))

        for binary_abs, entry, name in self._all_sites:
            rel = self._binary_abs_to_rel.get(binary_abs, "")
            pid, _ = parse_binary_provenance(rel)
            key_name = strip_linker_suffix((name or "").strip())
            if not key_name:
                continue
            grp[(pid, key_name)].append((binary_abs, entry))
            proj_names[pid][key_name].append((binary_abs, entry))

        self._refined_positive_candidates = [
            (pid, kn, sites)
            for (pid, kn), sites in grp.items()
            if len({b for b, _ in sites}) >= 2
        ]
        self._project_name_to_sites = {k: dict(v) for k, v in proj_names.items()}
        self._projects_multi_name = [pid for pid, m in self._project_name_to_sites.items() if len(m) >= 2]

    def _hints_for_binary(self, binary_abs: str) -> Any:
        from utils.binkit_provenance import VariantHints

        t = self._binary_meta.get(binary_abs)
        if not t:
            return VariantHints()
        return t[1]

    def _peek_precomputed_multimodal(self, binary_path: str, entry: str) -> Optional[Dict[str, Any]]:
        if not self._precomputed_features and self._precomputed_lazy_index is None:
            return None
        fid = self._function_id_for_precomputed(binary_path, entry)
        precomputed = self._precomputed_features.get(fid)
        if precomputed is None and self._precomputed_lazy_index is not None:
            precomputed = self._precomputed_lazy_index.get(fid)
        return precomputed if isinstance(precomputed, dict) else None

    def _graph_nodes_precomputed(self, binary_path: str, entry: str) -> Optional[int]:
        mm = self._peek_precomputed_multimodal(binary_path, entry)
        if not mm:
            return None
        g = mm.get("graph") or {}
        try:
            return int(g.get("num_nodes") or 0)
        except (TypeError, ValueError):
            return None

    def _cfg_ratio_ok(self, n1: int, n2: int) -> bool:
        if self.max_cfg_node_ratio <= 0:
            return True
        a, b = max(n1, n2), max(min(n1, n2), 1)
        return (a / b) <= self.max_cfg_node_ratio

    def _ensure_sites_with_nodes(self) -> None:
        if self._sites_with_nodes is not None:
            return
        acc: List[Tuple[str, str, int]] = []
        for binary_abs, entry, _n in self._all_sites:
            nn = self._graph_nodes_precomputed(binary_abs, entry)
            if nn is not None and nn > 0:
                acc.append((binary_abs, entry, nn))
        self._sites_with_nodes = acc
        self._rebuild_sites_by_num_nodes_from_acc(acc)

    def _mix_key_for_relation(self, rel: str) -> str:
        if rel == "unknown":
            return "any"
        if rel in ("cross_arch", "same_arch_cross_compiler", "same_arch_same_toolchain"):
            return rel
        return "any"

    def _random_pair_in_group(
        self,
        sites: List[Tuple[str, str]],
        mix_target: str,
        *,
        inner_attempts: int = 96,
    ) -> Optional[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """
        在组内随机抽一对跨二进制站点，满足 mix_target；O(1) 内存。
        替代物化全部 O(n^2) 候选对（大组会导致单次采样卡死数分钟）。
        """
        from utils.binkit_provenance import classify_pair_relation

        n = len(sites)
        if n < 2:
            return None
        for _ in range(inner_attempts):
            i, j = self._rng.sample(range(n), 2)
            sa, sb = sites[i], sites[j]
            if sa[0] == sb[0]:
                continue
            if mix_target != "any":
                h1 = self._hints_for_binary(sa[0])
                h2 = self._hints_for_binary(sb[0])
                rel = classify_pair_relation(h1, h2)
                if self._mix_key_for_relation(rel) != mix_target:
                    continue
            return (sa, sb)
        return None

    def _pick_weighted(self, keys: List[str], probs: List[float]) -> str:
        return self._rng.choices(keys, weights=probs, k=1)[0]

    def _features_from_sites(
        self, a: Tuple[str, str], b: Tuple[str, str]
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        feat1 = self._get_features(a[0], a[1])
        feat2 = self._get_features(b[0], b[1])
        if feat1 is None or feat2 is None:
            return None
        return (feat1, feat2)

    def _sample_positive_refined_sites(self) -> Optional[Tuple[Tuple[str, str], Tuple[str, str]]]:
        if not self._refined_positive_candidates:
            return self._sample_positive_legacy_sites()
        pool = self._refined_positive_candidates
        mix_target = self._pick_weighted(self._pair_mix_keys, self._pair_mix_probs)

        def _cfg_ok(sa: Tuple[str, str], sb: Tuple[str, str]) -> bool:
            if self.max_cfg_node_ratio <= 0:
                return True
            n1 = self._graph_nodes_precomputed(sa[0], sa[1])
            n2 = self._graph_nodes_precomputed(sb[0], sb[1])
            if n1 is None or n2 is None or n1 <= 0 or n2 <= 0:
                return True
            return self._cfg_ratio_ok(n1, n2)

        if self.prefer_cross_variant_positive:
            for _ in range(64):
                _pid, _kn, sites = self._rng.choice(pool)
                cand = self._random_pair_in_group(sites, mix_target, inner_attempts=96)
                if cand is None:
                    continue
                sa, sb = cand
                fa = self._hints_for_binary(sa[0]).fingerprint()
                fb = self._hints_for_binary(sb[0]).fingerprint()
                if fa and fb and fa != fb and _cfg_ok(sa, sb):
                    return (sa, sb)

        for _ in range(48):
            _pid, _kn, sites = self._rng.choice(pool)
            cand = self._random_pair_in_group(sites, mix_target, inner_attempts=96)
            if cand is None:
                continue
            sa, sb = cand
            if _cfg_ok(sa, sb):
                return (sa, sb)

        for _ in range(48):
            _pid, _kn, sites = self._rng.choice(pool)
            cand = self._random_pair_in_group(sites, "any", inner_attempts=96)
            if cand is None:
                continue
            sa, sb = cand
            if _cfg_ok(sa, sb):
                return (sa, sb)
        return None

    def _sample_positive_refined(self) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        sites = self._sample_positive_refined_sites()
        if sites is None:
            return None
        return self._features_from_sites(sites[0], sites[1])

    def _sample_positive_legacy_sites(self) -> Optional[Tuple[Tuple[str, str], Tuple[str, str]]]:
        if not self._positive_candidates:
            return None
        name, sites = self._rng.choice(self._positive_candidates)
        if len(sites) < 2:
            return None
        a, b = self._rng.sample(sites, 2)
        if self.max_cfg_node_ratio > 0:
            n1 = self._graph_nodes_precomputed(a[0], a[1])
            n2 = self._graph_nodes_precomputed(b[0], b[1])
            if n1 is not None and n2 is not None and n1 > 0 and n2 > 0:
                if not self._cfg_ratio_ok(n1, n2):
                    return None
        return (a, b)

    def _sample_positive_legacy(self) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        sites = self._sample_positive_legacy_sites()
        if sites is None:
            return None
        return self._features_from_sites(sites[0], sites[1])

    def _sample_negative_legacy_sites(self) -> Optional[Tuple[Tuple[str, str], Tuple[str, str]]]:
        if len(self._all_sites) < 2:
            return None
        a, b = self._rng.sample(self._all_sites, 2)
        tries = 0
        while a[2] == b[2] and a[0] == b[0] and tries < 50:
            a, b = self._rng.sample(self._all_sites, 2)
            tries += 1
        if a[2] == b[2] and a[0] == b[0]:
            return None
        return ((a[0], a[1]), (b[0], b[1]))

    def _sample_negative_legacy(self) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        sites = self._sample_negative_legacy_sites()
        if sites is None:
            return None
        return self._features_from_sites(sites[0], sites[1])

    def _sample_hard_same_project_sites(self) -> Optional[Tuple[Tuple[str, str], Tuple[str, str]]]:
        if not self._projects_multi_name:
            return None
        for _ in range(16):
            pid = self._rng.choice(self._projects_multi_name)
            names = list(self._project_name_to_sites[pid].keys())
            if len(names) < 2:
                continue
            na, nb = self._rng.sample(names, 2)
            sa = self._rng.choice(self._project_name_to_sites[pid][na])
            sb = self._rng.choice(self._project_name_to_sites[pid][nb])
            if sa[0] == sb[0] and sa[1] == sb[1]:
                continue
            return (sa, sb)
        return None

    def _sample_hard_same_project(self) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        sites = self._sample_hard_same_project_sites()
        if sites is None:
            return None
        return self._features_from_sites(sites[0], sites[1])

    def _sample_hard_similar_graph_sites(self) -> Optional[Tuple[Tuple[str, str], Tuple[str, str]]]:
        self._ensure_sites_with_nodes()
        sw = self._sites_with_nodes or []
        if len(sw) < 2:
            return None
        if not self._sites_by_num_nodes:
            self._rebuild_sites_by_num_nodes_from_acc(sw)
        dmax = self.graph_similar_max_delta
        by_nn = self._sites_by_num_nodes
        for _ in range(40):
            a0, a1, na = self._rng.choice(sw)
            d = self._rng.randint(-dmax, dmax)
            k = na + d
            if k < 1:
                continue
            bucket = by_nn.get(k)
            if not bucket:
                continue
            for __ in range(24):
                b0, b1 = self._rng.choice(bucket)
                if b0 == a0 and b1 == a1:
                    continue
                return ((a0, a1), (b0, b1))
        return None

    def _sample_hard_similar_graph(self) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        sites = self._sample_hard_similar_graph_sites()
        if sites is None:
            return None
        return self._features_from_sites(sites[0], sites[1])

    def _sample_negative_refined_sites(self) -> Optional[Tuple[Tuple[str, str], Tuple[str, str]]]:
        mode = self._pick_weighted(self._neg_keys, self._neg_probs)
        if mode == "hard_same_project":
            s = self._sample_hard_same_project_sites()
            if s is not None:
                return s
        elif mode == "hard_similar_graph":
            s = self._sample_hard_similar_graph_sites()
            if s is not None:
                return s
        return self._sample_negative_legacy_sites()

    def _sample_negative_refined(self) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        sites = self._sample_negative_refined_sites()
        if sites is None:
            return None
        return self._features_from_sites(sites[0], sites[1])

    def _sample_positive_sites(self) -> Optional[Tuple[Tuple[str, str], Tuple[str, str]]]:
        if self.pairing_mode == "binkit_refined" and self._refined_positive_candidates:
            return self._sample_positive_refined_sites()
        return self._sample_positive_legacy_sites()

    def _sample_negative_sites(self) -> Optional[Tuple[Tuple[str, str], Tuple[str, str]]]:
        if self.pairing_mode == "binkit_refined":
            return self._sample_negative_refined_sites()
        return self._sample_negative_legacy_sites()

    def _try_draw_pair_spec(self) -> Optional[Tuple[Tuple[str, str], Tuple[str, str], int]]:
        is_positive = self._rng.random() < self.positive_ratio
        label = 1 if is_positive else 0
        sites = self._sample_positive_sites() if is_positive else self._sample_negative_sites()
        max_retries = 20
        while sites is None and max_retries > 0:
            is_positive = self._rng.random() < self.positive_ratio
            label = 1 if is_positive else 0
            sites = self._sample_positive_sites() if is_positive else self._sample_negative_sites()
            max_retries -= 1
        if sites is None:
            return None
        (ba1, e1), (ba2, e2) = sites
        return ((ba1, e1), (ba2, e2), label)

    def regenerate_epoch_pairs(self) -> None:
        """在每个训练 epoch 开始时调用：固定本 epoch 的 num_pairs 条站点对，提升 JSONL/内存缓存命中率。"""
        if not self._fixed_pairs_per_epoch:
            self._epoch_pair_specs = None
            return
        log = logging.getLogger(__name__)
        log.info(
            "PairwiseFunctionDataset: 生成本 epoch 固定训练对 %d 条（仅站点坐标，特征在 __getitem__ 懒加载）…",
            self.num_pairs,
        )
        specs: List[Optional[Tuple[Tuple[str, str], Tuple[str, str], int]]] = []
        # 周期性 INFO（每 1000 对）便于日志管线观察；tqdm 仍输出进度条
        _log_every = max(1, min(2000, max(self.num_pairs // 20, 500)))
        for _ in _iter_with_prebuild_progress(
            range(self.num_pairs),
            total=self.num_pairs,
            desc="生成本 epoch 固定对",
            log=log,
            unit="对",
            log_info_every=_log_every,
        ):
            specs.append(self._try_draw_pair_spec())
        self._epoch_pair_specs = specs
        ok = sum(1 for x in specs if x is not None)
        log.info("PairwiseFunctionDataset: 固定对生成完成，有效 %d / %d", ok, self.num_pairs)

    def _sample_positive(self) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        if self.pairing_mode == "binkit_refined" and self._refined_positive_candidates:
            return self._sample_positive_refined()
        return self._sample_positive_legacy()

    def _sample_negative(self) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        if self.pairing_mode == "binkit_refined":
            return self._sample_negative_refined()
        return self._sample_negative_legacy()

    def _put_memory_cache(self, key: str, value: Dict[str, Any]) -> None:
        if self.memory_cache_max_items <= 0:
            return
        self._memory_cache[key] = value
        while len(self._memory_cache) > self.memory_cache_max_items:
            self._memory_cache.pop(next(iter(self._memory_cache)))

    def _put_lsir_cache(self, binary_path: str, value: Dict[str, Any]) -> None:
        if self.lsir_cache_max_binaries <= 0:
            return
        self._lsir_raw_cache[binary_path] = value
        while len(self._lsir_raw_cache) > self.lsir_cache_max_binaries:
            self._lsir_raw_cache.pop(next(iter(self._lsir_raw_cache)))

    def clear_runtime_cache(self, *, clear_memory: bool = True, clear_lsir: bool = True) -> None:
        """清空运行期缓存，供长训练过程中按 epoch 主动回收 RAM。"""
        if clear_memory:
            self._memory_cache.clear()
        if clear_lsir:
            self._lsir_raw_cache.clear()

    def _load_index(self, index_path: str) -> None:
        with open(index_path, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raw = [raw] if isinstance(raw, dict) else []

        for item in raw:
            if not isinstance(item, dict):
                continue
            binary_rel = item.get("binary", "")
            funcs = item.get("functions") or []
            binary_abs = os.path.join(self.project_root, binary_rel) if not os.path.isabs(binary_rel) else binary_rel
            self._binary_abs_to_rel[binary_abs] = binary_rel
            for f in funcs:
                name = f.get("name", "")
                entry = f.get("entry", "")
                if not name or not entry:
                    continue
                self._all_sites.append((binary_abs, entry, name))
                self._name_to_sites.setdefault(name, []).append((binary_abs, entry))
            self._index.append({"binary": binary_abs, "functions": funcs})

        self._positive_candidates = [(name, sites) for name, sites in self._name_to_sites.items() if len(sites) >= 2]

    def _function_id_for_precomputed(self, binary_path: str, entry: str) -> str:
        binary_rel = self._binary_abs_to_rel.get(binary_path)
        if not binary_rel:
            binary_rel = os.path.relpath(binary_path, self.project_root).replace("\\", "/")
        return _function_id(binary_rel, entry)

    def _get_features(self, binary_path: str, entry: str) -> Optional[Dict[str, Any]]:
        """获取单函数的多模态特征，优先缓存。"""
        ck = _cache_key(binary_path, entry)
        if self.memory_cache_max_items > 0 and ck in self._memory_cache:
            return self._memory_cache[ck]

        if self._precomputed_features or self._precomputed_lazy_index is not None:
            fid = self._function_id_for_precomputed(binary_path, entry)
            precomputed = self._precomputed_features.get(fid)
            if precomputed is None and self._precomputed_lazy_index is not None:
                t0 = time.perf_counter()
                precomputed = self._precomputed_lazy_index.get(fid)
                if self._precomputed_lazy_log_remaining > 0:
                    self._precomputed_lazy_log_remaining -= 1
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    logging.getLogger(__name__).info(
                        "PairwiseFunctionDataset: lazy JSONL get function_id=%r 耗时 %.1fms",
                        (fid[:96] + "…") if len(fid) > 96 else fid,
                        dt_ms,
                    )
            if precomputed is not None:
                self._put_memory_cache(ck, precomputed)
                return precomputed

        if self.use_disk_cache:
            disk_path = os.path.join(self.cache_dir, f"{ck}.json")
            if os.path.isfile(disk_path):
                try:
                    with open(disk_path, encoding="utf-8") as f:
                        data = json.load(f)
                    self._put_memory_cache(ck, data)
                    return data
                except Exception:
                    pass

        multimodal = self._extract_features(binary_path, entry)
        if multimodal is None:
            return None
        self._put_memory_cache(ck, multimodal)
        if self.use_disk_cache:
            try:
                disk_path = os.path.join(self.cache_dir, f"{ck}.json")
                with open(disk_path, "w", encoding="utf-8") as f:
                    json.dump(multimodal, f, ensure_ascii=False)
            except OSError:
                pass
        return multimodal

    def _extract_features(self, binary_path: str, entry: str) -> Optional[Dict[str, Any]]:
        """
        动态提取：(binary, entry) -> lsir_raw -> multimodal 特征。

        缓存策略（Plan B）：
        1. 先用 peek_binary_cache 检查 binary_cache（纯读，不创建任何目录）。
        2. 命中时直接使用，ghidra_temp 子目录不会被创建。
        3. 未命中时创建 ghidra_temp 子目录并调用 Ghidra；
           无论成功或异常，子目录均在 finally 中删除。
        _lsir_raw_cache 为进程内 lsir_raw 二级缓存，减少同二进制多次提取的重复解析。
        """
        import shutil as _shutil
        try:
            from utils.ghidra_runner import peek_binary_cache, run_ghidra_analysis
            from utils.ir_builder import build_lsir
            from utils.pcode_normalizer import normalize_lsir_raw
            from utils.feature_extractors import (
                extract_acfg_features,
                extract_graph_features,
                extract_sequence_features,
                fuse_features,
            )
        except ImportError:
            return None

        if binary_path not in self._lsir_raw_cache:
            # Plan B: 先 peek binary_cache，命中时不创建任何临时目录
            lsir_raw = peek_binary_cache(binary_path)
            if lsir_raw is not None:
                self._put_lsir_cache(binary_path, lsir_raw)
            else:
                output_dir = os.path.join(self.cache_dir, "ghidra_temp", _cache_key(binary_path, ""))
                os.makedirs(output_dir, exist_ok=True)
                try:
                    lsir_raw = run_ghidra_analysis(
                        binary_path=binary_path,
                        output_dir=output_dir,
                        script_name="extract_lsir_raw.java",
                        script_output_name="lsir_raw.json",
                        return_dict=True,
                    )
                    self._put_lsir_cache(binary_path, lsir_raw or {})
                except Exception:
                    return None
                finally:
                    # 子目录（成功或异常）用毕立即删除；binary_cache 已由 write_to_binary_cache 持久化
                    _shutil.rmtree(output_dir, ignore_errors=True)

        lsir_raw = self._lsir_raw_cache.get(binary_path)
        if lsir_raw is None:
            lsir_raw = peek_binary_cache(binary_path) or {}
            self._put_lsir_cache(binary_path, lsir_raw)
        funcs = lsir_raw.get("functions") or []
        target = None
        for f in funcs:
            if _entry_matches(f.get("entry", ""), entry):
                target = f
                break
        if target is None:
            return None

        raw = {"functions": [target]}
        raw = normalize_lsir_raw(raw)
        lsir = build_lsir(raw, include_cfg=True, include_dfg=True)
        fn_list = lsir.get("functions", [])
        if not fn_list:
            return None
        fn = fn_list[0]

        gf = extract_graph_features(fn)
        sf = extract_sequence_features(fn)
        acfg = extract_acfg_features(fn)
        fused = fuse_features(gf, sf, acfg_feats=acfg)
        return fused.get("multimodal")

    def __len__(self) -> int:
        return self.num_pairs

    def __getitem__(self, index: int) -> Dict[str, Any]:
        empty = {"graph": {"num_nodes": 0, "edge_index": [[], []], "node_list": [], "node_features": []},
                 "sequence": {"pcode_tokens": [], "jump_mask": [], "seq_len": 0}}

        if self._fixed_pairs_per_epoch:
            specs = self._epoch_pair_specs
            if specs is None:
                raise RuntimeError(
                    "fixed_pairs_per_epoch=True 时须在迭代 DataLoader 前调用 dataset.regenerate_epoch_pairs()（由 Trainer.on_epoch_begin 触发）"
                )
            if index < 0 or index >= len(specs):
                return {"feature1": empty, "feature2": empty, "label": 0}
            spec = specs[index]
            if spec is None:
                return {"feature1": empty, "feature2": empty, "label": 0}
            (ba1, e1), (ba2, e2), label = spec
            feat1 = self._get_features(ba1, e1)
            feat2 = self._get_features(ba2, e2)
            if feat1 is None or feat2 is None:
                return {"feature1": empty, "feature2": empty, "label": 0}
            return {"feature1": feat1, "feature2": feat2, "label": label}

        is_positive = self._rng.random() < self.positive_ratio
        pair = self._sample_positive() if is_positive else self._sample_negative()
        label = 1 if is_positive else 0

        max_retries = 20
        while pair is None and max_retries > 0:
            is_positive = self._rng.random() < self.positive_ratio
            pair = self._sample_positive() if is_positive else self._sample_negative()
            label = 1 if is_positive else 0
            max_retries -= 1

        if pair is None:
            return {"feature1": empty, "feature2": empty, "label": 0}

        feat1, feat2 = pair
        return {"feature1": feat1, "feature2": feat2, "label": label}


def _make_synthetic_multimodal(rng: "random.Random", vocab_keys: list, seq_len: int = 32, num_nodes: int = 8) -> Dict[str, Any]:
    """生成单个与 multimodal 兼容的随机特征。"""
    ops = [k for k in vocab_keys if k not in ("[PAD]", "[UNK]")]
    if not ops:
        ops = ["COPY", "INT_ADD", "INT_SUB"]
    tokens = [rng.choice(ops) for _ in range(seq_len)]
    jump_mask = [1 if rng.random() < 0.1 else 0 for _ in range(seq_len)]
    node_features = [{"pcode_opcodes": [rng.choice(ops) for _ in range(rng.randint(1, 5))]} for _ in range(num_nodes)]
    edges_src = list(range(num_nodes - 1))
    edges_dst = list(range(1, num_nodes))
    return {
        "graph": {
            "num_nodes": num_nodes,
            "edge_index": [edges_src, edges_dst],
            "node_list": [f"bb_{i}" for i in range(num_nodes)],
            "node_features": node_features,
        },
        "sequence": {"pcode_tokens": tokens, "jump_mask": jump_mask, "seq_len": seq_len},
    }


class PairwiseSyntheticDataset(Dataset):
    """
    合成成对数据集：从 JSON 加载或随机生成，用于快速验证训练流程。
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        num_pairs: int = 200,
        positive_ratio: float = 0.5,
        seed: Optional[int] = 42,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for PairwiseSyntheticDataset")
        self.num_pairs = num_pairs
        self.positive_ratio = positive_ratio
        self._rng = random.Random(seed)
        self._pairs: List[Dict[str, Any]] = []
        if data_path and os.path.isfile(data_path):
            try:
                with open(data_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._pairs = data.get("pairs", [])
            except Exception:
                pass
        if not self._pairs:
            vocab = _get_synthetic_vocab()
            vocab_keys = list(vocab.keys())
            for _ in range(num_pairs):
                f1 = _make_synthetic_multimodal(self._rng, vocab_keys)
                if self._rng.random() < positive_ratio:
                    f2 = copy.deepcopy(f1)
                    label = 1
                else:
                    f2 = _make_synthetic_multimodal(self._rng, vocab_keys)
                    label = 0
                self._pairs.append({"feature1": f1, "feature2": f2, "label": label})

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._pairs[index]


def _get_synthetic_vocab() -> Dict[str, int]:
    """合成数据用的 vocab 键列表。"""
    try:
        from features.models.multimodal_fusion import get_default_vocab
        return get_default_vocab()
    except ImportError:
        return {"[PAD]": 0, "[UNK]": 1, "COPY": 2, "INT_ADD": 3, "INT_SUB": 4}
