#!/usr/bin/env python3
"""
孪生网络训练脚本：MultiModalFusionModel + 成对对比学习。
支持 BinKit 索引与合成数据。
"""
import argparse
import random
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch
from torch.utils.data import DataLoader


def _setup_rotating_logging(
    *,
    log_dir: str,
    log_name: str,
    max_mb: int,
    backups: int,
    level: str = "INFO",
) -> None:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, log_name)
    max_bytes = max(1, int(max_mb)) * 1024 * 1024
    backup_count = max(0, int(backups))

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)


def _load_train_config(path: str) -> dict:
    """从 YAML 文件加载训练配置；职责单一，供 main 使用。"""
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return dict(data) if isinstance(data, dict) else {}
    except (OSError, ImportError) as e:
        print(f"警告: 无法加载配置 {path}: {e}", file=sys.stderr)
        return {}


def _collate_pairs(batch):
    """自定义 collate：保持 dict 列表，不 stacking。"""
    return {
        "feature1": [b["feature1"] for b in batch],
        "feature2": [b["feature2"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
    }


def _make_step_fn(
    vocab,
    device,
    loss_fn,
    threshold=0.5,
    max_seq_len: int = 512,
    max_graph_nodes: int = 128,
    max_dfg_nodes: int = 128,
):
    """构建训练步进函数：tensorize -> model -> loss + accuracy。"""

    def step_fn(batch, model, _loss_fn):
        from features.models.multimodal_fusion import _tensorize_multimodal

        f1_list = batch["feature1"] if isinstance(batch["feature1"], list) else [batch["feature1"]]
        f2_list = batch["feature2"] if isinstance(batch["feature2"], list) else [batch["feature2"]]
        labels = batch["label"]
        if torch.is_tensor(labels):
            labels = labels.float().to(device)
        else:
            labels = torch.tensor(labels, dtype=torch.float32, device=device)

        vec1_list = []
        vec2_list = []
        for f1, f2 in zip(f1_list, f2_list):
            try:
                t1, j1, n1, e1, p1, d1n, d1e = _tensorize_multimodal(
                    f1,
                    vocab,
                    device=device,
                    max_seq_len=max_seq_len,
                    max_graph_nodes=max_graph_nodes,
                    max_dfg_nodes=max_dfg_nodes,
                )
                t2, j2, n2, e2, p2, d2n, d2e = _tensorize_multimodal(
                    f2,
                    vocab,
                    device=device,
                    max_seq_len=max_seq_len,
                    max_graph_nodes=max_graph_nodes,
                    max_dfg_nodes=max_dfg_nodes,
                )
                v1 = model(
                    t1, j1, n1, e1, p1,
                    dfg_node_features=d1n, dfg_edge_index=d1e,
                )
                v2 = model(
                    t2, j2, n2, e2, p2,
                    dfg_node_features=d2n, dfg_edge_index=d2e,
                )
                if v1.dim() == 1:
                    v1 = v1.unsqueeze(0)
                if v2.dim() == 1:
                    v2 = v2.unsqueeze(0)
                vec1_list.append(v1)
                vec2_list.append(v2)
            except Exception:
                continue

        if not vec1_list:
            # 本 batch 全部样本 tensorize/forward 失败，跳过（count=0）
            return torch.tensor(0.0, device=device, requires_grad=True), 0, 0

        vec1 = torch.cat(vec1_list, dim=0)
        vec2 = torch.cat(vec2_list, dim=0)
        n = vec1.size(0)
        labels = labels[:n].to(device)
        loss = _loss_fn(vec1, vec2, labels)
        cos_sim = torch.nn.functional.cosine_similarity(vec1, vec2, dim=1)
        pred_sim = (cos_sim > threshold).float()
        correct = ((pred_sim == labels).float().sum().item())
        return loss, int(correct), n

    return step_fn


def main():
    parser = argparse.ArgumentParser(description="训练 MultiModalFusionModel 孪生网络")
    parser.add_argument("--config", default=None, help="YAML 配置文件路径；命令行参数覆盖配置文件")
    # 避免 -h 时过早退出：仅在非 help 时提前解析 --config
    if "-h" not in sys.argv and "--help" not in sys.argv:
        ns, _ = parser.parse_known_args()
        cfg = _load_train_config(ns.config) if ns.config and os.path.isfile(ns.config) else {}
    else:
        cfg = {}

    parser.add_argument("--index-file", default=None, help="binkit_functions.json 路径")
    parser.add_argument("--synthetic", action="store_true", help="使用合成数据")
    parser.add_argument("--synthetic-file", default=None, help="合成数据 JSON 路径（--synthetic 时）")
    parser.add_argument("--epochs", type=int, default=cfg.get("epochs", 20), help="训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=cfg.get("batch_size", 8), help="batch size")
    parser.add_argument("--lr", type=float, default=cfg.get("lr", 1e-4), help="学习率")
    parser.add_argument("--save-path", default=None, help="模型保存路径")
    parser.add_argument("--cache-dir", default=None, help="特征缓存目录")
    parser.add_argument("--tb-dir", default=None, help="TensorBoard 目录；未指定时使用 output/tensorboard/<timestamp>")
    parser.add_argument("--no-tb", action="store_true", help="禁用 TensorBoard")
    parser.add_argument("--precomputed-features", default=None, help="可选：预计算特征文件（{function_id: multimodal}），命中时优先读取")
    parser.add_argument(
        "--vocab-from-features",
        default=None,
        help="从特征文件构建词表：.json 为整表加载（大库易 OOM）；JSONL 侧车流式扫描（推荐）",
    )
    parser.add_argument("--num-pairs", type=int, default=cfg.get("num_pairs", 2000), help="数据集对数（真实数据时）")
    parser.add_argument("--vocab-size", type=int, default=cfg.get("vocab_size", 256), help="P-code vocab 大小")
    parser.add_argument("--use-disk-cache", action="store_true", help="启用特征磁盘缓存（默认已启用，可省略）")
    parser.add_argument("--no-disk-cache", action="store_true", help="禁用特征磁盘缓存")
    parser.add_argument("--num-workers", type=int, default=cfg.get("num_workers", 0), help="DataLoader worker 数（默认 0，降低 OOM 风险）")
    parser.add_argument("--memory-cache-max-items", type=int, default=cfg.get("memory_cache_max_items", 16384), help="数据集进程内特征缓存上限（0=禁用；预计算 JSONL 训练建议 ≥8k，过小会反复解析侧车）")
    parser.add_argument("--lsir-cache-max-binaries", type=int, default=cfg.get("lsir_cache_max_binaries", 2), help="数据集进程内 lsir_raw 缓存二进制上限（0=禁用）")
    parser.add_argument("--embed-dim", type=int, default=cfg.get("embed_dim", 64), help="嵌入维度")
    parser.add_argument("--hidden-dim", type=int, default=cfg.get("hidden_dim", 128), help="隐藏层维度")
    parser.add_argument("--num-gnn-layers", type=int, default=cfg.get("num_gnn_layers", 2), help="GNN 层数")
    parser.add_argument("--num-transformer-layers", type=int, default=cfg.get("num_transformer_layers", 2), help="Transformer 层数")
    parser.add_argument("--output-dim", type=int, default=cfg.get("output_dim", 128), help="输出嵌入维度")
    parser.add_argument("--max-seq-len", type=int, default=cfg.get("max_seq_len", 512), help="单样本最大序列长度（越大越耗显存）")
    parser.add_argument("--max-graph-nodes", type=int, default=cfg.get("max_graph_nodes", 128), help="单样本最大图节点数（越大越耗显存）")
    parser.add_argument(
        "--use-dfg",
        action=argparse.BooleanOptionalAction,
        default=bool(cfg.get("use_dfg", True)),
        help="启用 DFG 图分支（默认开；--no-use-dfg 仅用于消融）；需与推理/checkpoint meta 一致",
    )
    parser.add_argument("--max-dfg-nodes", type=int, default=cfg.get("max_dfg_nodes", 128), help="单样本 DFG 最大节点数")
    parser.add_argument("--max-log-mb", type=int, default=cfg.get("max_log_mb", 32), help="单个日志文件最大大小（MB）")
    parser.add_argument("--log-backups", type=int, default=cfg.get("log_backups", 5), help="日志滚动备份份数")
    parser.add_argument("--log-level", default=cfg.get("log_level", "INFO"), help="日志级别（DEBUG/INFO/WARNING/ERROR）")
    parser.add_argument("--max-tb-mb", type=int, default=cfg.get("max_tb_mb", 128), help="TensorBoard 根目录最大总大小（MB）")
    parser.add_argument("--tb-keep-runs", type=int, default=cfg.get("tb_keep_runs", 5), help="TensorBoard 仅保留最近 N 个 run 目录")
    parser.add_argument("--no-progress-bar", action="store_true", help="关闭 batch 级进度（tqdm 或周期性日志），仅保留每 epoch 汇总")
    parser.add_argument("--no-epoch-cleanup", action="store_true", help="关闭每个 epoch 后的 gc/CUDA cache 清理（默认开启，降低 OOM 风险）")
    parser.add_argument(
        "--progress-log-every",
        type=int,
        default=cfg.get("progress_log_every", 20),
        help="无 tqdm 时每隔多少个训练 batch 打印一行进度（需未加 --no-progress-bar）",
    )
    parser.add_argument("--seed", type=int, default=cfg.get("seed", 42), help="随机种子（可复现）")
    parser.add_argument(
        "--pairing-mode",
        choices=("legacy", "binkit_refined"),
        default=cfg.get("pairing_mode", "legacy"),
        help="legacy=跨二进制同名；binkit_refined=同源 project_id + 分层正负样本",
    )
    parser.add_argument(
        "--max-cfg-node-ratio",
        type=float,
        default=float(cfg.get("max_cfg_node_ratio", 0.0)),
        help="正对 CFG 节点比例上限（需预计算特征含 graph；0=不限制）",
    )
    parser.add_argument(
        "--prefer-cross-variant",
        action=argparse.BooleanOptionalAction,
        default=bool(cfg.get("prefer_cross_variant_positive", True)),
        help="refined 正对优先不同变体指纹（默认开）",
    )
    parser.add_argument(
        "--graph-similar-max-delta",
        type=int,
        default=int(cfg.get("graph_similar_max_delta", 4)),
        help="硬负例「图规模接近」时允许的 num_nodes 差",
    )
    parser.add_argument(
        "--init-weights",
        default=None,
        help="从已有 MultiModal 检查点加载 state_dict（strict=False）",
    )
    parser.add_argument(
        "--retrieval-val-dir",
        default=None,
        help="两阶段目录：含 library_features.json / query_features.json / ground_truth*.json，用于 epoch 末 Recall@1",
    )
    parser.add_argument(
        "--retrieval-val-gt",
        default="ground_truth_high_conf.json",
        help="相对 retrieval-val-dir 的 GT 文件名；不存在则回退 ground_truth.json",
    )
    parser.add_argument(
        "--retrieval-val-every",
        type=int,
        default=1,
        help="每 N 个 epoch 运行检索验证（默认 1；0=禁用）",
    )
    parser.add_argument(
        "--retrieval-val-subsample",
        type=int,
        default=0,
        help="检索验证最多评估的 query 数（0=全部）",
    )
    parser.add_argument("--wandb", action="store_true", help="启用 W&B 实验追踪")
    parser.add_argument("--wandb-project", default="sempatch", help="W&B 项目名（默认 sempatch）")
    parser.add_argument(
        "--no-precomputed-lazy-reuse-fp",
        action="store_true",
        help="禁用 JSONL 懒索引的单进程读句柄复用（每次 get 重新 open；num_workers>0 时脚本会自动等价关闭复用）",
    )
    parser.add_argument(
        "--precomputed-lazy-log-first-n",
        type=int,
        default=0,
        help="对前 N 次 JSONL 懒加载 get 打印 INFO 耗时（0=关闭；用于确认是否卡在磁盘解析）",
    )
    parser.add_argument(
        "--warmup-first-batch",
        action="store_true",
        help="在正式 tqdm 训练前先跑一个训练 batch 并记日志（会多一步梯度更新；用于把冷启动移出进度条起始位置）",
    )
    parser.add_argument(
        "--no-fixed-pairs-per-epoch",
        action="store_true",
        help="禁用「每 epoch 预生成固定 num_pairs 站点对」（默认：提供 --precomputed-features 时开启，可大幅提升 JSONL 缓存命中）",
    )
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=2,
        help="num_workers>0 时 DataLoader prefetch_factor（默认 2；仅 PyTorch DataLoader 支持）",
    )
    args = parser.parse_args()

    if not args.synthetic and not args.precomputed_features:
        try:
            from utils.ghidra_runner import GhidraEnvironmentError, require_ghidra_environment

            require_ghidra_environment()
        except GhidraEnvironmentError as e:
            print(
                f"错误: 未指定 --precomputed-features 时从索引动态提取需要 Ghidra: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

    # 13.4：随机种子管理，保证可复现
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy
        numpy.random.seed(seed)
    except ImportError:
        pass

    default_index = os.path.join(PROJECT_ROOT, "data", "binkit_functions.json")
    default_save = os.path.join(PROJECT_ROOT, "output", "best_model.pth")
    default_cache = os.path.join(PROJECT_ROOT, "data", "features_cache")

    index_path = args.index_file or default_index
    save_path = args.save_path or default_save
    cache_dir = args.cache_dir or default_cache

    _setup_rotating_logging(
        log_dir=os.path.join(PROJECT_ROOT, "output", "logs"),
        log_name="train_multimodal.log",
        max_mb=args.max_log_mb,
        backups=args.log_backups,
        level=args.log_level,
    )
    log = logging.getLogger("train_multimodal")

    # 13.2：可选 W&B；仅 except ImportError，不吞其他异常
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "num_pairs": args.num_pairs,
                "embed_dim": args.embed_dim,
                "hidden_dim": args.hidden_dim,
                "synthetic": args.synthetic,
                "seed": args.seed,
                "num_workers": args.num_workers,
                "max_seq_len": args.max_seq_len,
                "max_graph_nodes": args.max_graph_nodes,
            })
        except ImportError:
            log.warning("wandb 未安装，跳过 W&B 日志。运行: pip install wandb")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = (device.type == "cuda")
    num_workers = max(0, int(args.num_workers))
    from features.models.multimodal_fusion import MultiModalFusionModel, get_default_vocab
    from features.losses import ContrastiveLoss
    from features.trainer import Trainer

    if args.vocab_from_features and os.path.isfile(args.vocab_from_features):
        from features.baselines.safe import (
            collect_vocab_from_features_file,
            collect_vocab_from_features_jsonl,
        )
        from utils.precomputed_multimodal_io import is_jsonl_sidecar_path

        vpath = args.vocab_from_features
        if is_jsonl_sidecar_path(vpath):
            log.info("正在从 JSONL 流式构建词表: %s", vpath)
            vocab = collect_vocab_from_features_jsonl(vpath)
            vocab_src = "jsonl"
        else:
            log.info("正在从 JSON 加载并构建词表: %s", vpath)
            vocab = collect_vocab_from_features_file(vpath)
            vocab_src = "json"
    else:
        vocab = get_default_vocab()
        vocab_src = "default"
    vocab_size = max(len(vocab), args.vocab_size)
    log.info(
        "训练参数: device=%s epochs=%s batch_size=%s lr=%s synthetic=%s index=%s "
        "num_pairs=%s num_workers=%s pin_memory=%s vocab_source=%s vocab_size=%s "
        "precomputed_features=%s max_seq_len=%s max_graph_nodes=%s max_dfg_nodes=%s use_dfg=%s epoch_cleanup=%s "
        "memory_cache_max_items=%s lsir_cache_max_binaries=%s",
        device,
        args.epochs,
        args.batch_size,
        args.lr,
        args.synthetic,
        index_path,
        args.num_pairs,
        num_workers,
        use_pin_memory,
        vocab_src,
        vocab_size,
        args.precomputed_features,
        args.max_seq_len,
        args.max_graph_nodes,
        args.max_dfg_nodes,
        args.use_dfg,
        (not args.no_epoch_cleanup),
        args.memory_cache_max_items,
        args.lsir_cache_max_binaries,
    )
    model = MultiModalFusionModel(
        pcode_vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_transformer_layers=args.num_transformer_layers,
        output_dim=args.output_dim,
        use_dfg=args.use_dfg,
    ).to(device)

    if args.init_weights and os.path.isfile(args.init_weights):
        from features.models.multimodal_fusion import parse_multimodal_checkpoint

        try:
            raw = torch.load(args.init_weights, map_location=device, weights_only=False)
        except TypeError:
            raw = torch.load(args.init_weights, map_location=device)
        sd, _meta = parse_multimodal_checkpoint(raw)
        incomp = model.load_state_dict(sd, strict=False)
        log.info(
            "已加载 init_weights=%s missing_keys=%d unexpected_keys=%d",
            args.init_weights,
            len(incomp.missing_keys),
            len(incomp.unexpected_keys),
        )

    use_fixed_pairs = False
    if args.synthetic:
        from features.dataset import PairwiseSyntheticDataset
        syn_path = args.synthetic_file or os.path.join(PROJECT_ROOT, "data", "synthetic_pairs.json")
        log.info("使用合成数据: %s", syn_path)
        dataset = PairwiseSyntheticDataset(syn_path, num_pairs=args.num_pairs, seed=seed)
    else:
        from features.dataset import PairwiseFunctionDataset
        use_disk_cache = not args.no_disk_cache
        log.info("初始化 PairwiseFunctionDataset: index=%s cache_dir=%s use_disk_cache=%s", index_path, cache_dir, use_disk_cache)
        lazy_reuse_fp = (num_workers == 0) and (not args.no_precomputed_lazy_reuse_fp)
        use_fixed_pairs = (
            bool(args.precomputed_features)
            and (not args.synthetic)
            and (not args.no_fixed_pairs_per_epoch)
        )
        dataset = PairwiseFunctionDataset(
            index_path,
            project_root=PROJECT_ROOT,
            cache_dir=cache_dir,
            num_pairs=args.num_pairs,
            use_disk_cache=use_disk_cache,
            precomputed_features_path=args.precomputed_features,
            memory_cache_max_items=args.memory_cache_max_items,
            lsir_cache_max_binaries=args.lsir_cache_max_binaries,
            seed=seed,
            pairing_mode=str(args.pairing_mode),
            max_cfg_node_ratio=float(args.max_cfg_node_ratio),
            prefer_cross_variant_positive=bool(args.prefer_cross_variant),
            graph_similar_max_delta=int(args.graph_similar_max_delta),
            precomputed_lazy_reuse_read_file_handle=lazy_reuse_fp,
            precomputed_lazy_log_first_n=max(0, int(args.precomputed_lazy_log_first_n)),
            fixed_pairs_per_epoch=use_fixed_pairs,
        )

    n = len(dataset)
    split = max(1, int(0.9 * n))
    train_ds = torch.utils.data.Subset(dataset, range(split))
    val_ds = torch.utils.data.Subset(dataset, range(split, n))

    g = torch.Generator()
    g.manual_seed(seed)
    _dl_kw: dict = {}
    if num_workers > 0:
        _dl_kw["persistent_workers"] = True
        _dl_kw["prefetch_factor"] = max(1, int(args.dataloader_prefetch_factor))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_pairs,
        generator=g,
        pin_memory=use_pin_memory,
        **_dl_kw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_pairs,
        pin_memory=use_pin_memory,
        **_dl_kw,
    )
    try:
        log.info(
            "数据就绪: 逻辑样本数=%s | 每 epoch 训练 batch≈%s | 验证 batch≈%s",
            n,
            len(train_loader),
            len(val_loader),
        )
    except TypeError:
        log.info("数据就绪: 逻辑样本数=%s（DataLoader 长度未知）", n)
    if use_fixed_pairs:
        log.info(
            "已启用每 epoch 固定 num_pairs 站点对（fixed_pairs_per_epoch），将提升 JSONL/内存缓存命中；"
            "每个 epoch 开始时重新采样对。"
        )
    if num_workers > 0:
        log.info(
            "DataLoader num_workers=%s persistent_workers=True prefetch_factor=%s（若 RAM 紧张可改 --num-workers 0）",
            num_workers,
            _dl_kw.get("prefetch_factor", "n/a"),
        )

    loss_fn = ContrastiveLoss(margin=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    step_fn = _make_step_fn(
        vocab,
        device,
        loss_fn,
        max_seq_len=max(1, int(args.max_seq_len)),
        max_graph_nodes=max(1, int(args.max_graph_nodes)),
        max_dfg_nodes=max(1, int(args.max_dfg_nodes)),
    )

    # 13.1：默认启用 TensorBoard；--no-tb 显式禁用
    tb_writer = None
    if not args.no_tb:
        tb_dir = args.tb_dir
        if tb_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tb_dir = os.path.join(PROJECT_ROOT, "output", "tensorboard", f"multimodal_{timestamp}")
            try:
                from utils.retention import enforce_dir_size_limit, enforce_subdir_retention

                tb_root = os.path.join(PROJECT_ROOT, "output", "tensorboard")
                os.makedirs(tb_root, exist_ok=True)
                enforce_subdir_retention(
                    tb_root,
                    keep_recent_dirs=args.tb_keep_runs,
                    name_prefix="multimodal_",
                )
                enforce_dir_size_limit(
                    tb_root,
                    max_total_bytes=int(args.max_tb_mb) * 1024 * 1024,
                    keep_recent=50,
                )
            except Exception:
                pass
        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(tb_dir)
            tb_writer.add_hparams({"seed": seed}, {})
            log.info("TensorBoard 已启用: %s", tb_dir)
        except ImportError:
            log.warning("tensorboard 未安装，跳过 TensorBoard 日志。运行: pip install tensorboard")

    _retrieval_state: dict = {}

    def _on_epoch_end(epoch: int, train_loss: float, val_loss: float, val_acc: float) -> None:
        if wandb_run is not None:
            try:
                import wandb
                wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}, step=epoch)
            except ImportError:
                pass
        rdir = args.retrieval_val_dir
        every = max(0, int(args.retrieval_val_every))
        if not rdir or every <= 0 or (epoch + 1) % every != 0:
            return
        rdir_abs = os.path.abspath(rdir)
        gt_primary = os.path.join(rdir_abs, args.retrieval_val_gt)
        gt_fallback = os.path.join(rdir_abs, "ground_truth.json")
        gt_path = gt_primary if os.path.isfile(gt_primary) else gt_fallback
        lib_p = os.path.join(rdir_abs, "library_features.json")
        qry_p = os.path.join(rdir_abs, "query_features.json")
        if not all(os.path.isfile(p) for p in (gt_path, lib_p, qry_p)):
            return
        if not _retrieval_state:
            import json as _json

            with open(gt_path, encoding="utf-8") as f:
                gt_full = _json.load(f)
            with open(lib_p, encoding="utf-8") as f:
                lib_mm = _json.load(f)
            with open(qry_p, encoding="utf-8") as f:
                q_mm = _json.load(f)
            if not isinstance(gt_full, dict) or not isinstance(lib_mm, dict) or not isinstance(q_mm, dict):
                return
            sub = max(0, int(args.retrieval_val_subsample))
            if sub > 0:
                keys = list(gt_full.keys())[:sub]
                gt_use = {k: gt_full[k] for k in keys if k in gt_full}
            else:
                gt_use = gt_full
            _retrieval_state["gt"] = gt_use
            _retrieval_state["lib"] = lib_mm
            _retrieval_state["qry"] = q_mm
        try:
            from features.validation_retrieval import multimodal_retrieval_recall_at_1

            model.eval()
            rec, n_ev, n_ok = multimodal_retrieval_recall_at_1(
                model,
                vocab,
                device,
                _retrieval_state["lib"],
                _retrieval_state["qry"],
                _retrieval_state["gt"],
                max_seq_len=max(1, int(args.max_seq_len)),
                max_graph_nodes=max(1, int(args.max_graph_nodes)),
                max_dfg_nodes=max(1, int(args.max_dfg_nodes)),
            )
            log.info(
                "检索验证 epoch=%d Recall@1=%.4f (%d/%d) gt=%s",
                epoch + 1,
                rec,
                n_ok,
                n_ev,
                gt_path,
            )
            if tb_writer is not None:
                tb_writer.add_scalar("retrieval/recall_at_1", rec, epoch)
        except Exception as e:
            log.warning("检索验证跳过: %s", e)
        finally:
            model.train()

    ckpt_meta = {
        "use_dfg": bool(args.use_dfg),
        "pcode_vocab_size": int(vocab_size),
        "max_dfg_nodes": int(args.max_dfg_nodes),
        "max_graph_nodes": int(args.max_graph_nodes),
        "max_seq_len": int(args.max_seq_len),
    }
    def _unwrap_underlying_dataset(subset_ds):
        d = subset_ds
        while hasattr(d, "dataset"):
            d = d.dataset
        return d

    def _on_epoch_begin_pairwise(_epoch: int) -> None:
        d0 = _unwrap_underlying_dataset(train_ds)
        reg = getattr(d0, "regenerate_epoch_pairs", None)
        if callable(reg):
            reg()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        save_path=save_path,
        step_fn=step_fn,
        tb_writer=tb_writer,
        checkpoint_meta=ckpt_meta,
    )
    show_progress = not args.no_progress_bar
    if show_progress:
        log.info("训练中：每个 epoch 内会显示 train/val 进度（安装 tqdm 时为进度条）。")
    if not args.synthetic and args.precomputed_features:
        log.info(
            "冷启动说明：首个训练 batch 需完成随机成对采样、JSONL 懒读解析与首次 GPU 前向；"
            "进度条可能在首个 batch 结束前停留在起始位置，属正常现象。"
        )
    if args.warmup_first_batch:
        import time as _time

        _on_epoch_begin_pairwise(-1)
        log.info("warmup-first-batch: 预跑一个训练 batch（不计入 tqdm 进度）…")
        w0 = _time.perf_counter()
        model.train()
        try:
            warm_batch = next(iter(train_loader))
        except StopIteration:
            log.warning("warmup-first-batch: train_loader 为空，跳过。")
        else:
            loss_w, correct_w, count_w = step_fn(warm_batch, model, loss_fn)
            if count_w > 0:
                optimizer.zero_grad()
                loss_w.backward()
                optimizer.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            log.info(
                "warmup-first-batch: 完成 count=%s 耗时 %.2fs",
                count_w,
                _time.perf_counter() - w0,
            )
    trainer.fit(
        args.epochs,
        on_epoch_end=_on_epoch_end,
        on_epoch_begin=_on_epoch_begin_pairwise,
        progress_bar=show_progress,
        log_batches_every=max(1, int(args.progress_log_every)),
        cleanup_every_epoch=(not args.no_epoch_cleanup),
    )
    print(f"最佳模型已保存至 {save_path}")


if __name__ == "__main__":
    main()
