#!/usr/bin/env python3
"""
SAFE 对比学习训练脚本：_SafeEncoder + 成对对比学习。
支持目标校验：训练后若未达到指定 Recall 指标，自动扩样重训。
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

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

    # 文件滚动日志（只保留最近 backups 份）
    fh = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # 控制台保留关键输出（INFO+）
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)


def _collate_pairs(batch):
    """自定义 collate：保持 dict 列表，不 stacking。"""
    return {
        "feature1": [b["feature1"] for b in batch],
        "feature2": [b["feature2"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
    }


def _make_safe_step_fn(vocab, device, loss_fn, max_len=512, threshold=0.5):
    """构建 SAFE 训练步进函数：safe_tokenize -> _SafeEncoder -> ContrastiveLoss。"""

    def step_fn(batch, model, _loss_fn):
        from features.baselines.safe import safe_tokenize

        f1_list = batch["feature1"] if isinstance(batch["feature1"], list) else [batch["feature1"]]
        f2_list = batch["feature2"] if isinstance(batch["feature2"], list) else [batch["feature2"]]
        labels = batch["label"]
        if torch.is_tensor(labels):
            labels = labels.float().to(device)
        else:
            labels = torch.tensor(labels, dtype=torch.float32, device=device)

        # Tokenize all samples on CPU, then batch-transfer to GPU
        all_ids1, all_pad1, all_ids2, all_pad2, success_indices = [], [], [], [], []
        for idx, (f1, f2) in enumerate(zip(f1_list, f2_list)):
            try:
                ids1, pad1 = safe_tokenize(f1, vocab, max_len=max_len)
                ids2, pad2 = safe_tokenize(f2, vocab, max_len=max_len)
                all_ids1.append(ids1)
                all_pad1.append(pad1)
                all_ids2.append(ids2)
                all_pad2.append(pad2)
                success_indices.append(idx)
            except Exception:
                continue

        if not all_ids1:
            return torch.tensor(0.0, device=device, requires_grad=True), 0, 0

        # Stack into batch tensors and move to GPU once
        t1 = torch.tensor(all_ids1, dtype=torch.long, device=device)
        p1 = torch.tensor(all_pad1, dtype=torch.bool, device=device)
        t2 = torch.tensor(all_ids2, dtype=torch.long, device=device)
        p2 = torch.tensor(all_pad2, dtype=torch.bool, device=device)

        v1 = model(t1, p1)
        v2 = model(t2, p2)
        if v1.dim() == 1:
            v1 = v1.unsqueeze(0)
        if v2.dim() == 1:
            v2 = v2.unsqueeze(0)

        n = v1.size(0)
        labels = labels[success_indices].to(device)
        loss = _loss_fn(v1, v2, labels)
        cos_sim = torch.nn.functional.cosine_similarity(v1, v2, dim=1)
        pred_sim = (cos_sim > threshold).float()
        correct = (pred_sim == labels).float().sum().item()
        return loss, int(correct), n

    return step_fn


def _run_validation(
    save_path: str,
    library_features_path: str,
    query_features_path: str,
    ground_truth_path: str,
    coarse_k: int,
    rerank_model_path: str,
    *,
    val_batch_size: int = 128,
    val_rerank_batch_size: int = 1024,
    val_subsample: int = 0,
    val_rerank_k: int = 0,
    seed: int = 42,
    temp_dir: Optional[str] = None,
    max_temp_mb: int = 128,
) -> tuple[float, float]:
    """
    训练后校验：重建库嵌入、运行两阶段评估，返回 (coarse_recall, recall_at_1)。
    复用 build_embeddings_db._process_features_file 与 TwoStagePipeline。
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_emb_db",
        os.path.join(PROJECT_ROOT, "scripts", "build_embeddings_db.py"),
    )
    bld = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bld)
    from matcher.two_stage import TwoStagePipeline

    # 临时文件目录：默认 output/tmp/train_safe
    if temp_dir is None:
        temp_dir = os.path.join(PROJECT_ROOT, "output", "tmp", "train_safe")
    os.makedirs(temp_dir, exist_ok=True)
    embeddings_path = os.path.join(temp_dir, "library_safe_embeddings.json")

    all_emb = bld._process_features_file(
        library_features_path,
        model_path=save_path,
    )
    with open(embeddings_path, "w", encoding="utf-8") as f:
        json.dump({"functions": all_emb}, f, indent=2, ensure_ascii=False)

    pipeline = TwoStagePipeline(
        library_safe_embeddings_path=embeddings_path,
        library_features_path=library_features_path,
        query_features_path=query_features_path,
        coarse_k=coarse_k,
        rerank_model_path=rerank_model_path,
        safe_model_path=save_path,
    )

    with open(ground_truth_path, encoding="utf-8") as f:
        ground_truth = json.load(f)
    if not isinstance(ground_truth, dict):
        raise ValueError("ground_truth 应为 {query_id: [positive_id, ...]}")

    # 仅提取 query_features 的顶层 key（检查 qid 是否存在），不加载整个 value 到内存
    try:
        from scripts.sidechain.build_embeddings_db import _iter_json_object_records

        with open(query_features_path, "rb") as _qf:
            query_feature_ids = {fid for fid, _ in _iter_json_object_records(_qf)}
    except Exception:
        with open(query_features_path, encoding="utf-8") as f:
            query_feature_ids = set(json.load(f).keys())

    valid_ids = [qid for qid in ground_truth if qid in query_feature_ids]
    if not valid_ids:
        return 0.0, 0.0

    subsample = val_subsample if val_subsample and val_subsample > 0 else None
    rerank_k_opt = val_rerank_k if val_rerank_k and val_rerank_k > 0 else None
    coarse_recall, recall_at_1 = pipeline.evaluate(
        valid_ids,
        ground_truth,
        batch_size=val_batch_size,
        rerank_batch_size=val_rerank_batch_size,
        subsample=subsample,
        rerank_k=rerank_k_opt,
        seed=seed,
        progress_every=10,
    )
    # 清理临时目录总量，默认上限 128MB
    try:
        from utils.retention import enforce_dir_size_limit

        enforce_dir_size_limit(
            temp_dir,
            max_total_bytes=int(max_temp_mb) * 1024 * 1024,
            keep_recent=10,
        )
    except Exception:
        pass
    return coarse_recall, recall_at_1


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 SAFE 孪生网络（对比学习）")
    parser.add_argument("--index-file", default=None, help="binkit_functions.json 或两阶段合并索引")
    parser.add_argument("--synthetic", action="store_true", help="使用合成数据")
    parser.add_argument("--synthetic-file", default=None, help="合成数据 JSON 路径")
    parser.add_argument(
        "--vocab-from-features",
        default=None,
        help="从特征文件构建 vocab：.json 为整表加载（大库易 OOM）；JSONL 侧车流式扫描（推荐与大库联用）",
    )
    parser.add_argument("--epochs", type=int, default=10, help="训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader worker 数（默认 0，降低 OOM 风险）"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--save-path", default=None, help="模型保存路径")
    parser.add_argument("--cache-dir", default=None, help="特征缓存目录")
    parser.add_argument(
        "--precomputed-features",
        default=None,
        help="可选：预计算特征文件（{function_id: multimodal}），命中时优先读取",
    )
    parser.add_argument("--tb-dir", default=None, help="TensorBoard 目录")
    parser.add_argument("--no-tb", action="store_true", help="禁用 TensorBoard")
    parser.add_argument("--num-pairs", type=int, default=2000, help="数据集对数（真实数据时）")
    parser.add_argument("--embed-dim", type=int, default=64, help="嵌入维度")
    parser.add_argument("--output-dim", type=int, default=128, help="输出嵌入维度")
    parser.add_argument("--seed", type=int, default=145, help="随机种子")
    parser.add_argument(
        "--use-amp", action="store_true", help="启用混合精度训练 (AMP)，降低显存占用"
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="梯度累积步数（等效 batch_size = --batch-size * --accumulation-steps）",
    )
    parser.add_argument("--use-disk-cache", action="store_true", help="启用特征磁盘缓存")
    parser.add_argument("--no-disk-cache", action="store_true", help="禁用特征磁盘缓存")
    # 目标校验
    parser.add_argument(
        "--target-coarse-recall",
        type=float,
        default=0.50,
        help="粗筛 Recall@K 下限",
    )
    parser.add_argument(
        "--target-recall-at-1",
        type=float,
        default=0.45,
        help="两阶段 Recall@1 下限",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="未达标时最大重训次数",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="跳过目标校验，仅训练并保存",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="两阶段数据目录（启用校验时用于推断路径）",
    )
    parser.add_argument(
        "--coarse-k",
        type=int,
        default=100,
        help="粗筛 Top-K（与 eval_two_stage 一致）",
    )
    parser.add_argument("--val-batch-size", type=int, default=128, help="验证 query 批量大小")
    parser.add_argument(
        "--val-rerank-batch-size",
        type=int,
        default=1024,
        help="验证精排 batch 大小（候选 forward）",
    )
    parser.add_argument(
        "--val-subsample", type=int, default=0, help="验证抽样 query 数（0 表示不抽样）"
    )
    parser.add_argument(
        "--val-rerank-k", type=int, default=0, help="验证时精排只取前 K 个候选（0 表示用 coarse_k）"
    )
    parser.add_argument(
        "--max-temp-mb", type=int, default=128, help="训练/验证临时目录最大总大小（MB）"
    )
    parser.add_argument(
        "--temp-dir", default=None, help="训练/验证临时目录（默认 output/tmp/train_safe）"
    )
    parser.add_argument("--max-log-mb", type=int, default=32, help="单个日志文件最大大小（MB）")
    parser.add_argument("--log-backups", type=int, default=5, help="日志滚动备份份数")
    parser.add_argument("--log-level", default="INFO", help="日志级别（DEBUG/INFO/WARNING/ERROR）")
    parser.add_argument(
        "--max-tb-mb", type=int, default=128, help="TensorBoard 根目录最大总大小（MB）"
    )
    parser.add_argument(
        "--tb-keep-runs", type=int, default=5, help="TensorBoard 仅保留最近 N 个 run 目录"
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="关闭 batch 级进度（tqdm 或周期性日志），仅保留每 epoch 汇总",
    )
    parser.add_argument(
        "--progress-log-every",
        type=int,
        default=20,
        help="无 tqdm 时每隔多少个训练 batch 打印一行进度（需未加 --no-progress-bar）",
    )
    args = parser.parse_args()

    seed = args.seed
    from experiment_meta import set_deterministic

    set_deterministic(seed)

    default_save = os.path.join(PROJECT_ROOT, "output", "safe_best_model.pt")
    default_cache = os.path.join(PROJECT_ROOT, "data", "features_cache")
    default_index = os.path.join(PROJECT_ROOT, "data", "binkit_functions.json")
    default_data_dir = os.path.join(PROJECT_ROOT, "data", "two_stage")

    save_path = args.save_path or default_save
    cache_dir = args.cache_dir or default_cache
    index_path = args.index_file or default_index
    data_dir = args.data_dir or default_data_dir

    # 自动发现 .training.jsonl（未指定 --precomputed-features 时）
    if not args.precomputed_features:
        inferred = os.path.splitext(index_path)[0] + ".training.jsonl"
        if os.path.isfile(inferred):
            print(f"自动发现训练特征文件: {inferred}")
            args.precomputed_features = inferred

    _setup_rotating_logging(
        log_dir=os.path.join(PROJECT_ROOT, "output", "logs"),
        log_name="train_safe.log",
        max_mb=args.max_log_mb,
        backups=args.log_backups,
        level=args.log_level,
    )
    log = logging.getLogger("train_safe")

    # 校验时需两阶段数据路径
    if not args.skip_validation and not args.synthetic:
        library_features = os.path.join(data_dir, "library_features.json")
        query_features = os.path.join(data_dir, "query_features.json")
        ground_truth = os.path.join(data_dir, "ground_truth.json")
        for p, name in [
            (library_features, "library_features"),
            (query_features, "query_features"),
            (ground_truth, "ground_truth"),
        ]:
            if not os.path.isfile(p):
                print(f"错误: 启用校验需存在 {name}: {p}", file=sys.stderr)
                sys.exit(1)

    # Vocab：优先从特征文件构建（JSONL 流式，避免大 library_features.json 整文件进内存）
    if args.vocab_from_features and os.path.isfile(args.vocab_from_features):
        from features.baselines.safe import (
            collect_vocab_from_features_file,
            collect_vocab_from_features_jsonl,
        )
        from utils.precomputed_multimodal_io import is_jsonl_sidecar_path

        vpath = args.vocab_from_features
        if is_jsonl_sidecar_path(vpath):
            log.info("正在从 JSONL 流式构建词表（大文件可能需扫描整份侧车）: %s", vpath)
            vocab = collect_vocab_from_features_jsonl(vpath)
        else:
            log.info("正在从 JSON 加载并构建词表: %s", vpath)
            vocab = collect_vocab_from_features_file(vpath)
    else:
        from features.models.multimodal_fusion import get_default_vocab

        log.info("使用默认多模态词表（未指定 --vocab-from-features）")
        vocab = get_default_vocab()
    vocab_size = max(len(vocab), 256)
    log.info("词表就绪: vocab_size=%s", vocab_size)

    from features.baselines.safe import _SafeEncoder, safe_save_model
    from features.losses import ContrastiveLoss
    from features.trainer import Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = device.type == "cuda"
    num_workers = max(0, int(args.num_workers))
    num_pairs = args.num_pairs
    epochs = args.epochs

    rerank_model_path = os.path.join(PROJECT_ROOT, "output", "best_model.pth")

    for retry in range(args.max_retries):
        # 每次 retry 使用不同 seed，确保模型初始化权重不同
        if retry > 0:
            set_deterministic(seed + retry)
        log.info(
            "开始本轮训练 retry=%s/%s num_pairs=%s epochs=%s batch_size=%s device=%s seed=%s",
            retry + 1,
            args.max_retries,
            num_pairs,
            epochs,
            args.batch_size,
            device,
            seed + retry,
        )
        model = _SafeEncoder(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            output_dim=args.output_dim,
        ).to(device)

        if args.synthetic:
            from features.dataset import PairwiseSyntheticDataset

            syn_path = args.synthetic_file or os.path.join(
                PROJECT_ROOT, "data", "synthetic_pairs.json"
            )
            dataset = PairwiseSyntheticDataset(syn_path, num_pairs=num_pairs, seed=seed)
        else:
            from features.dataset import PairwiseFunctionDataset

            use_disk_cache = not args.no_disk_cache
            log.info("正在初始化 PairwiseFunctionDataset（索引加载 / JSONL 侧车建索引可能较慢）…")
            dataset = PairwiseFunctionDataset(
                index_path,
                project_root=PROJECT_ROOT,
                cache_dir=cache_dir,
                num_pairs=num_pairs,
                use_disk_cache=use_disk_cache,
                precomputed_features_path=args.precomputed_features,
                seed=seed,
                precomputed_lazy_reuse_read_file_handle=(num_workers == 0),
            )

        n = len(dataset)
        split = max(1, int(0.9 * n))
        train_ds = torch.utils.data.Subset(dataset, range(split))
        val_ds = torch.utils.data.Subset(dataset, range(split, n))

        g = torch.Generator()
        g.manual_seed(seed + retry)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=_collate_pairs,
            generator=g,
            pin_memory=use_pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate_pairs,
            pin_memory=use_pin_memory,
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

        loss_fn = ContrastiveLoss(margin=0.5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        step_fn = _make_safe_step_fn(vocab, device, loss_fn)

        tb_writer = None
        if not args.no_tb:
            tb_dir = args.tb_dir
            if tb_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                tb_dir = os.path.join(PROJECT_ROOT, "output", "tensorboard", f"safe_{timestamp}")
            try:
                from torch.utils.tensorboard import SummaryWriter

                # TensorBoard 保留策略：只保留近期 run，并限制根目录总大小
                try:
                    from utils.retention import enforce_dir_size_limit, enforce_subdir_retention

                    tb_root = os.path.join(PROJECT_ROOT, "output", "tensorboard")
                    os.makedirs(tb_root, exist_ok=True)
                    enforce_subdir_retention(
                        tb_root, keep_recent_dirs=args.tb_keep_runs, name_prefix="safe_"
                    )
                    enforce_dir_size_limit(
                        tb_root,
                        max_total_bytes=int(args.max_tb_mb) * 1024 * 1024,
                        keep_recent=50,
                    )
                except Exception:
                    pass
                os.makedirs(tb_dir, exist_ok=True)
                tb_writer = SummaryWriter(tb_dir)
            except ImportError:
                pass

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
            use_amp=args.use_amp,
            accumulation_steps=max(1, int(args.accumulation_steps)),
        )
        show_progress = not args.no_progress_bar
        if show_progress:
            log.info("训练中：每个 epoch 内会显示 train/val 进度（安装 tqdm 时为进度条）。")
        trainer.fit(
            epochs,
            progress_bar=show_progress,
            log_batches_every=max(1, int(args.progress_log_every)),
        )
        safe_save_model(
            model, vocab, save_path, embed_dim=args.embed_dim, output_dim=args.output_dim
        )

        # 写入实验 metadata
        from experiment_meta import save_metadata

        save_metadata(save_path, args)

        if args.skip_validation or args.synthetic:
            print(f"最佳模型已保存至 {save_path}（已跳过校验）")
            return

        # 目标校验
        library_features = os.path.join(data_dir, "library_features.json")
        query_features_path = os.path.join(data_dir, "query_features.json")
        ground_truth_path = os.path.join(data_dir, "ground_truth.json")

        coarse_recall, recall_at_1 = _run_validation(
            save_path=save_path,
            library_features_path=library_features,
            query_features_path=query_features_path,
            ground_truth_path=ground_truth_path,
            coarse_k=args.coarse_k,
            rerank_model_path=rerank_model_path,
            val_batch_size=args.val_batch_size,
            val_rerank_batch_size=args.val_rerank_batch_size,
            val_subsample=args.val_subsample,
            val_rerank_k=args.val_rerank_k,
            seed=seed,
            temp_dir=args.temp_dir,
            max_temp_mb=args.max_temp_mb,
        )

        print(
            f"校验 retry={retry + 1}: coarse_recall={coarse_recall:.4f}, recall_at_1={recall_at_1:.4f}"
        )

        if coarse_recall >= args.target_coarse_recall and recall_at_1 >= args.target_recall_at_1:
            print(f"达标，最佳模型已保存至 {save_path}")
            return

        if retry < args.max_retries - 1:
            num_pairs = int(num_pairs * 1.5)
            epochs += 5
            print(f"未达标，扩样重训: num_pairs={num_pairs}, epochs={epochs}")

    print(
        f"错误: 经 {args.max_retries} 次重训仍未达标。"
        f"最后一次: coarse_recall={coarse_recall:.4f} (目标>={args.target_coarse_recall}), "
        f"recall_at_1={recall_at_1:.4f} (目标>={args.target_recall_at_1})。"
        f"建议: 增加 --num-pairs、放宽 --target-* 或扩充两阶段数据。",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
