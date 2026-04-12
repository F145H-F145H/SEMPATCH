#!/usr/bin/env python3
"""快速训练 wrapper：保留特征缓存，只做必要的内存清理。"""
import argparse
import gc
import logging
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch
from torch.utils.data import DataLoader

from features.losses import ContrastiveLoss
from features.models.multimodal_fusion import MultiModalFusionModel, get_default_vocab
from features.trainer import Trainer


def _collate_pairs(batch):
    return {
        "feature1": [b["feature1"] for b in batch],
        "feature2": [b["feature2"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
    }


def _make_step_fn(vocab, device, loss_fn, max_seq_len, max_graph_nodes, max_dfg_nodes, vocab_size):
    def step_fn(batch, model, _loss_fn):
        from features.models.multimodal_fusion import tensorize_multimodal_many

        f1_list = batch["feature1"] if isinstance(batch["feature1"], list) else [batch["feature1"]]
        f2_list = batch["feature2"] if isinstance(batch["feature2"], list) else [batch["feature2"]]
        labels = batch["label"]
        if torch.is_tensor(labels):
            labels = labels.float().to(device)
        else:
            labels = torch.tensor(labels, dtype=torch.float32, device=device)

        batch1 = tensorize_multimodal_many(
            f1_list, vocab, device=device,
            max_seq_len=max_seq_len, max_graph_nodes=max_graph_nodes,
            max_dfg_nodes=max_dfg_nodes, pcode_vocab_size=vocab_size,
        )
        batch2 = tensorize_multimodal_many(
            f2_list, vocab, device=device,
            max_seq_len=max_seq_len, max_graph_nodes=max_graph_nodes,
            max_dfg_nodes=max_dfg_nodes, pcode_vocab_size=vocab_size,
        )
        v1 = model(*batch1)
        v2 = model(*batch2)
        if v1.dim() == 1:
            v1 = v1.unsqueeze(0)
        if v2.dim() == 1:
            v2 = v2.unsqueeze(0)
        n = v1.size(0)
        loss = _loss_fn(v1, v2, labels[:n])
        cos_sim = torch.nn.functional.cosine_similarity(v1, v2, dim=1)
        pred_sim = (cos_sim > 0.5).float()
        correct = (pred_sim == labels[:n]).float().sum().item()
        return loss, int(correct), n

    return step_fn


def main():
    parser = argparse.ArgumentParser(description="快速训练 MultiModalFusionModel")
    parser.add_argument("--index-file", required=True)
    parser.add_argument("--precomputed-features", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-pairs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-graph-nodes", type=int, default=64)
    parser.add_argument("--max-dfg-nodes", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--save-path", default="output/best_model.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairing-mode", default="binkit_refined")
    parser.add_argument("--memory-cache-max-items", type=int, default=8192,
                        help="特征缓存上限（LRU），控制内存占用")
    parser.add_argument("--use-amp", action="store_true", default=True,
                        help="启用混合精度训练（默认开启）；--no-use-amp 可关闭")
    parser.add_argument("--no-use-amp", action="store_true",
                        help="禁用混合精度训练")
    args = parser.parse_args()

    from experiment_meta import set_deterministic
    set_deterministic(args.seed)

    use_amp = args.use_amp and not args.no_use_amp
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("fast_train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = device.type == "cuda"
    log.info("Device: %s, AMP: %s, pin_memory: %s", device, use_amp, use_pin_memory)

    # 1. 构建 vocab
    try:
        from features.baselines.safe import collect_vocab_from_features_jsonl
        vocab = collect_vocab_from_features_jsonl(args.precomputed_features)
    except Exception:
        vocab = get_default_vocab()
    vocab_size = max(len(vocab), 256)
    log.info("Vocab size: %d", vocab_size)

    # 2. 构建数据集（LRU 缓存自动控制内存）
    from features.dataset import PairwiseFunctionDataset

    dataset = PairwiseFunctionDataset(
        args.index_file,
        project_root=PROJECT_ROOT,
        num_pairs=args.num_pairs,
        precomputed_features_path=args.precomputed_features,
        memory_cache_max_items=args.memory_cache_max_items,
        lsir_cache_max_binaries=0,  # 不需要 LSIR 缓存（预计算特征已有）
        seed=args.seed,
        pairing_mode=args.pairing_mode,
        precomputed_lazy_reuse_read_file_handle=True,
        fixed_pairs_per_epoch=True,
    )

    n = len(dataset)
    split = max(1, int(0.9 * n))
    train_ds = torch.utils.data.Subset(dataset, range(split))
    val_ds = torch.utils.data.Subset(dataset, range(split, n))

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate_pairs, generator=g,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=_collate_pairs,
        pin_memory=use_pin_memory,
    )

    # 3. 模型
    model = MultiModalFusionModel(
        pcode_vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_gnn_layers=2,
        num_transformer_layers=2,
        num_heads=4,
        use_dfg=False,
    ).to(device)
    log.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    # 4. 训练（自定义 epoch 间清理：只清 LSIR + GC，保留特征缓存）
    loss_fn = ContrastiveLoss(margin=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    step_fn = _make_step_fn(
        vocab, device, loss_fn,
        max_seq_len=args.max_seq_len,
        max_graph_nodes=args.max_graph_nodes,
        max_dfg_nodes=args.max_dfg_nodes,
        vocab_size=vocab_size,
    )

    def lightweight_cleanup():
        """只做轻量清理：清 LSIR 缓存 + GC，保留 _memory_cache。"""
        ds = dataset
        while hasattr(ds, "dataset"):
            ds = ds.dataset
        # 只清 LSIR（大对象），不清 memory_cache（特征缓存）
        if hasattr(ds, "_lsir_raw_cache"):
            ds._lsir_raw_cache.clear()
        gc.collect()
        # 不调用 torch.cuda.empty_cache() — 让 PyTorch 自己管理

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        save_path=args.save_path,
        step_fn=step_fn,
        use_amp=use_amp,
    )

    for epoch in range(args.epochs):
        dataset.regenerate_epoch_pairs()
        train_loss = trainer.train_epoch(phase="train", num_epochs=args.epochs, epoch=epoch)
        val_loss, val_acc = trainer.validate(phase="val", num_epochs=args.epochs, epoch=epoch)
        log.info("Epoch %d/%d  train=%.4f  val=%.4f  val_acc=%.4f",
                 epoch + 1, args.epochs, train_loss, val_loss, val_acc)

        if trainer.best_val_loss is None or val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.save_path)

        # 轻量清理（保留特征缓存）
        lightweight_cleanup()

    log.info("Done. Best model: %s", args.save_path)


if __name__ == "__main__":
    main()
