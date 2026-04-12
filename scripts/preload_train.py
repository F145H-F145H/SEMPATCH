#!/usr/bin/env python3
"""预加载训练：一次性将 JSONL 特征全部载入内存列表，训练循环零 IO。"""
import argparse
import gc
import json
import logging
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch
import torch.nn.functional as F

from features.losses import ContrastiveLoss
from features.models.multimodal_fusion import (
    MultiModalFusionModel,
    get_default_vocab,
    tensorize_multimodal_many,
)
from utils.binkit_provenance import parse_binary_provenance
from utils.training_function_filter import strip_linker_suffix


def _load_all_features(jsonl_path: str) -> List[Dict[str, Any]]:
    """一次性读取 JSONL 全部特征到列表。"""
    features = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            fid = obj.get("function_id", "")
            mm = obj.get("multimodal")
            if isinstance(fid, str) and isinstance(mm, dict):
                features.append({"function_id": fid, "multimodal": mm})
    return features


def _fid_to_binary_entry(fid: str) -> Tuple[str, str]:
    """从 function_id 提取 (binary_rel, entry)。"""
    parts = fid.rsplit("|", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return fid, ""


def _make_index(index_path: str) -> List[Dict[str, Any]]:
    with open(index_path, encoding="utf-8") as f:
        return json.load(f)


def _build_positive_map(
    features: List[Dict[str, Any]],
) -> Dict[str, List[int]]:
    """按函数名分组，返回 name -> [indices into features list]。"""
    name_to_indices: Dict[str, List[int]] = {}
    for i, feat in enumerate(features):
        fid = feat["function_id"]
        parts = fid.rsplit("|", 1)
        if len(parts) != 2:
            continue
        binary_rel = parts[0]
        # 从 index 文件推断函数名 → 这里用 function_id 的第一段作为近似
        # 更精确需要从 index 文件匹配，但快速验证够用
    return name_to_indices


def _build_pairs_from_index(
    index_path: str,
    features: List[Dict[str, Any]],
    num_pairs: int,
    positive_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[int, int, int]], List[Dict[str, Any]]]:
    """
    基于 index 文件的函数名，生成 (idx1, idx2, label) 对。
    返回 (pairs, valid_features)。
    """
    rng = random.Random(seed)

    # 从 index 构建 binary|entry -> 函数名映射
    index_data = _make_index(index_path)
    entry_to_name: Dict[str, str] = {}
    for item in index_data:
        binary = item.get("binary", "")
        for f in item.get("functions", []):
            entry = f.get("entry", "")
            name = f.get("name", "")
            if entry and name:
                entry_to_name[f"{binary}|{entry}"] = name

    # 从 features 构建 name -> indices 映射
    name_to_indices: Dict[str, List[int]] = {}
    for i, feat in enumerate(features):
        fid = feat["function_id"]
        name = entry_to_name.get(fid, "")
        if not name:
            # 回退：用 function_id 本身作为键
            name = fid
        name_to_indices.setdefault(name, []).append(i)

    positive_names = [n for n, idxs in name_to_indices.items() if len(idxs) >= 2]
    all_indices = list(range(len(features)))

    pairs = []
    for _ in range(num_pairs):
        if rng.random() < positive_ratio and positive_names:
            # 正对：同一函数名的不同实例
            name = rng.choice(positive_names)
            idxs = name_to_indices[name]
            a, b = rng.sample(idxs, 2)
            pairs.append((a, b, 1))
        else:
            # 负对：随机两个不同函数
            a, b = rng.sample(all_indices, 2)
            tries = 0
            while features[a]["function_id"].rsplit("|", 1)[1] == features[b]["function_id"].rsplit("|", 1)[1] and tries < 20:
                a, b = rng.sample(all_indices, 2)
                tries += 1
            pairs.append((a, b, 0))

    return pairs, features


def main():
    parser = argparse.ArgumentParser(description="预加载快速训练")
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
    args = parser.parse_args()

    from experiment_meta import set_deterministic
    set_deterministic(args.seed)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("preload_train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # 1. 一次性加载全部特征
    log.info("加载特征: %s", args.precomputed_features)
    features = _load_all_features(args.precomputed_features)
    log.info("已加载 %d 个函数特征到内存", len(features))

    # 2. 构建 vocab
    try:
        from features.baselines.safe import collect_vocab_from_features_jsonl
        vocab = collect_vocab_from_features_jsonl(args.precomputed_features)
    except Exception:
        vocab = get_default_vocab()
    vocab_size = max(len(vocab), 256)
    log.info("Vocab size: %d", vocab_size)

    # 3. 生成训练对
    all_pairs, all_features = _build_pairs_from_index(
        args.index_file, features, args.num_pairs, 0.5, args.seed
    )
    rng = random.Random(args.seed)
    rng.shuffle(all_pairs)
    split = max(1, int(0.9 * len(all_pairs)))
    train_pairs = all_pairs[:split]
    val_pairs = all_pairs[split:]
    log.info("训练对: %d, 验证对: %d", len(train_pairs), len(val_pairs))

    # 4. 模型
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

    loss_fn = ContrastiveLoss(margin=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 5. 训练循环（纯内存，零 IO）
    def run_batch(pairs_batch):
        """从内存列表取特征，直接 tensorize → forward → loss。"""
        f1_list = [all_features[a]["multimodal"] for a, _, _ in pairs_batch]
        f2_list = [all_features[b]["multimodal"] for _, b, _ in pairs_batch]
        labels = torch.tensor(
            [float(l) for _, _, l in pairs_batch], dtype=torch.float32, device=device
        )

        batch1 = tensorize_multimodal_many(
            f1_list, vocab, device=device,
            max_seq_len=args.max_seq_len, max_graph_nodes=args.max_graph_nodes,
            max_dfg_nodes=args.max_dfg_nodes, pcode_vocab_size=vocab_size,
        )
        batch2 = tensorize_multimodal_many(
            f2_list, vocab, device=device,
            max_seq_len=args.max_seq_len, max_graph_nodes=args.max_graph_nodes,
            max_dfg_nodes=args.max_dfg_nodes, pcode_vocab_size=vocab_size,
        )

        v1 = model(*batch1)
        v2 = model(*batch2)
        if v1.dim() == 1:
            v1 = v1.unsqueeze(0)
        if v2.dim() == 1:
            v2 = v2.unsqueeze(0)

        n = v1.size(0)
        loss = loss_fn(v1, v2, labels[:n])
        cos_sim = F.cosine_similarity(v1, v2, dim=1)
        correct = ((cos_sim > 0.5).float() == labels[:n]).float().sum().item()
        return loss, int(correct), n

    best_val_loss = None
    for epoch in range(args.epochs):
        # Shuffle training pairs
        rng.shuffle(train_pairs)

        # Train
        model.train()
        total_loss, total_correct, total_count = 0.0, 0, 0
        for i in range(0, len(train_pairs), args.batch_size):
            batch = train_pairs[i : i + args.batch_size]
            optimizer.zero_grad()
            loss, correct, count = run_batch(batch)
            if count == 0:
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * count
            total_correct += correct
            total_count += count

        train_loss = total_loss / total_count if total_count else 0.0
        train_acc = total_correct / total_count if total_count else 0.0

        # Validate
        model.eval()
        val_loss_sum, val_correct, val_count = 0.0, 0, 0
        with torch.no_grad():
            for i in range(0, len(val_pairs), args.batch_size):
                batch = val_pairs[i : i + args.batch_size]
                loss, correct, count = run_batch(batch)
                if count == 0:
                    continue
                val_loss_sum += loss.item() * count
                val_correct += correct
                val_count += count

        val_loss = val_loss_sum / val_count if val_count else 0.0
        val_acc = val_correct / val_count if val_count else 0.0

        log.info(
            "Epoch %d/%d  train_loss=%.4f  train_acc=%.4f  val_loss=%.4f  val_acc=%.4f",
            epoch + 1, args.epochs, train_loss, train_acc, val_loss, val_acc,
        )

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.save_path)

    log.info("Done. Best val_loss=%.4f, model: %s", best_val_loss, args.save_path)


if __name__ == "__main__":
    main()
