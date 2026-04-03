"""
验证训练流程的随机种子可复现性（Phase 13，约束 31）。
相同 seed 下，数据集与 DataLoader 应产生一致结果。
"""

import os
import random
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def _run_one_epoch_total_loss(loader, model, loss_fn, step_fn, device):
    """跑一个 epoch，返回总 loss（不反传，仅前向）。"""
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            loss, correct, n = step_fn(batch, model, loss_fn)
            total += loss.item() * n
            count += n
    return total / count if count else 0.0


def test_seed_reproducibility():
    """相同 seed 下两次独立构造的数据集与 loader，跑 1 epoch 后 loss 应一致。"""
    from features.dataset import PairwiseSyntheticDataset
    from features.models.multimodal_fusion import MultiModalFusionModel, get_default_vocab, _tensorize_multimodal
    from features.losses import ContrastiveLoss
    from torch.utils.data import DataLoader

    def collate(batch):
        return {
            "feature1": [b["feature1"] for b in batch],
            "feature2": [b["feature2"] for b in batch],
            "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
        }

    def make_step_fn(vocab, device, loss_fn):
        def step_fn(batch, model, _loss_fn):
            from features.models.multimodal_fusion import _tensorize_multimodal
            f1_list = batch["feature1"]
            f2_list = batch["feature2"]
            labels = batch["label"].float().to(device)
            vecs1, vecs2 = [], []
            for f1, f2 in zip(f1_list, f2_list):
                try:
                    t1, j1, n1, e1, p1, d1n, d1e = _tensorize_multimodal(f1, vocab, device=device)
                    t2, j2, n2, e2, p2, d2n, d2e = _tensorize_multimodal(f2, vocab, device=device)
                    v1 = model(t1, j1, n1, e1, p1, dfg_node_features=d1n, dfg_edge_index=d1e)
                    v2 = model(t2, j2, n2, e2, p2, dfg_node_features=d2n, dfg_edge_index=d2e)
                    if v1.dim() == 1:
                        v1 = v1.unsqueeze(0)
                    if v2.dim() == 1:
                        v2 = v2.unsqueeze(0)
                    vecs1.append(v1)
                    vecs2.append(v2)
                except Exception:
                    continue
            if not vecs1:
                return torch.tensor(0.0, device=device), 0, 1
            v1 = torch.cat(vecs1, dim=0)
            v2 = torch.cat(vecs2, dim=0)
            labels = labels[: v1.size(0)]
            loss = _loss_fn(v1, v2, labels)
            return loss, 0, v1.size(0)
        return step_fn

    seed = 42
    device = torch.device("cpu")
    vocab = get_default_vocab()
    loss_fn = ContrastiveLoss(margin=0.5).to(device)
    model = MultiModalFusionModel(
        pcode_vocab_size=max(len(vocab), 64),
        embed_dim=32,
        hidden_dim=64,
        num_gnn_layers=1,
        num_transformer_layers=1,
        output_dim=64,
    ).to(device)
    step_fn = make_step_fn(vocab, device, loss_fn)

    def run_with_seed(s):
        torch.manual_seed(s)
        random.seed(s)
        try:
            import numpy
            numpy.random.seed(s)
        except ImportError:
            pass
        ds = PairwiseSyntheticDataset(None, num_pairs=50, positive_ratio=0.5, seed=s)
        g = torch.Generator()
        g.manual_seed(s)
        loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate, generator=g)
        return _run_one_epoch_total_loss(loader, model, loss_fn, step_fn, device)

    loss1 = run_with_seed(seed)
    loss2 = run_with_seed(seed)
    assert loss1 == loss2, f"相同 seed 下 loss 应一致: {loss1} vs {loss2}"
