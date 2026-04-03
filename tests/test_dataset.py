"""PairwiseFunctionDataset 单元测试。"""

import json
import os
import sys
import tempfile

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def _make_minimal_index(path: str, project_root: str) -> None:
    """创建最小索引：同一二进制内两函数，用于正对需跨二进制则无正对。"""
    bin_path = os.path.join(project_root, "data", "binkit_subset", "addpart.elf")
    data = [
        {
            "binary": "data/binkit_subset/addpart.elf",
            "functions": [
                {"name": "func_a", "entry": "0x401000"},
                {"name": "func_b", "entry": "0x40100d"},
            ],
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


@pytest.fixture
def minimal_index(tmp_path):
    """临时最小索引（无正对）。"""
    p = tmp_path / "index.json"
    _make_minimal_index(str(p), PROJECT_ROOT)
    return str(p)


def test_pairwise_dataset_structure(minimal_index):
    """验证 PairwiseFunctionDataset 返回结构含 feature1、feature2、label。"""
    from features.dataset import PairwiseFunctionDataset

    ds = PairwiseFunctionDataset(
        minimal_index,
        project_root=PROJECT_ROOT,
        num_pairs=5,
        positive_ratio=0.0,  # 只采样负对，避免 Ghidra
    )
    assert len(ds) == 5
    for i in range(min(3, len(ds))):
        item = ds[i]
        assert "feature1" in item
        assert "feature2" in item
        assert "label" in item
        assert item["label"] in (0, 1)
        f1 = item["feature1"]
        f2 = item["feature2"]
        assert "graph" in f1 and "sequence" in f1
        assert "graph" in f2 and "sequence" in f2
        assert "pcode_tokens" in f1.get("sequence", {})
        assert "node_features" in f1.get("graph", {})


def test_pairwise_dataset_prefers_precomputed_features(minimal_index, tmp_path):
    """提供预计算特征时，优先命中 map，不走动态提取。"""
    from features.dataset import PairwiseFunctionDataset

    precomputed_path = tmp_path / "precomputed.json"
    precomputed = {
        "data/binkit_subset/addpart.elf|0x401000": {
            "graph": {"num_nodes": 1, "edge_index": [[], []], "node_list": ["bb_0"], "node_features": [[]]},
            "sequence": {"pcode_tokens": ["COPY"], "jump_mask": [0], "seq_len": 1},
        }
    }
    with open(precomputed_path, "w", encoding="utf-8") as f:
        json.dump(precomputed, f, ensure_ascii=False)

    ds = PairwiseFunctionDataset(
        minimal_index,
        project_root=PROJECT_ROOT,
        num_pairs=1,
        positive_ratio=0.0,
        precomputed_features_path=str(precomputed_path),
    )
    v = ds._get_features(os.path.join(PROJECT_ROOT, "data", "binkit_subset", "addpart.elf"), "0x401000")
    assert v is not None
    assert v["sequence"]["seq_len"] == 1


def test_trainer_smoke():
    """验证 Trainer 可运行一个 epoch 不崩溃。"""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from features.trainer import Trainer

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

        def forward(self, x):
            return self.linear(x)

    model = DummyModel()
    X = torch.randn(10, 8)
    ds = TensorDataset(X)
    loader = DataLoader(ds, batch_size=2)

    def step_fn(batch, model, loss_fn):
        x, = batch
        out = model(x)
        loss = out.mean()
        return loss, 0, x.size(0)

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        loss_fn=None,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        device=torch.device("cpu"),
        save_path="/tmp/test_best.pth",
        step_fn=step_fn,
    )
    trainer.train_epoch()
    val_loss, val_acc = trainer.validate()
    assert isinstance(val_loss, (int, float))


def test_pairwise_binkit_refined_with_precomputed(tmp_path):
    """同源 project_id + 同名：refined 模式可采到正对（不跑 Ghidra）。"""
    from features.dataset import PairwiseFunctionDataset

    idx = tmp_path / "refined.json"
    mm = {
        "graph": {
            "num_nodes": 4,
            "edge_index": [[0, 1], [1, 2]],
            "node_list": [f"bb_{i}" for i in range(4)],
            "node_features": [[] for _ in range(4)],
        },
        "sequence": {"pcode_tokens": ["COPY"] * 20, "jump_mask": [0] * 20, "seq_len": 20},
    }
    data = [
        {
            "binary": "data/pkg/x86_64/foo_O2_gcc.elf",
            "functions": [{"name": "bar", "entry": "0x401000"}],
        },
        {
            "binary": "data/pkg/x86_64/foo_O3_gcc.elf",
            "functions": [{"name": "bar", "entry": "0x401000"}],
        },
    ]
    with open(idx, "w", encoding="utf-8") as f:
        json.dump(data, f)
    pre = tmp_path / "pc.json"
    precomputed = {
        "data/pkg/x86_64/foo_O2_gcc.elf|0x401000": mm,
        "data/pkg/x86_64/foo_O3_gcc.elf|0x401000": mm,
    }
    with open(pre, "w", encoding="utf-8") as f:
        json.dump(precomputed, f)

    ds = PairwiseFunctionDataset(
        str(idx),
        project_root=PROJECT_ROOT,
        num_pairs=8,
        positive_ratio=1.0,
        precomputed_features_path=str(pre),
        pairing_mode="binkit_refined",
        seed=0,
    )
    assert ds._refined_positive_candidates
    got_pos = 0
    for _ in range(20):
        item = ds[_ % len(ds)]
        if int(item["label"]) == 1:
            got_pos += 1
    assert got_pos > 0


def test_contrastive_loss():
    """验证 ContrastiveLoss 可计算且梯度正常。"""
    import torch
    from features.losses import ContrastiveLoss

    loss_fn = ContrastiveLoss(margin=0.5)
    B, D = 4, 128
    vec1 = torch.randn(B, D, requires_grad=True)
    vec2 = torch.randn(B, D, requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = loss_fn(vec1, vec2, labels)
    assert loss.dim() == 0
    loss.backward()
    assert vec1.grad is not None
    assert vec2.grad is not None
