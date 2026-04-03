"""测试 SAFE 训练接口：safe_tokenize、模型保存/加载、embed_batch_safe(model_path)。"""

import json
import os
import tempfile

import pytest

from features.baselines.safe import (
    collect_vocab_from_features_file,
    embed_batch_safe,
    safe_load_model,
    safe_save_model,
    safe_tokenize,
)
from features.baselines.safe import _SafeEncoder


def _minimal_multimodal(pcode_tokens=None):
    """构造最小 multimodal 特征。"""
    if pcode_tokens is None:
        pcode_tokens = ["COPY", "INT_SUB", "INT_ADD"]
    return {
        "graph": {"num_nodes": 1, "edge_index": [[], []], "node_list": [], "node_features": [[]]},
        "sequence": {"pcode_tokens": pcode_tokens, "jump_mask": [0] * len(pcode_tokens), "seq_len": len(pcode_tokens)},
    }


def test_safe_tokenize_basic():
    """safe_tokenize 返回 (token_ids, pad_mask)，长度等于 max_len。"""
    mm = _minimal_multimodal(["COPY", "INT_ADD"])
    vocab = {"[PAD]": 0, "[UNK]": 1, "COPY": 2, "INT_ADD": 3}
    ids, pad_mask = safe_tokenize(mm, vocab, max_len=10)
    assert len(ids) == 10
    assert len(pad_mask) == 10
    assert ids[:2] == [2, 3]
    assert ids[2:] == [0] * 8
    assert pad_mask[:2] == [False, False]
    assert pad_mask[2:] == [True] * 8


def test_safe_tokenize_unknown_token():
    """未知 token 映射到 [UNK] (id=1)。"""
    mm = _minimal_multimodal(["UNKNOWN_OP"])
    vocab = {"[PAD]": 0, "[UNK]": 1}
    ids, _ = safe_tokenize(mm, vocab, max_len=5)
    assert ids[0] == 1


def test_safe_tokenize_truncate():
    """超长序列截断至 max_len。"""
    mm = _minimal_multimodal(["A"] * 20)
    vocab = {"[PAD]": 0, "[UNK]": 1, "A": 2}
    ids, pad_mask = safe_tokenize(mm, vocab, max_len=5)
    assert len(ids) == 5
    assert all(p == 2 for p in ids)
    assert not any(pad_mask)


def test_safe_save_load_roundtrip():
    """safe_save_model 与 safe_load_model 往返后模型可前向传播。"""
    vocab = {"[PAD]": 0, "[UNK]": 1, "COPY": 2}
    model = _SafeEncoder(vocab_size=256, embed_dim=32, output_dim=64)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        safe_save_model(model, vocab, path, embed_dim=32, output_dim=64)
        loaded_model, loaded_vocab = safe_load_model(path)
        assert loaded_vocab == vocab
        import torch
        ids = torch.tensor([[2, 1, 0]], dtype=torch.long)
        pad = torch.tensor([[False, False, True]], dtype=torch.bool)
        out = loaded_model(ids, pad)
        assert out.shape == (1, 64)
    finally:
        os.unlink(path)


def test_embed_batch_safe_without_model_path():
    """embed_batch_safe 无 model_path 时输出 128 维向量。"""
    features = {
        "functions": [
            {"name": "f1", "features": {"multimodal": _minimal_multimodal()}},
        ]
    }
    result = embed_batch_safe(features)
    assert len(result) == 1
    assert result[0]["name"] == "f1"
    assert len(result[0]["vector"]) == 128


def test_embed_batch_safe_with_model_path():
    """embed_batch_safe 指定 model_path 时加载权重并输出一致维度。"""
    vocab = {"[PAD]": 0, "[UNK]": 1, "COPY": 2, "INT_SUB": 3, "INT_ADD": 4}
    model = _SafeEncoder(vocab_size=256, embed_dim=32, output_dim=128)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        safe_save_model(model, vocab, path, embed_dim=32, output_dim=128)
        features = {
            "functions": [
                {"name": "f1", "features": {"multimodal": _minimal_multimodal(["COPY", "INT_ADD"])}},
            ]
        }
        result = embed_batch_safe(features, model_path=path)
        assert len(result) == 1
        assert len(result[0]["vector"]) == 128
    finally:
        os.unlink(path)


def test_collect_vocab_from_features_file():
    """从 features 文件构建 vocab，覆盖 sequence 与 graph。"""
    mm1 = _minimal_multimodal(["OP1", "OP2"])
    mm2 = _minimal_multimodal(["OP2", "OP3"])
    mm2["graph"]["node_features"] = [{"pcode_opcodes": ["OP4"]}]
    data = {"f1|0x1": mm1, "f2|0x2": mm2}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        vocab = collect_vocab_from_features_file(path)
        assert "[PAD]" in vocab
        assert "[UNK]" in vocab
        assert "OP1" in vocab
        assert "OP2" in vocab
        assert "OP3" in vocab
        assert "OP4" in vocab
    finally:
        os.unlink(path)


def test_safe_step_fn_produces_gradient():
    """训练 step_fn 前向传播得到可反向传播的 loss。"""
    import torch

    from features.baselines.safe import safe_tokenize
    from features.losses import ContrastiveLoss

    vocab = {"[PAD]": 0, "[UNK]": 1, "COPY": 2, "INT_ADD": 3}
    model = _SafeEncoder(vocab_size=256, embed_dim=16, output_dim=32)
    loss_fn = ContrastiveLoss(margin=0.5)

    mm1 = _minimal_multimodal(["COPY", "INT_ADD"])
    mm2 = _minimal_multimodal(["COPY", "INT_ADD"])
    ids1, pad1 = safe_tokenize(mm1, vocab, max_len=8)
    ids2, pad2 = safe_tokenize(mm2, vocab, max_len=8)

    t1 = torch.tensor([ids1], dtype=torch.long)
    p1 = torch.tensor([pad1], dtype=torch.bool)
    t2 = torch.tensor([ids2], dtype=torch.long)
    p2 = torch.tensor([pad2], dtype=torch.bool)

    v1 = model(t1, p1)
    v2 = model(t2, p2)
    labels = torch.tensor([1.0])
    loss = loss_fn(v1, v2, labels)
    loss.backward()
    assert isinstance(loss.item(), float)
