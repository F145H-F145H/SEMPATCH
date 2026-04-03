"""jTrans 风格基线：块序 token、embed_batch_jtrans_style、检查点往返。"""

import os
import tempfile

import pytest

from features.baselines.jtrans_style import (
    _JTransStyleEncoder,
    embed_batch_jtrans_style,
    jtrans_style_load_model,
    jtrans_style_save_model,
    jtrans_style_tokenize,
)


def _mm_with_blocks():
    return {
        "graph": {
            "num_nodes": 2,
            "edge_index": [[0], [1]],
            "node_list": ["a", "b"],
            "node_features": [
                {"pcode_opcodes": ["COPY", "INT_ADD"]},
                {"pcode_opcodes": ["INT_SUB"]},
            ],
        },
        "sequence": {"pcode_tokens": [], "jump_mask": [], "seq_len": 0},
    }


def test_jtrans_style_tokenize_uses_block_prefix():
    mm = _mm_with_blocks()
    vocab = {"": 0, "[UNK]": 1, "@0:COPY": 2, "@0:INT_ADD": 3, "@1:INT_SUB": 4}
    ids, pad = jtrans_style_tokenize(mm, vocab, max_len=16)
    assert pad[0] is False
    assert ids[0] == 2 and ids[1] == 3 and ids[2] == 4


def test_jtrans_style_tokenize_unknown_goes_unk():
    mm = _mm_with_blocks()
    vocab = {"": 0, "[UNK]": 1}
    ids, _ = jtrans_style_tokenize(mm, vocab, max_len=8)
    assert all(x == 1 for x in ids[:3])


def test_embed_batch_jtrans_style_dim():
    features = {
        "functions": [
            {"name": "f1", "features": {"multimodal": _mm_with_blocks()}},
        ]
    }
    out = embed_batch_jtrans_style(features)
    assert len(out) == 1
    assert out[0]["name"] == "f1"
    assert len(out[0]["vector"]) == 128


def test_jtrans_style_checkpoint_roundtrip():
    pytest.importorskip("torch")
    import torch

    vocab = {"": 0, "[UNK]": 1, "@0:COPY": 2}
    model = _JTransStyleEncoder(vocab_size=128, embed_dim=32, output_dim=64)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        jtrans_style_save_model(model, vocab, path, embed_dim=32, output_dim=64)
        loaded, v2 = jtrans_style_load_model(path)
        assert v2 == vocab
        ids = torch.tensor([[2, 1, 0]], dtype=torch.long)
        pad = torch.tensor([[False, False, True]], dtype=torch.bool)
        assert loaded(ids, pad).shape == (1, 64)
    finally:
        os.unlink(path)


def test_embed_batch_jtrans_style_with_model_path():
    pytest.importorskip("torch")
    vocab = {"": 0, "[UNK]": 1, "@0:COPY": 2, "@0:INT_ADD": 3, "@1:INT_SUB": 4}
    model = _JTransStyleEncoder(vocab_size=256, embed_dim=32, output_dim=128)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        jtrans_style_save_model(model, vocab, path, embed_dim=32, output_dim=128)
        features = {
            "functions": [
                {"name": "g1", "features": {"multimodal": _mm_with_blocks()}},
            ]
        }
        out = embed_batch_jtrans_style(features, model_path=path)
        assert len(out[0]["vector"]) == 128
    finally:
        os.unlink(path)
