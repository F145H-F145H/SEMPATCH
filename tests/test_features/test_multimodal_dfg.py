"""MultiModalFusionModel DFG 分支与张量化契约（阶段 H）。"""
import pytest

torch = pytest.importorskip("torch")

from features.models.multimodal_fusion import (  # noqa: E402
    MultiModalFusionModel,
    _tensorize_multimodal,
    get_default_vocab,
    infer_use_dfg_from_state_dict,
    parse_multimodal_checkpoint,
)


def _minimal_mm(with_dfg: bool) -> dict:
    mm = {
        "graph": {
            "num_nodes": 2,
            "edge_index": [[0], [1]],
            "node_list": ["bb_0", "bb_1"],
            "node_features": [
                {"pcode_opcodes": ["COPY"]},
                {"pcode_opcodes": ["RETURN"]},
            ],
        },
        "sequence": {
            "pcode_tokens": ["COPY", "RETURN"],
            "jump_mask": [0, 1],
            "seq_len": 2,
        },
    }
    if with_dfg:
        mm["dfg"] = {
            "num_nodes": 2,
            "edge_index": [[0], [1]],
            "node_list": ["a", "b"],
            "node_features": [42, 99],
        }
    return mm


def test_tensorize_returns_dfg_tensors():
    vocab = get_default_vocab()
    mm = _minimal_mm(True)
    t = _tensorize_multimodal(mm, vocab, max_seq_len=16, max_graph_nodes=8, max_dfg_nodes=8)
    assert len(t) == 7
    _, _, _, _, _, dfg_n, dfg_e = t
    assert dfg_n.shape[1] <= 8
    assert dfg_e.shape[0] == 2


def test_forward_use_dfg_false_matches_dim():
    vocab = get_default_vocab()
    mm = _minimal_mm(True)
    t = _tensorize_multimodal(mm, vocab, max_seq_len=16, max_graph_nodes=8, max_dfg_nodes=8)
    tok, jmp, gn, ge, pm, dn, de = t
    m = MultiModalFusionModel(pcode_vocab_size=256, use_dfg=False)
    m.eval()
    with torch.no_grad():
        out = m(tok, jmp, gn, ge, padding_mask=pm, dfg_node_features=dn, dfg_edge_index=de)
    assert out.shape == (128,)


def test_forward_use_dfg_true():
    vocab = get_default_vocab()
    mm = _minimal_mm(True)
    t = _tensorize_multimodal(mm, vocab, max_seq_len=16, max_graph_nodes=8, max_dfg_nodes=8)
    tok, jmp, gn, ge, pm, dn, de = t
    m = MultiModalFusionModel(pcode_vocab_size=256, use_dfg=True)
    m.eval()
    with torch.no_grad():
        out = m(tok, jmp, gn, ge, padding_mask=pm, dfg_node_features=dn, dfg_edge_index=de)
    assert out.shape == (128,)


def test_forward_empty_dfg_use_dfg_true():
    vocab = get_default_vocab()
    mm = _minimal_mm(False)
    t = _tensorize_multimodal(mm, vocab, max_seq_len=16, max_graph_nodes=8, max_dfg_nodes=8)
    tok, jmp, gn, ge, pm, dn, de = t
    m = MultiModalFusionModel(pcode_vocab_size=256, use_dfg=True)
    m.eval()
    with torch.no_grad():
        out = m(tok, jmp, gn, ge, padding_mask=pm, dfg_node_features=dn, dfg_edge_index=de)
    assert out.shape == (128,)


def test_infer_use_dfg_from_state_dict():
    assert not infer_use_dfg_from_state_dict({"seq_embed.weight": 1})
    assert infer_use_dfg_from_state_dict({"dfg_node_embed.weight": 1})


def test_parse_multimodal_checkpoint_wrapped():
    sd = {"seq_embed.weight": torch.zeros(3, 3)}
    raw = {"state_dict": sd, "meta": {"use_dfg": True}}
    a, b = parse_multimodal_checkpoint(raw)
    assert a == sd
    assert b.get("use_dfg") is True
