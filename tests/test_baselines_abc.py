"""验证每个基线适配器满足 BaseSimilarityModel ABC 契约。"""

import pytest

from features.baselines.acfg_model import ACFGModel
from features.baselines.base import BaseSimilarityModel
from features.baselines.jtrans_style_model import JTransStyleModel
from features.baselines.safe_model import SafeModel


ALL_MODELS = [SafeModel, JTransStyleModel, ACFGModel]


@pytest.mark.parametrize("model_cls", ALL_MODELS)
def test_is_subclass_of_base(model_cls):
    assert issubclass(model_cls, BaseSimilarityModel)


@pytest.mark.parametrize("model_cls", ALL_MODELS)
def test_has_name(model_cls):
    m = model_cls()
    assert isinstance(m.name, str)
    assert len(m.name) > 0


@pytest.mark.parametrize("model_cls", ALL_MODELS)
def test_has_output_dim(model_cls):
    m = model_cls()
    assert isinstance(m.output_dim, int)
    assert m.output_dim > 0


def test_acfg_embed_batch_returns_expected_format():
    """ACFGModel.embed_batch 返回 [{name, vector}, ...] 格式。"""
    model = ACFGModel()
    features = {
        "func_a": {
            "cfg": {"nodes": ["n0", "n1"], "edges": [("n0", "n1")]},
            "basic_blocks": [
                {"instructions": [{"pcode": [{"opcode": "COPY"}, {"opcode": "INT_ADD"}]}]},
            ],
        },
    }
    results = model.embed_batch(features)
    assert len(results) == 1
    entry = results[0]
    assert entry["name"] == "func_a"
    assert isinstance(entry["vector"], list)
    assert len(entry["vector"]) == 128
    assert all(isinstance(x, float) for x in entry["vector"])


def test_acfg_empty_input_returns_empty():
    """ACFGModel 对空输入返回空列表。"""
    model = ACFGModel()
    results = model.embed_batch({})
    assert results == []
