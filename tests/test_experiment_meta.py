"""测试 experiment_meta 模块：metadata 收集与确定性种子。"""

import json
import os
import tempfile


def test_collect_metadata_returns_expected_keys():
    """collect_metadata 返回包含 git_commit、seed 等必选 key。"""
    from experiment_meta import collect_metadata

    class FakeArgs:
        def __init__(self):
            self.seed = 42
            self.epochs = 10

    meta = collect_metadata(FakeArgs())
    assert "git_commit" in meta
    assert "seed" in meta
    assert meta["seed"] == 42
    assert "cli_args" in meta
    assert meta["cli_args"]["seed"] == 42
    assert "python_version" in meta
    assert "torch_version" in meta
    assert "numpy_version" in meta
    assert "hostname" in meta
    assert "timestamp" in meta


def test_collect_metadata_with_extra():
    """collect_metadata 附加 extra dict。"""
    from experiment_meta import collect_metadata

    class FakeArgs:
        def __init__(self):
            self.seed = 99

    meta = collect_metadata(FakeArgs(), extra={"model": "test"})
    assert meta["extra"]["model"] == "test"


def test_save_metadata_creates_file():
    """save_metadata 创建 .metadata.json 文件且可解析。"""
    from experiment_meta import save_metadata

    class FakeArgs:
        def __init__(self):
            self.seed = 42

    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "model.pth")
        meta_path = save_metadata(model_path, FakeArgs())
        assert meta_path == model_path + ".metadata.json"
        assert os.path.isfile(meta_path)
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["seed"] == 42


def test_set_deterministic_sets_flags():
    """set_deterministic 设置 cuDNN 标志与 PYTHONHASHSEED。"""
    import torch
    from experiment_meta import set_deterministic

    set_deterministic(42)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert os.environ.get("PYTHONHASHSEED") == "42"
