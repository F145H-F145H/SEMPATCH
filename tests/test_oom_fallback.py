"""测试 CUDA OOM 回退 CPU 失败时抛出 EmbeddingError。"""

import pytest

from exceptions import EmbeddingError


def test_oom_fallback_raises_embedding_error():
    """CUDA OOM + CPU 回退也失败 → EmbeddingError。"""
    from features.inference import run_with_cuda_oom_fallback

    call_count = 0

    def _always_fail(device):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("CUDA out of memory")

    # 模拟 CUDA 设备
    class FakeCudaDevice:
        type = "cuda"

    with pytest.raises(EmbeddingError, match="CPU 回退也失败"):
        run_with_cuda_oom_fallback(_always_fail, FakeCudaDevice(), context="test")

    assert call_count == 2  # CUDA 一次 + CPU 一次


def test_non_oom_error_propagates():
    """非 OOM 的 RuntimeError 直接抛出，不转 EmbeddingError。"""
    from features.inference import run_with_cuda_oom_fallback

    def _value_error(device):
        raise RuntimeError("something else broke")

    class FakeCudaDevice:
        type = "cuda"

    with pytest.raises(RuntimeError, match="something else broke"):
        run_with_cuda_oom_fallback(_value_error, FakeCudaDevice())


def test_success_path_unchanged():
    """成功路径不受影响。"""
    from features.inference import run_with_cuda_oom_fallback

    def _success(device):
        return 42

    result = run_with_cuda_oom_fallback(_success, "cpu")
    assert result == 42
