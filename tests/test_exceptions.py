"""测试异常继承链与 isinstance 检查。"""

from exceptions import (
    DataIntegrityError,
    EmbeddingError,
    FeatureExtractionError,
    SemPatchError,
)


def test_all_exceptions_inherit_from_sempatch_error():
    assert issubclass(FeatureExtractionError, SemPatchError)
    assert issubclass(EmbeddingError, SemPatchError)
    assert issubclass(DataIntegrityError, SemPatchError)
    assert issubclass(SemPatchError, Exception)


def test_isinstance_catches_subclass():
    for exc_cls in (FeatureExtractionError, EmbeddingError, DataIntegrityError):
        exc = exc_cls("test")
        assert isinstance(exc, SemPatchError)
        assert isinstance(exc, Exception)


def test_sempatch_error_message():
    e = SemPatchError("something broke")
    assert str(e) == "something broke"


def test_exception_not_interchangeable():
    """子类异常不应被兄弟类捕获。"""
    import pytest

    with pytest.raises(EmbeddingError):
        raise EmbeddingError("oom")
    with pytest.raises(FeatureExtractionError):
        raise FeatureExtractionError("cfg empty")
    # EmbeddingError 不应被 FeatureExtractionError 捕获
    try:
        raise EmbeddingError("oom")
    except FeatureExtractionError:
        raise AssertionError("EmbeddingError 不应被 FeatureExtractionError 捕获")
    except EmbeddingError:
        pass  # 期望行为
