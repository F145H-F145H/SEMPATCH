"""SemPatch 项目级异常层次。"""


class SemPatchError(Exception):
    """SemPatch 基础异常，所有项目异常的父类。"""


class FeatureExtractionError(SemPatchError):
    """特征提取失败（CFG 不完整、LSIR 格式异常等）。"""


class EmbeddingError(SemPatchError):
    """嵌入生成失败（含 CUDA+CPU 双重 OOM）。"""


class DataIntegrityError(SemPatchError):
    """数据完整性校验失败（校验和不匹配等）。"""
