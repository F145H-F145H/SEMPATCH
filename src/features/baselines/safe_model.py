"""SAFE 基线适配器：委托 embed_batch_safe，实现 BaseSimilarityModel 接口。"""

from typing import Any, Dict, List, Optional

from features.baselines.base import BaseSimilarityModel
from features.baselines.safe import embed_batch_safe


class SafeModel(BaseSimilarityModel):
    """SAFE (Structural-Aware Function Embedding) 基线。"""

    @property
    def name(self) -> str:
        return "safe"

    @property
    def output_dim(self) -> int:
        return 128

    def embed_batch(
        self,
        features: Dict[str, Any],
        *,
        model_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return embed_batch_safe(features, model_path=model_path)
