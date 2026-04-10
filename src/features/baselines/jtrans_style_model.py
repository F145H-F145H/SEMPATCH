"""jTrans-style 基线适配器：委托 embed_batch_jtrans_style，实现 BaseSimilarityModel 接口。"""

from typing import Any, Dict, List, Optional

from features.baselines.base import BaseSimilarityModel
from features.baselines.jtrans_style import embed_batch_jtrans_style


class JTransStyleModel(BaseSimilarityModel):
    """jTrans 风格基线（基于 CFG 块级 opcode 序列）。"""

    @property
    def name(self) -> str:
        return "jtrans_style"

    @property
    def output_dim(self) -> int:
        return 128

    def embed_batch(
        self,
        features: Dict[str, Any],
        *,
        model_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return embed_batch_jtrans_style(features, model_path=model_path)
