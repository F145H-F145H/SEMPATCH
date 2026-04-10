"""基线模型统一接口：BaseSimilarityModel ABC。"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseSimilarityModel(ABC):
    """
    基线相似度模型接口。

    所有基线（SAFE、jTrans-style、ACFG 等）实现此接口，
    保证 embed_batch 返回统一的 [{"name": str, "vector": list[float]}, ...] 格式。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """模型标识名，如 'safe'、'jtrans_style'、'acfg'。"""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """输出向量维度，如 128。"""

    @abstractmethod
    def embed_batch(
        self,
        features: Dict[str, Any],
        *,
        model_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量嵌入。

        Args:
            features: {function_id: multimodal_dict} 或包含 function_id 键的容器
            model_path: 可选，模型权重路径

        Returns:
            [{"name": function_id, "vector": [float, ...]}, ...]，按 function_id 顺序
        """
