"""
对比学习损失函数，用于孪生网络训练。
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None  # type: ignore


class ContrastiveLoss(nn.Module if TORCH_AVAILABLE else object):
    """
    对比损失（余弦相似度版本）。
    label=1（相似）时：(1 - cos_sim)^2
    label=0（不相似）时：max(0, cos_sim - margin)^2
    """

    def __init__(self, margin: float = 0.5):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for ContrastiveLoss")
        super().__init__()
        self.margin = margin

    def forward(
        self,
        vec1: "torch.Tensor",
        vec2: "torch.Tensor",
        labels: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Args:
            vec1: (B, D) 第一个嵌入向量
            vec2: (B, D) 第二个嵌入向量
            labels: (B,) 0 或 1，1 表示相似

        Returns:
            标量损失
        """
        cos_sim = F.cosine_similarity(vec1, vec2, dim=1)
        pos_loss = (1 - cos_sim).pow(2)
        # 对负样本（label=0）：希望 cos_sim <= margin；超出 margin 才惩罚
        neg_loss = torch.clamp(cos_sim - self.margin, min=0).pow(2)
        loss = labels * pos_loss + (1 - labels) * neg_loss
        return loss.mean()
