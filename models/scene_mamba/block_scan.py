# Copyright (c) 2023, Tri Dao, Albert Gu.
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

# 尝试导入高性能的RMSNorm实现，如果失败则置为None
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    RMSNorm = None
# 导入timm库中的DropPath，用于实现随机深度（Stochastic Depth）
from timm.models.layers import DropPath


class Mlp(nn.Module):
    """
    标准的多层感知机（MLP）或前馈网络（FFN）模块。
    """
    def __init__(self, in_features, mlp_ratio=4., hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        # 计算隐藏层维度，通常是输入维度的4倍
        hidden_features = hidden_features or int(in_features * mlp_ratio)
        # 第一个全连接层：从输入维度扩展到隐藏维度
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活函数
        self.act = act_layer()
        # 第二个全连接层：从隐藏维度压缩回输出维度
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout层，用于正则化
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.
    ):
        """
        一个通用的神经网络块，封装了mixer模块、归一化和残差连接。

        Args:
            dim (int): 特征维度。
            mixer_cls (nn.Module): 核心计算模块，可以是Mlp, MultiHeadAttention, 或 MambaBlock。
            norm_cls (nn.Module): 归一化层，如LayerNorm或RMSNorm。
            fused_add_norm (bool): 是否使用融合的加法和归一化操作以提升性能。
            drop_path (float): DropPath的概率，一种正则化技术。
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        # 实例化核心的mixer模块
        self.mixer = mixer_cls(dim)
        # 实例化归一化层
        self.norm = norm_cls(dim)
        # 实例化DropPath层
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        """
        前向传播。结构为：Add -> LN -> Mixer。
        """
        # 注意这里的残差连接方式：先将mixer的输出通过drop_path，然后加到输入上。
        # 这里的hidden_states是上一层的输出。
        # 整个流程是：LN(hidden_states) -> Mixer -> DropPath -> + hidden_states
        hidden_states = hidden_states + self.drop_path(
            self.mixer(self.norm(hidden_states), inference_params=inference_params)
        )
        return hidden_states