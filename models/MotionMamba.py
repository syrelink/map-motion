# motionmamba.py
#
# A hybrid Transformer-Mamba decoder architecture for motion generation,
# inspired by the design principles of MambaVision and the multi-level
# feature fusion from the Affordance paper.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from mamba_ssm import Mamba

class AttentionBlock(nn.Module):
    """一个标准的 Multi-Head Attention 模块，用于自注意力和交叉注意力。"""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    def forward(self, query, key, value, key_padding_mask=None):
        out, _ = self.attn(query, key, value, key_padding_mask=key_padding_mask)
        return out

class TransMambaDecoderBlock(nn.Module):
    """一个 Trans-Mamba 混合解码器层，直接定义在主模型文件中。"""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        
        # 定义所有子模块
        self.norm1 = nn.LayerNorm(d_model)
        self.self_mamba = MambaBlock(d_model=d_model)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = AttentionBlock(d_model, nhead, dropout=dropout)

        self.norm_mem = nn.LayerNorm(d_model)
        self.scene_mamba = MambaBlock(d_model=d_model)
        
        self.norm3 = nn.LayerNorm(d_model)
        self.cross_attn = AttentionBlock(d_model, nhead, dropout=dropout)
        
        self.norm4 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # 残差连接的缩放因子
        self.gamma_sm = nn.Parameter(torch.ones(d_model))
        self.gamma_sa = nn.Parameter(torch.ones(d_model))
        self.gamma_ca = nn.Parameter(torch.ones(d_model))
        self.gamma_ff = nn.Parameter(torch.ones(d_model))

    def forward(self, x, mem, x_mask=None, mem_mask=None):
        # 串行增强流程
        x = x + self.gamma_sm * self.self_mamba(self.norm1(x))
        x = x + self.gamma_sa * self.self_attn(self.norm2(x), self.norm2(x), self.norm2(x), key_padding_mask=x_mask)
        mem_enhanced = self.scene_mamba(self.norm_mem(mem))
        x = x + self.gamma_ca * self.cross_attn(self.norm3(x), mem_enhanced, mem_enhanced, key_padding_mask=mem_mask)
        x = x + self.gamma_ff * self.ffn(self.norm4(x))
        return x