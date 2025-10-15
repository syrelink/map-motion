# motion_wkv.py
# 本文件代码深度借鉴自 Vision-RWKV 官方代码库
# 我们将其改造为专用于 (Batch, Seq_Len, Dim) 格式的1D时序动作序列。

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp, DropPath, to_2tuple, trunc_normal_

# 核心：为1D时序数据改编的 Bi-WKV 模块 (纯 PyTorch 实现)
class Motion_BiWKV(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # 这些是 RWKV 架构的核心可学习参数
        self.w = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.u = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
    def forward(self, r, k, v):
        # r, k, v 的形状: (B, T, C)
        B, T, C = r.shape
        r = r.sigmoid()
        
        # 将绝对位置偏置转换为相对偏置
        t = torch.arange(T, device=r.device).reshape(1, T, 1)
        w = self.w.exp() * -1
        w = w * (T - 1 - t) / (T - 1) # Bounded exponential decay

        p = torch.einsum('btc, blc -> btlc', k, w)
        p = p - p.amax(dim=-1, keepdim=True).detach()
        p = p.exp()
        
        # 前向扫描
        p_f = p / (p.sum(dim=-1, keepdim=True) + 1e-6)
        x_f = torch.einsum('btlc, blc -> btc', p_f, v)
        
        # 后向扫描 (通过翻转序列实现)
        k_b = k.flip(dims=(1,))
        v_b = v.flip(dims=(1,))
        w_b = self.w.exp() * -1
        w_b = w_b * t / (T - 1)

        p_b = torch.einsum('btc, blc -> btlc', k_b, w_b)
        p_b = p_b - p_b.amax(dim=-1, keepdim=True).detach()
        p_b = p_b.exp()
        
        p_b_s = p_b / (p_b.sum(dim=-1, keepdim=True) + 1e-6)
        x_b = torch.einsum('btlc, blc -> btc', p_b_s, v_b)
        x_b = x_b.flip(dims=(1,))
        
        # 融合前向和后向结果
        x = (x_f + x_b) / 2
        return r * x

# 核心：为1D时序数据改编的 Temporal-Shift (动作版的 Q-Shift)
class TemporalShift(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 可学习的混合参数，借鉴自 VRWKV 的 Q-Shift 设计
        self.mu = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        B, T, C = x.shape
        # 将特征沿通道维度切分为两半
        x_past, x_future = x.chunk(2, dim=-1)
        
        # 时间平移：第一半向右移一帧（看过去），第二半向左移一帧（看未来）
        x_past = F.pad(x_past, (0, 0, 1, 0))[:, :-1, :]
        x_future = F.pad(x_future, (0, 0, 0, 1))[:, 1:, :]
        
        x_shifted = torch.cat([x_past, x_future], dim=-1)
        
        # 通过可学习参数 mu 与原始 x 混合
        return x + self.mu * x_shifted

# 核心：一个完整的 WKV 块，包含 Spatial-Mix (我们的 Motion-Mix) 和 Channel-Mix
class MotionWKVBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        
        # 1. Motion-Mix 模块 (原论文的 Spatial-Mix)
        self.motion_shift = TemporalShift(dim)
        self.r_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.wkv = Motion_BiWKV(dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(dim)
        
        # 2. Channel-Mix 模块 (即标准的前馈网络 FFN)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward(self, x):
        # Motion-Mix 分支
        shortcut = x
        x = self.ln1(x)
        x_shifted = self.motion_shift(x)
        r = self.r_proj(x_shifted)
        k = self.k_proj(x_shifted)
        v = self.v_proj(x_shifted)
        x = self.wkv(r, k, v)
        x = self.out_proj(x)
        x = shortcut + self.drop_path(x)
        
        # Channel-Mix 分支
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x