# 文件名: res_mamba_trans.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from mamba_trans import *
import math


# --- MambaVision Mixer 核心模块 ---
class MambaVisionMixer(nn.Module):
    """
    MambaVision Mixer 模块，根据 MambaVision 论文的核心思想实现。
    它采用双分支架构：一个SSM（状态空间模型）分支用于捕捉长距离依赖，
    另一个对称的非SSM分支用于捕捉局部和空间信息。
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", conv_bias=True, bias=False, **kwargs):
        """
        初始化函数。

        Args:
            d_model (int): 模型的核心特征维数 (D)。
            d_state (int): SSM的隐状态维数 (N)。
            d_conv (int): 一维卷积核的大小。
            expand (int): 内部维度的扩展因子。
            dt_rank (int or "auto"): SSM参数Δ的秩，用于低秩分解。
            conv_bias (bool): 卷机层是否使用偏置。
            bias (bool): 线性层是否使用偏置。
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        # 内部处理维度 d_inner = expand * d_model
        self.d_inner = int(self.expand * self.d_model)
        # 自动计算 dt 的秩
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        
        # 输入线性投射层：将 d_model 扩展到 d_inner
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias)
        # 输出线性投射层：将 d_inner 还原到 d_model
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        # SSM 分支 (x-path) 使用的一维深度卷积 (Depthwise Conv1d)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2, out_channels=self.d_inner // 2, bias=conv_bias,
            kernel_size=d_conv, groups=self.d_inner // 2, padding=d_conv - 1
        )
        # 对称的非SSM分支 (z-path) 使用的一维深度卷积
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2, out_channels=self.d_inner // 2, bias=conv_bias,
            kernel_size=d_conv, groups=self.d_inner // 2, padding=d_conv - 1
        )

        # 线性层，用于从输入x动态生成SSM参数(Δ, B, C)
        self.x_proj = nn.Linear(self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False)
        # 线性层，用于将低秩的Δ恢复到完整维度
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True)

        # 初始化SSM参数A (状态转移矩阵)。存储其对数值log(A)并设为可训练参数
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner // 2)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        # 初始化SSM参数D (直连参数)，设为可训练参数
        self.D = nn.Parameter(torch.ones(self.d_inner // 2))

    def forward(self, hidden_states):
        """
        前向传播函数。

        Args:
            hidden_states (Tensor): 输入张量，形状为 (B, L, D)。
        
        Returns:
            Tensor: 输出张量，形状为 (B, L, D)。
        """
        B, L, _ = hidden_states.shape

        # 1. 输入投射与分割
        xz = self.in_proj(hidden_states)  # (B, L, D) -> (B, L, d_inner)
        # 在特征维度上将张量一分为二，分别送入两个分支
        x, z = xz.chunk(2, dim=2)        # x 和 z 的形状均为 (B, L, d_inner/2)

        # --- 2. SSM 分支 (x-path) ---
        # 维度变换以适应Conv1d的输入格式 (B, C, L)
        x_ssm_input = x.permute(0, 2, 1).contiguous()
        # 通过一维卷积
        x_conv_output = self.conv1d_x(x_ssm_input)[:, :, :L] # 裁剪掉padding部分
        # 通过SiLU激活函数
        x_ssm_input = F.silu(x_conv_output)
        
        # 从激活后的x生成动态SSM参数
        x_for_proj = x_ssm_input.permute(0, 2, 1) # 维度换回 (B, L, C)
        x_dbl = self.x_proj(x_for_proj)          # 生成 (Δ, B, C) 的拼接形式

        # 分割出 dt, B, C
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # 处理dt (Δ)
        dt = self.dt_proj(dt).permute(0, 2, 1) # 恢复维度并变换为 (B, C, L)

        # 变换B和C的维度以适配selective_scan_fn的输入
        B_ssm = B_ssm.permute(0, 2, 1).contiguous() # (B, L, d_state) -> (B, d_state, L)
        C_ssm = C_ssm.permute(0, 2, 1).contiguous() # (B, L, d_state) -> (B, d_state, L)

        # 计算A矩阵，取负指数以保证系统稳定性
        A = -torch.exp(self.A_log.float())
        
        # Mamba核心计算：执行选择性扫描
        y_ssm = selective_scan_fn(x_ssm_input, dt, A, B_ssm, C_ssm, self.D.float(), z=None,
                                  delta_bias=self.dt_proj.bias.float(), delta_softplus=True)

        # --- 3. 对称的非SSM分支 (z-path) ---
        z_conv_input = z.permute(0, 2, 1).contiguous()
        z_conv_output = self.conv1d_z(z_conv_input)[:, :, :L]
        z_non_ssm = F.silu(z_conv_output)

        # --- 4. 融合与输出 ---
        # 根据论文的最佳实践，在通道维度上拼接两个分支的输出
        y = torch.cat([y_ssm, z_non_ssm], dim=1).permute(0, 2, 1) # (B, d_inner, L) -> (B, L, d_inner)
        
        # 通过输出线性层将维度从d_inner还原到d_model
        return self.out_proj(y)



class MambaTransHybridLayer(nn.Module):
    """
    一个混合层，结合了 MambaVisionMixer 和 Transformer 的前馈网络。
    结构: MambaBlock -> FFNBlock
    """
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, 
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        
        # --- Mamba 模块部分 (替换自注意力) ---
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba_mixer = MambaVisionMixer(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.dropout1 = nn.Dropout(dropout)

        # --- FFN 模块部分 (与 Transformer 保持一致) ---
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播，采用 Pre-Norm 结构。
        
        Args:
            x (Tensor): 输入张量，形状为 (B, L, D)
        
        Returns:
            Tensor: 输出张量，形状为 (B, L, D)
        """
        # Mamba 模块 + 第一个残差连接
        x_res = x
        x = self.norm1(x)
        x = self.mamba_mixer(x)
        x = x_res + self.dropout1(x)
        
        # FFN 模块 + 第二个残差连接
        x_res = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x_res + self.dropout2(x)
        
        return x


# --- mamba Block  ---
class ResidualHybridBlock(nn.Module):
    def __init__(self, dim, num_heads, is_transformer_layer=False, mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.is_transformer_layer = is_transformer_layer
        if self.is_transformer_layer:
            self.mixer = WindowedAttention(dim, num_heads=num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.mixer = MambaVisionMixer(
                d_model=dim,
                expand=4,  # <-- 尝试新值
                d_state=32,  # <-- 尝试新值
                d_conv=5  # <-- 也可以尝试调整
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        shortcut = x
        x_normed = self.norm1(x)
        if self.is_transformer_layer:
            B, L, C = x_normed.shape
            # 对于一维动作数据，无法假设 L 是完美平方数，此处不再进行窗口化
            # 回退到全局注意力
            x_processed = self.mixer(x_normed)
        else:
            x_processed = self.mixer(x_normed)
        x = shortcut + self.drop_path(x_processed)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# --- ResidualMambaVisionBackbone ---
class ResidualMambaVisionBackbone(nn.Module):
    def __init__(self, num_layers, latent_dim, num_heads, ff_size, dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        # num_mamba_layers = num_layers // 2
        num_mamba_layers = int(num_layers * 0.75)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            is_transformer = i >= num_mamba_layers
            self.blocks.append(
                ResidualHybridBlock(
                    dim=latent_dim,
                    num_heads=num_heads,
                    is_transformer_layer=is_transformer,
                    mlp_ratio=ff_size / latent_dim if latent_dim > 0 else 4.0,
                    drop=dropout,
                    attn_drop=dropout,
                    drop_path=dpr[i],
                )
            )

    def forward(self, x, src_key_padding_mask=None):
        x = x.permute(1, 0, 2)
        for blk in self.blocks:
            # 简化后的全局注意力暂不处理 mask
            x = blk(x)
        x = x.permute(1, 0, 2)
        return x