# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math
from types import SimpleNamespace # 用于创建简单的配置对象

# =============== 依赖项与核心模块定义 ===============
# 确保已安装: pip install timm einops mamba-ssm causal-conv1d
try:
    from timm.models.layers import DropPath, Mlp
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from einops import rearrange, repeat
    import torch.nn.functional as F
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所有必要的库: pip install torch timm einops mamba-ssm causal-conv1d")
    exit()

# 模块 1: 借鉴 MambaVisionMixer 实现的 Mamba 混合器
class MambaMixer(nn.Module):
    """
    一个更精细的 Mamba 实现，借鉴自 MambaVision。
    它作为一个混合器模块，处理 (B, L, D) 形状的序列数据。
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        bias=False,
        conv_bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * 1.0
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=self.d_inner).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, x, **mixer_kwargs): # mixer_kwargs 用于保持接口一致性，但 Mamba 不使用它
        B, L, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)[:, :, :L]
        x = F.silu(x).permute(0, 2, 1)

        x_dbl = self.x_proj(x)
        dt, B_val, C_val = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt).permute(0, 2, 1)

        A = -torch.exp(self.A_log.float())
        y = selective_scan_fn(
            x, dt, A, B_val.contiguous(), C_val.contiguous(), self.D.float(),
            z=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
            return_last_state=False
        )
        
        y = y * F.silu(z)
        return self.out_proj(y)

# 模块 2: 自注意力封装器，以统一接口
class SelfAttentionWrapper(nn.Module):
    """
    将 PyTorch 原生的 TransformerEncoderLayer 封装起来，
    使其 forward 接口与其他混合器（如 MambaMixer）保持一致，
    并能够正确处理 src_key_padding_mask。
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.attn = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
    
    def forward(self, src, src_key_padding_mask=None):
        return self.attn(src, src_key_padding_mask=src_key_padding_mask)

# 模块 3: 最终的、统一的 DecoderBlock 模块
class DecoderBlock(nn.Module):
    """
    一个完整的处理块，整合了 Norm -> Mixer -> DropPath -> Norm -> MLP 的流程。
    这是模型的基本构建单元。
    """
    def __init__(self, dim, mixer_module, mlp_ratio=4., drop=0., drop_path=0., layer_scale=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = mixer_module
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)
        
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x, **mixer_kwargs):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x), **mixer_kwargs))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x