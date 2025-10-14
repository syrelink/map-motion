# =========================================================================================
# 辅助模块 (根据 mamba_vision.py 改编，用于构建 trans_DCA)
# 您可以将这部分代码放在模型文件的顶部，或导入它们
# =========================================================================================
from timm.models.layers import DropPath, Mlp
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat

# 1. 专为1D时序动作数据改编的 Mamba Mixer
class MotionMambaMixer(nn.Module):
    """
    这个模块是 MambaVisionMixer 的直接改编版，专为 (B, Seq_Len, Dim) 的1D序列数据设计。
    它实现了论文中描述的双分支（SSM + 非SSM）和拼接融合的结构。
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 线性投射层
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        # Mamba SSM 分支的核心参数
        self.x_proj = nn.Linear(self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True)

        # 1D 卷积层 (非因果)
        # 使用 'same' padding 来保持序列长度不变，这对于非自回归任务至关重要。
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner // 2, out_channels=self.d_inner // 2,
            kernel_size=d_conv, bias=True,
            groups=self.d_inner // 2, padding= (d_conv - 1) // 2
        )

        # 状态空间模型 (SSM) 的参数 A 和 D
        A = repeat(torch.arange(1, self.d_state + 1), "n -> d n", d=self.d_inner // 2)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner // 2))

    def forward(self, hidden_states):
        # hidden_states: (B, L, D)
        B, L, D = hidden_states.shape

        # 1. 输入投射并拆分为两个分支 x 和 z
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1) # x 和 z 的形状都是 (B, L, D_inner/2)

        # 2. 对分支 x 应用 1D卷积 -> SiLU -> SSM
        x = x.transpose(1, 2) # (B, D_inner/2, L) 以便应用 Conv1d
        x = F.silu(self.conv1d(x))
        x = x.transpose(1, 2) # (B, L, D_inner/2)

        x_dbl = self.x_proj(x) # (B, L, dt_rank + 2*d_state)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt).transpose(1, 2) # (B, D_inner/2, L)
        B_ssm = B_ssm.transpose(1, 2).contiguous() # (B, d_state, L)
        C_ssm = C_ssm.transpose(1, 2).contiguous() # (B, d_state, L)
        A = -torch.exp(self.A_log.float()) # (D_inner/2, d_state)

        # Mamba 核心扫描操作
        y_ssm = selective_scan_fn(x.transpose(1, 2), dt, A, B_ssm, C_ssm, self.D.float(), z=None, delta_bias=None, delta_softplus=True)
        y_ssm = y_ssm.transpose(1, 2) # (B, L, D_inner/2)

        # 3. 对分支 z 仅应用 SiLU (模拟非 SSM 路径)
        z = F.silu(z)

        # 4. 拼接 (Concatenate) 两个分支的输出
        # 这是 MambaVision Mixer 的关键设计，保留了两种路径的信息。
        y = torch.cat([y_ssm, z], dim=-1)

        # 5. 输出投射
        out = self.out_proj(y)
        return out

# 2. 一个标准的自注意力模块 (可以直接从 mamba_vision.py 或 timm 库中获取)
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # 使用 PyTorch 内置的高效 Flash Attention
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x