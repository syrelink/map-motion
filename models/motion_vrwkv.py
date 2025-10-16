# motion_wkv.py
# 本文件代码深度借鉴自 Vision-RWKV 官方代码库
# 我们将其改造为专用于 (Batch, Seq_Len, Dim) 格式的1D时序动作序列。

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# --- 关键部分 1: 加载高性能 CUDA 内核 ---
# 这是 RWKV 实现线性复杂度和高效率的核心。
# 它通过即时编译(JIT)加载一个自定义的 C++/CUDA 扩展，用于执行 WKV 计算。
# 请确保您的环境中已安装 C++ 和 CUDA 工具链，并且源文件路径正确。
try:
    wkv_cuda = load(name="bi_wkv", 
                    sources=["models/cuda/bi_wkv.cpp", "models/cuda/bi_wkv_kernel.cu"],
                    verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', '-gencode arch=compute_80,code=sm_80']) # 您可以根据您的GPU架构修改 sm_80

    class WKV(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, u, k, v):
            ctx.save_for_backward(w, u, k, v)
            y = wkv_cuda.bi_wkv_forward(w.float(), u.float(), k.float(), v.float())
            return y.to(w.dtype)

        @staticmethod
        def backward(ctx, gy):
            w, u, k, v = ctx.saved_tensors
            gw, gu, gk, gv = wkv_cuda.bi_wkv_backward(w.float(), u.float(), k.float(), v.float(), gy.float())
            return gw.to(w.dtype), gu.to(w.dtype), gk.to(w.dtype), gv.to(w.dtype)

    def RUN_CUDA(w, u, k, v):
        return WKV.apply(w, u, k, v)

except Exception as e:
    print(f"警告: CUDA WKV 内核加载失败: {e}")
    print("模型将回退到效率较低的纯 PyTorch 实现 (如果提供)。")
    RUN_CUDA = None # 如果加载失败，则设置为空

# --- 关键部分 2: 1D 时序位移函数 ---
# 原始的 q_shift 是为2D图像设计的。我们将其简化为1D时序位移。
# 它的作用是让当前时间步的 token 可以看到上一个时间步的信息，这是信息混合的关键。
def temporal_shift(x):
    """
    对 (B, T, C) 的序列在时间维度 T 上进行位移。
    """
    B, T, C = x.size()
    prev_x = torch.zeros_like(x)
    prev_x[:, 1:, :] = x[:, :T-1, :]
    return prev_x

# --- 关键部分 3: WKV 模块 ---
# 我们将原始的 SpatialMix 和 ChannelMix 概念，统一改造并封装到 MotionWKVBlock 中。

class MotionWKV_TokenMix(nn.Module):
    """
    Token-Mixing 模块 (等价于 Attention)。
    它负责在不同时间步的 token 之间混合信息。
    """
    def __init__(self, n_embd, n_layer, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd

        # WKV 的核心参数: decay, bonus, receptance, key, value
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)
            
            decay_speed = torch.ones(n_embd)
            for h in range(n_embd):
                decay_speed[h] = -5 + 8 * (h / (n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)

            zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(n_embd)]) * 0.5)
            self.time_first = nn.Parameter(torch.ones(n_embd) * torch.math.log(0.3) + zigzag)
            
            x = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                x[0, 0, i] = i / n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        
        # 通过与上一个时间步的 x 混合，来生成 k, v, r
        xx = temporal_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        
        # 使用高性能 CUDA 内核执行 WKV 计算
        if RUN_CUDA is not None:
            # 这里的 time_decay 和 time_first 经过调整以适应1D序列的性质
            x = RUN_CUDA(self.time_decay, self.time_first, k, v)
        else:
            # 如果CUDA不可用，这里可以放一个纯PyTorch的备用实现（会慢很多且消耗显存）
            raise NotImplementedError("CUDA WKV kernel is required for efficient execution.")

        x = sr * x
        x = self.output(x)
        return x

class MotionWKV_ChannelMix(nn.Module):
    """
    Channel-Mixing 模块 (等价于 FFN/MLP)。
    它负责在每个 token 内部的特征通道之间混合信息。
    """
    def __init__(self, n_embd, n_layer, layer_id, mlp_ratio=4):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)
            x = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                x[0, 0, i] = i / n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        hidden_sz = int(mlp_ratio * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x):
        # 同样通过与上一时间步的x混合来增强表达能力
        xx = temporal_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k)) # 使用 square-ReLU 激活
        kv = self.value(k)
        
        x = torch.sigmoid(self.receptance(xr)) * kv
        return x

class MotionWKVBlock(nn.Module):
    """
    一个完整的 WKV 块。
    这是将在您的 CMDM 模型中堆叠的基本单元。
    结构: Pre-LN -> TokenMix -> Residual -> Pre-LN -> ChannelMix -> Residual
    """
    def __init__(self, dim, n_layer, layer_id, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop_path = nn.Identity() if drop_path == 0. else DropPath(drop_path)

        self.att = MotionWKV_TokenMix(n_embd=dim, n_layer=n_layer, layer_id=layer_id)
        self.ffn = MotionWKV_ChannelMix(n_embd=dim, n_layer=n_layer, layer_id=layer_id, mlp_ratio=mlp_ratio)
        
    def forward(self, x):
        x = x + self.drop_path(self.att(self.ln1(x)))
        x = x + self.drop_path(self.ffn(self.ln2(x)))
        return x