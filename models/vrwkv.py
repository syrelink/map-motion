# Copyright (c) Shanghai AI Lab. All rights reserved.

from typing import Sequence
import math, os

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp

from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcls.models.builder import BACKBONES
from mmcls.models.utils import resize_pos_embed
from mmcls.models.backbones.base_backbone import BaseBackbone

from mmcls_custom.models.utils import DropPath

logger = logging.getLogger(__name__)


# --- CUDA 加速模块 ---
# 论文中提到，为了达到RNN形式的高效率，必须在CUDA层面重写算子。
# 这部分代码通过JIT（即时编译）加载一个自定义的 C++/CUDA 扩展，
# 该扩展实现了高效的双向WKV（Bi-WKV）前向和后向传播。
# C++ 和 CUDA 源代码被编译成一个动态库，PyTorch 可以直接调用。
from torch.utils.cpp_extension import load
wkv_cuda = load(name="bi_wkv", 
                sources=["mmcls_custom/models/backbones/cuda_new/bi_wkv.cpp", "mmcls_custom/models/backbones/cuda_new/bi_wkv_kernel.cu"],
                verbose=True, 
                # extra_cuda_cflags 提供了额外的编译器标志以进行优化
                extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', '-gencode arch=compute_86,code=sm_86'])

# WKV 类定义了一个自定义的 autograd 函数，将我们的 CUDA 内核与 PyTorch 的自动微分引擎连接起来。
class WKV(torch.autograd.Function):
    """
    WKV class provides a custom autograd function to bridge the CUDA kernel with PyTorch's automatic differentiation.
    """
    @staticmethod
    def forward(ctx, w, u, k, v):
        """
        前向传播函数。
        Args:
            ctx: 上下文对象，用于存储反向传播所需的信息。
            w, u, k, v: WKV 计算所需的张量 (time_decay, bonus, key, value)。
        Returns:
            计算结果 y。
        """
        # 检查输入的数据类型以支持半精度（fp16）和 bfloat16。
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        
        # ctx.save_for_backward 用于保存张量，以便在反向传播中使用。
        ctx.save_for_backward(w, u, k, v)
        
        # CUDA 内核通常期望输入是连续的 32 位浮点数。
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        
        # 调用编译好的 CUDA 前向传播函数。
        y = wkv_cuda.bi_wkv_forward(w, u, k, v)
        
        # 将输出转换回原始数据类型。
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        """
        反向传播函数。
        Args:
            ctx: 上下文对象，从中检索保存的张量。
            gy: 输出 y 的梯度。
        Returns:
            输入 w, u, k, v 的梯度。
        """
        # 从上下文中检索保存的张量。
        w, u, k, v = ctx.saved_tensors
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)

        # 调用 CUDA 反向传播函数来计算梯度。
        gw, gu, gk, gv = wkv_cuda.bi_wkv_backward(w.float().contiguous(),
                                                  u.float().contiguous(),
                                                  k.float().contiguous(),
                                                  v.float().contiguous(),
                                                  gy.float().contiguous())
        
        # 将计算出的梯度转换回原始数据类型。
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            return (gw, gu, gk, gv)


def RUN_CUDA(w, u, k, v):
    """
    一个方便的包装函数，用于调用 WKV autograd 函数。
    它确保张量在传递给 CUDA 内核之前位于正确的设备上。
    """
    return WKV.apply(w.cuda(), u.cuda(), k.cuda(), v.cuda())


def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    """
    Token-shifting 操作，用于在 2D 图像数据上模拟 "前一个时间步" 的概念。
    它将部分通道在四个方向（上、下、左、右）上移动，以实现空间信息交换。
    
    Args:
        input (torch.Tensor): 输入张量，形状为 (B, N, C)，其中 N 是 token 数量。
        shift_pixel (int): 移动的像素数量。
        gamma (float): 用于移动的通道比例。总共有 4*gamma 的通道会被移动。
        patch_resolution (tuple): patch 网格的分辨率 (H, W)。
    
    Returns:
        torch.Tensor: 经过 token-shifting 后的张量。
    """
    assert gamma <= 1/4
    B, N, C = input.shape
    # 将输入从 (B, N, C) 重塑为 (B, C, H, W) 以进行 2D 操作。
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    
    output = torch.zeros_like(input)
    
    # 将第一个 gamma 比例的通道向右移动。
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    # 将第二个 gamma 比例的通道向左移动。
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    # 将第三个 gamma 比例的通道向下移动。
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    # 将第四个 gamma 比例的通道向上移动。
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    # 剩余通道保持不变。
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    
    # 将张量展平回 (B, N, C) 形状。
    return output.flatten(2).transpose(1, 2)


class VRWKV_SpatialMix(BaseModule):
    """
    VRWKV 空间混合模块，功能类似于 Transformer 中的自注意力机制。
    它使用 WKV 算子在空间维度（token 之间）混合信息。
    """
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, init_mode='fancy', 
                 key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        
        self._init_weights(init_mode)
        
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        # 线性层用于生成 key, value, 和 receptance。
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        
        # 可选的 LayerNorm
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
            
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        # scale_init 用于特殊的权重初始化（通常设置为 0）。
        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        # with_cp 用于梯度检查点，以节省内存。
        self.with_cp = with_cp

    def _init_weights(self, init_mode):
        """
        初始化 RWKV 的核心参数：decay, first (bonus), 和 mix 权重。
        'fancy' 模式使用一种依赖于层深度的复杂初始化策略，有助于稳定训练。
        """
        if init_mode=='fancy':
            with torch.no_grad(): # fancy init
                # ratio_0_to_1 从 0 线性增加到 1，表示层在网络中的相对深度。
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1)) # 0 to 1
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                
                # fancy time_decay (空间衰减) 初始化
                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd-1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                # fancy time_first (空间初始偏置) 初始化
                zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)
                
                # fancy time_mix (k, v, r 的混合比例) 初始化
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        # 其他初始化模式（local, global）提供了更简单的权重设置。
        elif init_mode=='local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode=='global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution):
        """
        此函数包含核心的混合和线性投影逻辑。
        """
        B, T, C = x.size()
        
        # 1. 使用 token-shifting 获取 "前一个时间步" 的 token (xx)。
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            # 2. 将当前 token (x) 与移位的 token (xx) 混合，生成 xk, xv, xr。
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk, xv, xr = x, x, x

        # 3. 将混合后的 token 投影到 k, v, r。
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        # 4. 使用 sigmoid 激活 receptance (r) 以作为门控。
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            B, T, C = x.size()
            self.device = x.device

            # 生成 sr, k, v。
            sr, k, v = self.jit_func(x, patch_resolution)
            
            # 调用 CUDA 内核执行 WKV 计算。
            # 注意：decay 和 first 参数根据序列长度 T 进行了缩放。
            x = RUN_CUDA(self.spatial_decay / T, self.spatial_first / T, k, v)
            
            # 可选的 LayerNorm
            if self.key_norm is not None:
                x = self.key_norm(x)
                
            # 将 WKV 的输出与 sigmoid 门控相乘。
            x = sr * x
            # 最后进行输出投影。
            x = self.output(x)
            return x
        
        # 使用梯度检查点来节省内存。
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class VRWKV_ChannelMix(BaseModule):
    """
    VRWKV 通道混合模块，功能类似于 Transformer 中的前馈网络 (FFN)。
    它在通道维度上混合信息，同时通过 token-shifting 融入局部空间信息。
    """
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.with_cp = with_cp
        
        self._init_weights(init_mode)
        
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        # FFN 的隐藏层大小。
        hidden_sz = hidden_rate * n_embd
        
        # 线性层定义了 FFN 的结构。
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self, init_mode):
        """
        初始化通道混合的混合参数。
        """
        if init_mode == 'fancy':
            with torch.no_grad(): # fancy init of time_mix
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            # 同样使用 token-shifting 来混合局部空间信息。
            if self.shift_pixel > 0:
                xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
                xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
                xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
            else:
                xk = x
                xr = x
            
            # FFN 的核心逻辑
            k = self.key(xk)
            k = torch.square(torch.relu(k)) # 使用平方 ReLU 激活函数
            if self.key_norm is not None:
                k = self.key_norm(k)
            kv = self.value(k)
            
            # 使用 sigmoid 门控，类似于 Gated Linear Unit (GLU)。
            x = torch.sigmoid(self.receptance(xr)) * kv
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class Block(BaseModule):
    """
    VRWKV 的基础块，类似于一个 Transformer 块。
    它包含一个空间混合模块 (att) 和一个通道混合模块 (ffn)，
    以及层归一化和残差连接。
    """
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, drop_path=0., hidden_rate=4,
                 init_mode='fancy', init_values=None, post_norm=False, key_norm=False,
                 with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 第一个块之前有一个额外的层归一化。
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        # 实例化空间混合模块。
        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, shift_mode,
                                      channel_gamma, shift_pixel, init_mode,
                                      key_norm=key_norm)

        # 实例化通道混合模块。
        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, shift_mode,
                                      channel_gamma, shift_pixel, hidden_rate,
                                      init_mode, key_norm=key_norm)
        
        # LayerScale 是一种稳定训练的技术，为残差分支引入一个可学习的缩放因子。
        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
            
        self.with_cp = with_cp

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            if self.layer_id == 0:
                x = self.ln0(x)
            
            # post_norm: x + LayerNorm(SubLayer(x))
            # pre_norm: x + SubLayer(LayerNorm(x))
            if self.post_norm:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.gamma2 * self.ln2(self.ffn(x, patch_resolution)))
                else:
                    x = x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))
            else: # pre-norm
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), patch_resolution))
                else:
                    x = x + self.drop_path(self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@BACKBONES.register_module()
class VRWKV(BaseBackbone):
    """
    Vision RWKV (VRWKV) 主模型。
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 embed_dims=256,
                 depth=12,
                 drop_path_rate=0.,
                 channel_gamma=1/4,
                 shift_pixel=1,
                 shift_mode='q_shift',
                 init_mode='fancy',
                 post_norm=False,
                 key_norm=False,
                 init_values=None,
                 hidden_rate=4,
                 final_norm=True,
                 interpolate_mode='bicubic',
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_extra_tokens = 0
        self.num_layers = depth
        self.drop_path_rate = drop_path_rate

        # 1. Patch Embedding: 将输入图像转换为一系列 token。
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=True)
        
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # 2. Positional Embedding: 为每个 token 添加位置信息。
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dims))
        
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # 设置输出索引，用于特征金字塔。
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices
        
        # 3. 构建 VRWKV 块的堆栈。
        # Drop path rate 随层数线性增加。
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = ModuleList()
        for i in range(self.num_layers):
            self.layers.append(Block(
                n_embd=embed_dims,
                n_layer=depth,
                layer_id=i,
                channel_gamma=channel_gamma,
                shift_pixel=shift_pixel,
                shift_mode=shift_mode,
                hidden_rate=hidden_rate,
                drop_path=dpr[i],
                init_mode=init_mode,
                post_norm=post_norm,
                key_norm=key_norm,
                init_values=init_values,
                with_cp=with_cp
            ))

        # 4. Final Normalization: 在所有块之后应用最终的层归一化。
        self.final_norm = final_norm
        if final_norm:
            self.ln1 = nn.LayerNorm(self.embed_dims)


    def forward(self, x):
        B = x.shape[0]
        # 将图像转换为 patch token。
        x, patch_resolution = self.patch_embed(x)

        # 添加可插值的位置嵌入，以支持不同的输入分辨率。
        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        
        x = self.drop_after_pos(x)

        outs = []
        # 依次通过所有 VRWKV 块。
        for i, layer in enumerate(self.layers):
            x = layer(x, patch_resolution)

            # 在最后一层后应用最终的归一化。
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            # 如果当前层索引在输出索引中，则保存其输出作为特征图。
            if i in self.out_indices:
                B, _, C = x.shape
                # 将 token 序列重塑回 2D 特征图格式 (B, C, H, W)。
                patch_token = x.reshape(B, *patch_resolution, C)
                patch_token = patch_token.permute(0, 3, 1, 2)

                out = patch_token
                outs.append(out)
                
        return tuple(outs)