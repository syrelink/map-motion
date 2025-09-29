import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
from collections import OrderedDict
from einops import rearrange

# 导入自定义的PointTransformer模块
from models.scene_models.pointtransformer import TransitionDown, TransitionUp, PointTransformerBlock

def get_positional_encoding(max_len: int, time_emb_dim: int) -> torch.Tensor:
    """
    生成经典的sin/cos位置编码
    
    Args:
        max_len: 序列的最大长度
        time_emb_dim: 编码的维度
    
    Return:
        pe: 位置编码张量 [max_len, 1, time_emb_dim]
    """
    pe = torch.zeros(max_len, time_emb_dim)
    # position: [max_len, 1]
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    # div_term: [time_emb_dim / 2]
    div_term = torch.exp(torch.arange(0, time_emb_dim, 2).float() * (-np.log(10000.0) / time_emb_dim))
    
    # 偶数维度使用sin
    pe[:, 0::2] = torch.sin(position * div_term)
    # 奇数维度使用cos
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # 增加批次维度并调整形状以匹配输入
    pe = pe.unsqueeze(0).transpose(0, 1) # [max_len, 1, d]
    return pe

class PositionalEncoding(nn.Module):
    """
    一个模块，将位置编码添加到输入序列中。
    """
    def __init__(self, time_emb_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # register_buffer将pe张量注册为模型的一部分，但它不是可训练的参数
        self.register_buffer('pe', get_positional_encoding(max_len, time_emb_dim)) # [max_len, 1, d]

    def forward(self, x):
        # 将位置编码加到输入x上
        # x的形状: [seq_len, batch_size, dim]
        x = x + self.pe[:x.shape[0]]
        return self.dropout(x)

class TimestepEmbedder(nn.Module):
    """
    将扩散模型中的时间步t编码为特征向量。
    """
    def __init__(self, d_model, time_embed_dim, max_len=5000):
        super().__init__()
        self.register_buffer('pe', get_positional_encoding(max_len, time_embed_dim))
        self.d_model = d_model
        self.time_embed_dim = time_embed_dim
        
        # 一个小型MLP，用于学习时间步的复杂表示
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.d_model),
            nn.SiLU(), # SiLU是一种激活函数
            nn.Linear(self.d_model, self.d_model),
        )

    def forward(self, timesteps):
        # 从预计算的pe中根据时间步索引，然后通过MLP
        return self.time_embed(self.pe[timesteps]) # [bs, 1, d]

class SceneMapEncoderDecoder(nn.Module):
    """
    用于处理3D点云的编码器-解码器（U-Net）结构。
    """
    def __init__(self, point_feat_dim: int, planes: List, blocks: List, num_points: int=8192) -> None:
        super().__init__()
        self.num_points = num_points
        # 输入特征维度 = 点云自身特征维度 + 3 (x,y,z坐标)
        self.c = point_feat_dim + 3
        block = PointTransformerBlock # 使用PointTransformer的基本块

        self.in_planes, planes = self.c, planes
        share_planes = 8
        stride, nsample = [1, 4, 4, 4], [8, 16, 16, 16]

        # --- 编码器（下采样路径）---
        # 每一层都通过TransitionDown减少点的数量，并通过PointTransformerBlock提取特征
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64

        # --- 解码器（上采样路径）---
        # 每一层都通过TransitionUp增加点的数量，并融合来自编码器对应层的特征
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3], is_head=True)
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])
    
    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        """构建单个编码器层"""
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        """构建单个解码器层"""
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)
        
    def forward(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # p: 点的坐标 [b, n, 3], x: 点的特征 [b, n, c]
        
        # 处理变长点云的批次，计算每个点云的偏移量
        offset, count = [], 0
        for item in p:
            count += item.shape[0]
            offset.append(count)
        
        # 将批次数据展平
        p0 = rearrange(p, 'b n c -> (b n) c')
        x0 = rearrange(x, 'b n c -> (b n) c')
        o0 = torch.IntTensor(offset).to(p0.device)

        # 拼接坐标和特征
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        
        # --- 编码过程 ---
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])

        # --- 解码过程 (带有跳跃连接) ---
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        
        # 将不同尺度的输出特征重塑为批次形式并返回
        return [
            rearrange(x4, '(b n) c -> b n c', b=len(offset)),
            rearrange(x3, '(b n) c -> b n c', b=len(offset)),
            rearrange(x2, '(b n) c -> b n c', b=len(offset)),
            rearrange(x1, '(b n) c -> b n c', b=len(offset)),
        ]

class SceneMapEncoder(nn.Module):
    """
    仅包含编码器部分的点云处理网络。
    """
    def __init__(self, point_feat_dim: int, planes: List, blocks: List, num_points: int=8192) -> None:
        super().__init__()
        # ... (初始化代码与SceneMapEncoderDecoder的编码器部分完全相同) ...
        self.num_points = num_points
        self.c = point_feat_dim + 3
        block = PointTransformerBlock

        self.in_planes, planes = self.c, planes
        share_planes = 8
        stride, nsample = [1, 4, 4, 4], [8, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])
    
    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        # ... (与上面相同) ...
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)
        
    def forward(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # ... (前向传播的编码部分与上面完全相同) ...
        offset, count = [], 0
        for item in p:
            count += item.shape[0]
            offset.append(count)
        p0 = rearrange(p, 'b n c -> (b n) c')
        x0 = rearrange(x, 'b n c -> (b n) c')
        o0 = torch.IntTensor(offset).to(p0.device)

        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])

        # 只返回最深层次的编码特征
        return rearrange(x4, '(b n) c -> b n c', b=len(offset))


# --- 以下是Transformer核心组件 ---

KVCache = Tuple[torch.Tensor, torch.Tensor] # 类型提示，用于缓存K和V

class RotaryPositionEmbedding:
    """
    旋转位置编码 (RoPE)
    """
    def __init__(self, frq_pos_enc: torch.Tensor, right_align: bool = False):
        self.frq_pos_enc = rearrange(frq_pos_enc, "b n c -> b 1 n c")
        self.rotate_dim = frq_pos_enc.shape[-1]
        self.right_align = right_align

    def rotate(self, t):
        """根据位置旋转输入张量t"""
        seq_len = t.shape[-2]
        # 根据对齐方式选择位置编码的部分
        pos_enc = self.frq_pos_enc[..., -seq_len:, :] if self.right_align else self.frq_pos_enc[..., :seq_len, :]
        
        # 将输入t分为旋转部分和不旋转部分
        t_rot, t_pass = t[..., : self.rotate_dim], t[..., self.rotate_dim :]
        
        # 应用旋转公式
        t_rot = (t_rot * pos_enc.cos()) + (self._rotate_half(t_rot) * pos_enc.sin())
        
        return torch.cat((t_rot, t_pass), dim=-1)

    @staticmethod
    def _rotate_half(x):
        """辅助函数，用于将特征维度两两一组进行翻转和取反，实现旋转"""
        x = rearrange(x, "... (c r) -> ... c r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... c r -> ... (c r)")


class ModuleOutput(OrderedDict):
    """一个自定义字典，允许通过属性访问其键值"""
    def __getattr__(self, name):
        try: return self[name]
        except KeyError: raise AttributeError(f"No such attribute: {name}")
    def __setattr__(self, name, value): self[name] = value


class Residual(nn.Module):
    """残差连接模块，将输入加到模块的输出上"""
    def __init__(self, module: nn.Module, dropout: float = 0.0):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(dropout)

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        # output.last_hidden_state是模块的主要输出
        output.last_hidden_state = self.dropout(output.last_hidden_state) + args[0]
        return output


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制的核心实现。
    """
    def __init__(self, num_heads, num_q_input_channels, num_kv_input_channels, **kwargs):
        super().__init__()
        # ... (参数初始化) ...
        num_qk_channels = kwargs.get('num_qk_channels') or num_q_input_channels
        num_v_channels = kwargs.get('num_v_channels') or num_qk_channels
        num_output_channels = kwargs.get('num_output_channels') or num_q_input_channels
        
        self.num_heads = num_heads
        num_qk_channels_per_head = num_qk_channels // num_heads
        self.dp_scale = num_qk_channels_per_head**-0.5 # 缩放因子
        
        # 线性层，用于将输入投影到Q, K, V
        self.q_proj = nn.Linear(num_q_input_channels, num_qk_channels, bias=kwargs.get('qkv_bias', True))
        self.k_proj = nn.Linear(num_kv_input_channels, num_qk_channels, bias=kwargs.get('qkv_bias', True))
        self.v_proj = nn.Linear(num_kv_input_channels, num_v_channels, bias=kwargs.get('qkv_bias', True))
        # 输出线性层
        self.o_proj = nn.Linear(num_v_channels, num_output_channels, bias=kwargs.get('out_bias', True))
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.0))

        self.causal_attention = kwargs.get('causal_attention', False)

    def forward(self, x_q, x_kv, pad_mask=None, rot_pos_emb_q=None, rot_pos_emb_k=None, kv_cache=None):
        # 1. 投影到 Q, K, V
        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        # (可选) 处理KV缓存，用于高效的自回归生成
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
            kv_cache = (k, v)

        # 2. 拆分成多头
        q, k, v = (rearrange(x, "b n (h c) -> b h n c", h=self.num_heads) for x in [q, k, v])
        
        # 3. 缩放Q
        q = q * self.dp_scale

        # 4. (可选) 应用旋转位置编码
        if rot_pos_emb_q is not None: q = rot_pos_emb_q.rotate(q)
        if rot_pos_emb_k is not None: k = rot_pos_emb_k.rotate(k)
        
        # 5. 计算注意力分数
        attn = torch.einsum("b h i c, b h j c -> b h i j", q, k)
        attn_max_neg = -torch.finfo(attn.dtype).max

        # 6. 应用掩码 (padding mask 和 causal mask)
        if pad_mask is not None:
            pad_mask = rearrange(pad_mask, "b j -> b 1 1 j")
            attn.masked_fill_(pad_mask, attn_max_neg)

        if self.causal_attention:
            i, j = q.shape[2], k.shape[2]
            causal_mask = torch.ones((i, j), device=x_q.device, dtype=torch.bool).triu(j - i + 1)
            attn.masked_fill_(causal_mask, attn_max_neg)
            
        # 7. Softmax 和 Dropout
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # 8. 将注意力分数应用于V
        o = torch.einsum("b h i j, b h j c -> b h i c", attn, v)
        
        # 9. 合并多头并进行最终投影
        o = rearrange(o, "b h n c -> b n (h c)", h=self.num_heads)
        o = self.o_proj(o)

        return ModuleOutput(last_hidden_state=o, kv_cache=kv_cache)

class CrossAttention(nn.Module):
    """
    交叉注意力模块 (Pre-LayerNorm 结构)
    先进行层归一化，再进行多头注意力计算。
    """
    def __init__(self, num_heads, num_q_input_channels, num_kv_input_channels, **kwargs):
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_input_channels)
        self.kv_norm = nn.LayerNorm(num_kv_input_channels)
        self.attention = MultiHeadAttention(num_heads, num_q_input_channels, num_kv_input_channels, **kwargs)

    def forward(self, x_q, x_kv, **kwargs):
        # 先对Q和KV进行LayerNorm
        x_q_norm = self.q_norm(x_q)
        x_kv_norm = self.kv_norm(x_kv)
        return self.attention(x_q_norm, x_kv_norm, **kwargs)

class SelfAttention(nn.Module):
    """
    自注意力模块 (Pre-LayerNorm 结构)
    """
    def __init__(self, num_heads, num_channels, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(num_heads, num_channels, num_channels, **kwargs)

    def forward(self, x, **kwargs):
        # 先对输入x进行LayerNorm
        x_norm = self.norm(x)
        # Q, K, V都是x_norm
        return self.attention(x_norm, x_norm, **kwargs)

class AbstractAttentionLayer(nn.Sequential):
    """注意力层的抽象基类，用于处理KV缓存"""
    def empty_kv_cache(self, x) -> KVCache:
        # ... (用于生成空的KV缓存) ...
        k_cache = torch.empty(x.shape[0], 0, self.num_qk_channels, dtype=x.dtype, device=x.device)
        v_cache = torch.empty(x.shape[0], 0, self.num_v_channels, dtype=x.dtype, device=x.device)
        return k_cache, v_cache

    def forward(self, *args, kv_cache: Optional[KVCache] = None, **kwargs):
        # [0]是注意力模块, [1]是MLP模块
        attn_output = self[0](*args, kv_cache=kv_cache, **kwargs)
        mlp_output = self[1](attn_output.last_hidden_state)
        return ModuleOutput(last_hidden_state=mlp_output.last_hidden_state, kv_cache=attn_output.kv_cache)

class CrossAttentionLayer(AbstractAttentionLayer):
    """
    一个完整的交叉注意力层 = 交叉注意力 + 残差连接 + MLP + 残差连接
    """
    def __init__(self, num_heads, num_q_input_channels, num_kv_input_channels, **kwargs):
        cross_attn = CrossAttention(num_heads, num_q_input_channels, num_kv_input_channels, **kwargs)
        self.num_qk_channels = cross_attn.attention.num_qk_channels
        self.num_v_channels = cross_attn.attention.num_v_channels

        super().__init__(
            Residual(cross_attn, kwargs.get('residual_dropout', 0.0)),
            Residual(MLP(num_q_input_channels, kwargs.get('widening_factor', 1)), kwargs.get('residual_dropout', 0.0))
        )

class SelfAttentionLayer(AbstractAttentionLayer):
    """
    一个完整的自注意力层 = 自注意力 + 残差连接 + MLP + 残差连接
    """
    def __init__(self, num_heads, num_channels, **kwargs):
        self_attn = SelfAttention(num_heads, num_channels, **kwargs)
        self.num_qk_channels = self_attn.attention.num_qk_channels
        self.num_v_channels = self_attn.attention.num_v_channels
        
        super().__init__(
            Residual(self_attn, kwargs.get('residual_dropout', 0.0)),
            Residual(MLP(num_channels, kwargs.get('widening_factor', 1)), kwargs.get('residual_dropout', 0.0))
        )

class SelfAttentionBlock(nn.Sequential):
    """
    由多个自注意力层堆叠而成的块。
    """
    def __init__(self, num_layers, **kwargs):
        layers = [SelfAttentionLayer(**kwargs) for _ in range(num_layers)]
        super().__init__(*layers)

    def forward(self, x, kv_cache=None, **kwargs):
        # ... (处理多层之间的KV缓存传递) ...
        for layer in self:
            output = layer(x, **kwargs)
            x = output.last_hidden_state
        return ModuleOutput(last_hidden_state=x)

class MLP(nn.Sequential):
    """
    Transformer层中的前馈网络 (FFN) 部分
    结构: LayerNorm -> Linear -> GELU -> Linear
    """
    def __init__(self, num_channels: int, widening_factor: int, bias: bool = True):
        super().__init__(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, widening_factor * num_channels, bias=bias),
            nn.GELU(),
            nn.Linear(widening_factor * num_channels, num_channels, bias=bias),
        )

    def forward(self, x):
        return ModuleOutput(last_hidden_state=super().forward(x))