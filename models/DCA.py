# 文件名: DCA.py
# 描述: 包含DCA和修正后的CrossDCA模块的完整代码。

import torch
import torch.nn as nn
import einops

# ==============================================================================
# 辅助模块 (Helper Modules)
# ==============================================================================

class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x

class UpsampleConv(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                kernel_size=(3, 3),
                padding=(1, 1),
                norm_type=None,
                activation=False,
                scale=(2, 2),
                conv='conv') -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale,
                              mode='bilinear',
                              align_corners=True)
        if conv == 'conv':
            self.conv = conv_block(in_features=in_features,
                                    out_features=out_features,
                                    kernel_size=(1, 1),
                                    padding=(0, 0),
                                    norm_type=norm_type,
                                    activation=activation)
        elif conv == 'depthwise':
            self.conv = depthwise_conv_block(in_features=in_features,
                                    out_features=out_features,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    norm_type=norm_type,
                                    activation=activation)
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class PoolEmbedding(nn.Module):
    def __init__(self,
                pooling,
                patch,
                ) -> None:
        super().__init__()
        self.projection = pooling(output_size=(patch, patch))

    def forward(self, x):
        x = self.projection(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')
        return x

class ScaleDotProduct(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2, x3, scale):
        x2 = x2.transpose(-2, -1)
        # 调整einsum以适应多头注意力的维度 (b h c w, b h w k -> b h c k)
        x12 = torch.matmul(x1, x2) * scale
        att = self.softmax(x12)
        x123 = torch.matmul(att, x3)
        return x123

class depthwise_conv_block(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=None,
                norm_type='bn',
                activation=True,
                use_bias=True,
                pointwise=False,
                ):
        super().__init__()
        self.pointwise_flag = pointwise
        self.norm = norm_type
        self.act = activation
        self.depthwise = nn.Conv2d(
            in_channels=in_features,
            out_channels=in_features if pointwise else out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_features if groups is None else groups, # groups should be in_features for depthwise
            dilation=dilation,
            bias=use_bias)
        if self.pointwise_flag:
            self.pointwise_conv = nn.Conv2d(in_features,
                                        out_features,
                                        kernel_size=(1, 1),
                                        stride=(1, 1),
                                        padding=(0, 0),
                                        dilation=(1, 1),
                                        bias=use_bias)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm_layer = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm_layer = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.depthwise(x)
        if self.pointwise_flag:
            x = self.pointwise_conv(x)
        if self.norm_type is not None:
            x = self.norm_layer(x)
        if self.act:
            x = self.relu(x)
        return x

class depthwise_projection(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 groups,
                 kernel_size=(1, 1),
                 padding=(0, 0),
                 norm_type=None,
                 activation=False,
                 pointwise=False) -> None:
        super().__init__()

        self.proj = depthwise_conv_block(in_features=in_features,
                                         out_features=out_features,
                                         kernel_size=kernel_size,
                                         padding=padding,
                                         groups=groups,
                                         pointwise=pointwise,
                                         norm_type=norm_type,
                                         activation=activation)

    def forward(self, x):
        P = int(x.shape[1] ** 0.5)
        x = einops.rearrange(x, 'B (H W) C-> B C H W', H=P)
        x = self.proj(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.q_map = nn.Linear(out_features, out_features, bias=False)
        self.k_map = nn.Linear(in_features, out_features, bias=False)
        self.v_map = nn.Linear(in_features, out_features, bias=False)

        self.projection = nn.Linear(out_features, out_features)
        self.sdp = ScaleDotProduct()

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q, k, v = self.q_map(q), self.k_map(k), self.v_map(v)
        b, hw, c_q = q.shape
        c_out = c_q
        scale = (c_q // self.n_heads) ** -0.5

        q = q.reshape(b, hw, self.n_heads, c_q // self.n_heads).permute(0, 2, 3, 1) # B, h, C/h, HW
        k = k.reshape(b, hw, self.n_heads, c_q // self.n_heads).permute(0, 2, 3, 1) # B, h, C/h, HW
        v = v.reshape(b, hw, self.n_heads, c_q // self.n_heads).permute(0, 2, 3, 1) # B, h, C/h, HW
        
        att = self.sdp(q, k, v, scale).permute(0, 3, 1, 2).reshape(b, hw, c_out) # B, HW, C
        att = self.projection(att)
        return att


class SpatialAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4) -> None:
        super().__init__()
        self.n_heads = n_heads

        self.q_map = nn.Linear(out_features, out_features, bias=False)
        self.k_map = nn.Linear(in_features, out_features, bias=False)
        self.v_map = nn.Linear(in_features, out_features, bias=False)

        self.projection = nn.Linear(out_features, out_features)
        self.sdp = ScaleDotProduct()

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q, k, v = self.q_map(q), self.k_map(k), self.v_map(v)

        b, hw, c_q = q.shape
        c_out = c_q
        scale = (c_q // self.n_heads) ** -0.5

        q = q.reshape(b, hw, self.n_heads, c_q // self.n_heads).permute(0, 2, 1, 3) # B, h, HW, C/h
        k = k.reshape(b, hw, self.n_heads, c_q // self.n_heads).permute(0, 2, 1, 3) # B, h, HW, C/h
        v = v.reshape(b, hw, self.n_heads, c_q // self.n_heads).permute(0, 2, 1, 3) # B, h, HW, C/h
        
        att = self.sdp(q, k, v, scale).transpose(1, 2).reshape(b, hw, c_out) # B, HW, C
        x = self.projection(att)
        return x

class CCSABlock(nn.Module):
    def __init__(self,
                 features,
                 channel_head,
                 spatial_head,
                 spatial_att=True,
                 channel_att=True) -> None:
        super().__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        if self.channel_att:
            self.channel_norm = nn.ModuleList([nn.LayerNorm(in_features, eps=1e-6) for in_features in features])
            self.c_attention = nn.ModuleList([ChannelAttention(
                in_features=sum(features),
                out_features=feature,
                n_heads=head,
            ) for feature, head in zip(features, channel_head)])
        if self.spatial_att:
            self.spatial_norm = nn.ModuleList([nn.LayerNorm(in_features, eps=1e-6) for in_features in features])
            self.s_attention = nn.ModuleList([SpatialAttention(
                in_features=sum(features),
                out_features=feature,
                n_heads=head,
            ) for feature, head in zip(features, spatial_head)])

    def forward(self, x):
        if self.channel_att:
            x_ca = self.channel_attention(x)
            x = self.m_sum(x, x_ca)
        if self.spatial_att:
            x_sa = self.spatial_attention(x)
            x = self.m_sum(x, x_sa)
        return x

    def channel_attention(self, x):
        x_c = self.m_apply(x, self.channel_norm)
        x_cin = self.cat(*x_c)
        x_in = [[q, x_cin, x_cin] for q in x_c]
        x_att = self.m_apply(x_in, self.c_attention)
        return x_att

    def spatial_attention(self, x):
        x_c = self.m_apply(x, self.spatial_norm)
        x_cin = self.cat(*x_c)
        x_in = [[v, x_cin, x_cin] for v in x_c] # Spatial attention query is v
        x_att = self.m_apply(x_in, self.s_attention)
        return x_att

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [xi + xj for xi, xj in zip(x, y)]

    def cat(self, *args):
        return torch.cat(args, dim=2)

# ==============================================================================
# 基础 DCA 模块 (Base DCA Module)
# ==============================================================================

class DCA(nn.Module):
    def __init__(self,
                 features,
                 strides=[8, 4, 2, 1],
                 patch=28,
                 channel_att=True,
                 spatial_att=True,
                 n=1,
                 channel_head=[1, 1, 1, 1],
                 spatial_head=[4, 4, 4, 4],
                 ):
        super().__init__()
        self.n = n
        self.features = features
        self.spatial_head = spatial_head
        self.channel_head = channel_head
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.patch = patch
        self.patch_avg = nn.ModuleList([PoolEmbedding(
            pooling=nn.AdaptiveAvgPool2d,
            patch=patch,
        ) for _ in features])
        self.avg_map = nn.ModuleList([depthwise_projection(in_features=feature,
                                                           out_features=feature,
                                                           kernel_size=(1, 1),
                                                           padding=(0, 0),
                                                           groups=feature
                                                           )
                                      for feature in features])

        self.attention = nn.ModuleList([
            CCSABlock(features=features,
                      channel_head=channel_head,
                      spatial_head=spatial_head,
                      channel_att=channel_att,
                      spatial_att=spatial_att)
            for _ in range(n)])

        self.upconvs = nn.ModuleList([UpsampleConv(in_features=feature,
                                                   out_features=feature,
                                                   kernel_size=(1, 1),
                                                   padding=(0, 0),
                                                   norm_type=None,
                                                   activation=False,
                                                   scale=stride,
                                                   conv='conv')
                                      for feature, stride in zip(features, strides)])
        self.bn_relu = nn.ModuleList([nn.Sequential(
            nn.BatchNorm2d(feature),
            nn.ReLU()
        ) for feature in features])

    def forward(self, raw):
        x = self.m_apply(raw, self.patch_avg)
        x = self.m_apply(x, self.avg_map)
        for block in self.attention:
            x = block(x)
        x = [self.reshape(i) for i in x]
        x = self.m_apply(x, self.upconvs)
        x_out = self.m_sum(x, raw)
        x_out = self.m_apply(x_out, self.bn_relu)
        return (*x_out,)

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [xi + xj for xi, xj in zip(x, y)]

    def reshape(self, x):
        return einops.rearrange(x, 'B (H W) C-> B C H W', H=self.patch)

# ==============================================================================
# 【修正后】的交叉注意力 DCA 模块 (Corrected CrossDCA Module)
# ==============================================================================

class CrossDCA(DCA):  # 明确继承自 DCA 类
    def __init__(self, features, query_dim, **kwargs):
        """
        用于交叉注意力的DCA模块。
        - features: memory (场景特征) 的多尺度特征维度列表。
        - query_dim: query (动作序列) 的特征维度。
        - **kwargs: 传递给父类DCA的其他参数 (如 patch, strides, n, channel_head等)。
        """
        # 1. 调用父类(DCA)的构造函数
        #    它会创建为处理多尺度 memory (场景特征) 准备的模块
        #    例如 self.patch_avg, self.avg_map, self.attention, self.upconvs
        super().__init__(features=features, **kwargs)
        
        # 2. 为单尺度的 Query (动作序列) 创建专属的处理模块
        self.query_patch_avg = PoolEmbedding(
            pooling=nn.AdaptiveAvgPool2d,
            patch=self.patch
        )
        self.query_avg_map = depthwise_projection(
            in_features=query_dim,
            out_features=query_dim,
            groups=query_dim,
            kernel_size=(1, 1),
            padding=(0, 0),
        )

        # 3. 为 Query 准备专用的上采样和后处理模块
        #    我们直接复用父类为第一个尺度创建的模块，因为Query是单尺度的
        #    假设 stride[0] 是最大的缩放比例
        if not self.upconvs or not self.bn_relu:
             raise ValueError("Parent DCA class was not initialized correctly. 'upconvs' or 'bn_relu' is missing.")
        self.query_upconv = self.upconvs[0]
        self.query_bn_relu = self.bn_relu[0]


    def forward(self, query, memory):
        """
        前向传播函数
        query: 动作序列特征, 形状 [bs, seq_len, query_dim]
        memory: 多尺度的场景特征, 一个包含多个特征图的元组/列表
        """
        # --- 1. 预处理 Memory 和 Query ---
        # 对多尺度的 Memory (场景特征) 进行处理，得到 token 序列
        mem_tokens = self.m_apply(memory, self.patch_avg)
        mem_tokens = self.m_apply(mem_tokens, self.avg_map)

        # 对单尺度的 Query (动作特征) 进行处理
        # 首先需要将 [bs, seq_len, C] -> [bs, C, H, W] 的图像格式
        B, L, C = query.shape
        H = W = int(L**0.5)
        if H * W != L:
            # 如果序列长度不是一个完美的平方数，会引发错误。
            # 您的模型设计需要保证这一点。
            # 例如，如果 latent_dim=512, 序列长度为 64, 则 H=W=8。
            raise ValueError(f"Sequence length {L} is not a perfect square. Cannot reshape to a square image.")
        query_reshaped = einops.rearrange(query, 'B (H W) C -> B C H W', H=H, W=W)

        # 将 Query 也转换成与 memory tokens 格式一致的 token 序列
        query_tokens = self.query_patch_avg(query_reshaped)
        query_tokens = self.query_avg_map(query_tokens)

        # --- 2. 执行交叉注意力 ---
        # Q 来自 query_tokens, K 和 V 来自 mem_tokens
        # 遍历每一个注意力块 (在您的配置中 n 通常为 1)
        for block in self.attention:
            # 为 Query 和 Memory 分别进行归一化
            # CCSABlock 中的 norm 层是 ModuleList，我们需要手动选择
            # Query 是单尺度的，我们用第一个 norm 层
            q_norm = block.spatial_norm[0](query_tokens)
            mem_norm_list = block.m_apply(mem_tokens, block.spatial_norm)

            # 用于生成 Key 和 Value 的源，由所有尺度的 memory 拼接而成
            kv_source = block.cat(*mem_norm_list)

            # 构建交叉注意力的输入: [[Query, Key_Source, Value_Source]]
            # 这里的输入格式是为 s_attention 模块准备的
            x_in_spatial = [[q_norm, kv_source, kv_source]]
            
            # s_attention 也是 ModuleList，我们只对 Query 进行计算，所以也只用第一个
            att_out = block.s_attention[0](x_in_spatial[0]) # 直接传入列表元素
            
            # 残差连接
            query_tokens = query_tokens + att_out
            
            # (如果需要，也可以在这里加入通道交叉注意力，逻辑类似)

        # --- 3. 后处理并返回 ---
        # 此时的 query_tokens 是经过场景信息增强后的动作 tokens
        # 将其变回 [B, C, H, W] 的图像格式
        x = self.reshape(query_tokens)

        # 上采样回原始分辨率
        x = self.query_upconv(x)

        # 与原始的 query(reshaped) 进行残差连接
        x_out = x + query_reshaped
        x_out = self.query_bn_relu(x_out)

        # 最后，重新变回 [B, L, C] 的序列格式
        x_out = einops.rearrange(x_out, 'B C H W -> B (H W) C')

        return x_out

# ==============================================================================
# 测试代码 (Test Code)
# ==============================================================================

if __name__ == '__main__':
    # --- 测试基础 DCA 模块 ---
    print("--- Testing Base DCA Module ---")
    features = [32, 64, 128, 256]
    dca_model = DCA(features=features, patch=14)
    input1 = torch.randn(2, features[0], 224, 224)
    input2 = torch.randn(2, features[1], 112, 112)
    input3 = torch.randn(2, features[2], 56, 56)
    input4 = torch.randn(2, features[3], 28, 28)

    print("Input Shapes:")
    print(f"input1: {input1.shape}")
    print(f"input2: {input2.shape}")
    print(f"input3: {input3.shape}")
    print(f"input4: {input4.shape}")

    outputs = dca_model((input1, input2, input3, input4))

    print("\nOutput Shapes:")
    for i, out in enumerate(outputs):
        print(f"output{i+1}: {out.shape}")

    # --- 测试 CrossDCA 模块 ---
    print("\n--- Testing CrossDCA Module ---")
    batch_size = 2
    seq_len = 196  # 14*14，可以开方
    query_dim = 512
    memory_features = [64, 128, 256, 512]
    
    # 实例化 CrossDCA
    cross_dca_model = CrossDCA(
        features=memory_features, 
        query_dim=query_dim,
        patch=7, # patch size for memory
        strides=[16, 8, 4, 2],
        n=1,
        channel_head=1,
        spatial_head=4
    )
    
    # 创建模拟输入
    query_input = torch.randn(batch_size, seq_len, query_dim)
    memory_input = (
        torch.randn(batch_size, memory_features[0], 112, 112),
        torch.randn(batch_size, memory_features[1], 56, 56),
        torch.randn(batch_size, memory_features[2], 28, 28),
        torch.randn(batch_size, memory_features[3], 14, 14),
    )

    print("\nCrossDCA Input Shapes:")
    print(f"Query: {query_input.shape}")
    for i, mem in enumerate(memory_input):
        print(f"Memory {i+1}: {mem.shape}")

    # 前向传播
    cross_dca_output = cross_dca_model(query_input, memory_input)
    
    print("\nCrossDCA Output Shape:")
    print(f"Output: {cross_dca_output.shape}")

    # 检查输出维度是否与输入query一致
    assert cross_dca_output.shape == query_input.shape
    print("\nTest Passed: CrossDCA output shape is correct.")