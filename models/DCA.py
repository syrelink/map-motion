# 论文地址：https://arxiv.org/pdf/2303.17696.pdf
# 论文：Dual Cross-Attention for Medical Image Segmentation, Engineering Applications of Artificial Intelligence
import torch
import torch.nn as nn
import einops

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
        x12 = torch.einsum('bhcw, bhwk -> bhck', x1, x2) * scale
        att = self.softmax(x12)
        x123 = torch.einsum('bhcw, bhwk -> bhck', att, x3)
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
        self.pointwise = pointwise
        self.norm = norm_type
        self.act = activation
        self.depthwise = nn.Conv2d(
            in_channels=in_features,
            out_channels=in_features if pointwise else out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=use_bias)
        if pointwise:
            self.pointwise = nn.Conv2d(in_features,
                                        out_features,
                                        kernel_size=(1, 1),
                                        stride=(1, 1),
                                        padding=(0, 0),
                                        dilation=(1, 1),
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
        x = self.depthwise(x)
        if self.pointwise:
            x = self.pointwise(x)
        if self.norm_type is not None:
            x = self.norm(x)
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
        self.q_map = depthwise_projection(in_features=out_features,
                                          out_features=out_features,
                                          groups=out_features)
        self.k_map = depthwise_projection(in_features=in_features,
                                          out_features=in_features,
                                          groups=in_features)
        self.v_map = depthwise_projection(in_features=in_features,
                                          out_features=in_features,
                                          groups=in_features)

        self.projection = depthwise_projection(in_features=out_features,
                                               out_features=out_features,
                                               groups=out_features)
        self.sdp = ScaleDotProduct()

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)
        b, hw, c_q = q.shape
        c = k.shape[2]
        scale = c ** -0.5
        q = q.reshape(b, hw, self.n_heads, c_q // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        v = v.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        att = self.sdp(q, k, v, scale).permute(0, 3, 1, 2).flatten(2)
        att = self.projection(att)
        return att


class SpatialAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4) -> None:
        super().__init__()
        self.n_heads = n_heads

        self.q_map = depthwise_projection(in_features=in_features,
                                          out_features=in_features,
                                          groups=in_features)
        self.k_map = depthwise_projection(in_features=in_features,
                                          out_features=in_features,
                                          groups=in_features)
        self.v_map = depthwise_projection(in_features=out_features,
                                          out_features=out_features,
                                          groups=out_features)

        self.projection = depthwise_projection(in_features=out_features,
                                               out_features=out_features,
                                               groups=out_features)
        self.sdp = ScaleDotProduct()

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)
        b, hw, c = q.shape
        c_v = v.shape[2]
        scale = (c // self.n_heads) ** -0.5
        q = q.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        v = v.reshape(b, hw, self.n_heads, c_v // self.n_heads).permute(0, 2, 1, 3)
        att = self.sdp(q, k, v, scale).transpose(1, 2).flatten(2)
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
            self.channel_norm = nn.ModuleList([nn.LayerNorm(in_features,
                                                            eps=1e-6)
                                               for in_features in features])

            self.c_attention = nn.ModuleList([ChannelAttention(
                in_features=sum(features),
                out_features=feature,
                n_heads=head,
            ) for feature, head in zip(features, channel_head)])
        if self.spatial_att:
            self.spatial_norm = nn.ModuleList([nn.LayerNorm(in_features,
                                                            eps=1e-6)
                                               for in_features in features])

            self.s_attention = nn.ModuleList([SpatialAttention(
                in_features=sum(features),
                out_features=feature,
                n_heads=head,
            )
                for feature, head in zip(features, spatial_head)])

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
        x_in = [[x_cin, x_cin, v] for v in x_c]
        x_att = self.m_apply(x_in, self.s_attention)
        return x_att

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [xi + xj for xi, xj in zip(x, y)]

    def cat(self, *args):
        return torch.cat((args), dim=2)

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
        )
            for _ in features])
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
        )
            for feature in features])

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

class CrossAttentionCCSABlock(nn.Module):
    """
    一个经过改造的CCSABlock，专门用于交叉注意力。
    它接收一个单尺度的Query和一个多尺度的Memory。
    """
    def __init__(self,
                 query_features,
                 memory_features, # memory_features 是一个多尺度特征维度的列表
                 channel_head,
                 spatial_head,
                 spatial_att=True,
                 channel_att=True) -> None:
        super().__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att

        # --- 为 Query 设置 Norm 和 Attention ---
        # Query只有一个尺度
        self.query_channel_norm = nn.LayerNorm(query_features, eps=1e-6)
        self.query_spatial_norm = nn.LayerNorm(query_features, eps=1e-6)

        # --- 为 Memory 设置 Norm 和 Attention ---
        # Memory是多尺度的
        if self.channel_att:
            self.mem_channel_norm = nn.ModuleList([nn.LayerNorm(f, eps=1e-6) for f in memory_features])
            self.c_attention = ChannelAttention(
                in_features=sum(memory_features), # Key/Value 的维度
                out_features=query_features,       # Query 的维度
                n_heads=channel_head,
            )
        if self.spatial_att:
            self.mem_spatial_norm = nn.ModuleList([nn.LayerNorm(f, eps=1e-6) for f in memory_features])
            self.s_attention = SpatialAttention(
                in_features=sum(memory_features), # Key/Value 的维度
                out_features=query_features,       # Query 的维度
                n_heads=spatial_head,
            )

    def forward(self, query, memory):
        # query: [B, L, C_q]
        # memory: 一个多尺度特征的列表/元组
        
        x_out = query
        
        if self.channel_att:
            q_norm = self.query_channel_norm(query)
            mem_norm = [self.mem_channel_norm[i](mem) for i, mem in enumerate(memory)]
            
            # Key 和 Value 来自于所有尺度的 memory
            kv_source = torch.cat(mem_norm, dim=2)
            
            # 执行通道交叉注意力
            att_out = self.c_attention([q_norm, kv_source, kv_source])
            x_out = x_out + att_out # 残差连接
            
        if self.spatial_att:
            q_norm = self.query_spatial_norm(x_out) # 使用上一轮的结果
            mem_norm = [self.mem_spatial_norm[i](mem) for i, mem in enumerate(memory)]
            
            kv_source = torch.cat(mem_norm, dim=2)
            
            # 执行空间交叉注意力
            att_out = self.s_attention([q_norm, kv_source, kv_source])
            x_out = x_out + att_out # 残差连接
            
        return x_out

class CrossDCA(nn.Module):
    def __init__(self,
                 query_dim,
                 memory_features, # e.g., [64, 128, 256, 512]
                 strides=[8, 4, 2, 1],
                 patch=28,
                 n=1,
                 channel_head=1,
                 spatial_head=4,
                 ):
        super().__init__()
        self.patch = patch

        # --- 为 Memory (多尺度) 设置预处理模块 ---
        self.mem_patch_avg = nn.ModuleList([PoolEmbedding(nn.AdaptiveAvgPool2d, patch) for _ in memory_features])
        self.mem_avg_map = nn.ModuleList([
            depthwise_projection(in_features=f, out_features=f, groups=f) for f in memory_features
        ])

        # --- 核心注意力模块 ---
        # 使用我们新设计的 CrossAttentionCCSABlock
        self.attention = nn.ModuleList([
            CrossAttentionCCSABlock(
                query_features=query_dim,
                memory_features=memory_features,
                channel_head=channel_head,
                spatial_head=spatial_head)
            for _ in range(n)])

        # --- 为 Query (单尺度) 设置后处理模块 ---
        # 只需要一个上采样层，将 patch 大小的特征图恢复到原始尺寸
        # 注意：这里假设 query 的 H, W 和 memory 最大的特征图 H, W 相同
        self.query_upconv = UpsampleConv(
            in_features=query_dim,
            out_features=query_dim,
            scale=strides[0], # 使用最大的 stride 进行上采样
            conv='conv'
        )
        self.query_bn_relu = nn.Sequential(nn.BatchNorm2d(query_dim), nn.ReLU())

    def forward(self, query, memory):
        # query: 动作序列特征, [B, L, C_q]
        # memory: 多尺度场景特征, tuple of [B, C, H, W]

        # 1. 预处理 Memory
        mem_tokens = [self.mem_patch_avg[i](mem) for i, mem in enumerate(memory)]
        mem_tokens = [self.mem_avg_map[i](mem) for i, mem in enumerate(mem_tokens)]

        # 2. 预处理 Query (只需要转换格式，不需要下采样)
        B, L, C = query.shape
        H = W = int(L**0.5)
        query_reshaped = einops.rearrange(query, 'B (H W) C -> B C H W', H=H, W=W)
        # 将其转换成和memory tokens一样的token序列格式
        query_tokens = einops.rearrange(query, 'B L C -> B C L 1') # 变成伪4D张量
        query_tokens = nn.AdaptiveAvgPool2d((self.patch, self.patch))(query_tokens)
        query_tokens = einops.rearrange(query_tokens, 'B C H W -> B (H W) C')

        # 3. 执行交叉注意力循环
        att_query_tokens = query_tokens
        for block in self.attention:
            att_query_tokens = block(att_query_tokens, mem_tokens)
        
        # 4. 后处理 Query
        x = einops.rearrange(att_query_tokens, 'B (H W) C -> B C H W', H=self.patch)
        x = self.query_upconv(x)
        
        # 5. 残差连接并返回
        x_out = x + query_reshaped
        x_out = self.query_bn_relu(x_out)
        
        # 变回序列格式
        x_out = einops.rearrange(x_out, 'B C H W -> B (H W) C')
        
        return x_out
    def __init__(self, features, query_dim, **kwargs):
        # 调用父类(DCA)的构造函数，但传入的features只用于构建memory相关的部分
        super().__init__(features=features, **kwargs)
        
        # --- 核心改造 ---
        # 原始DCA的patch_avg是为memory准备的，我们需要为Query也准备一个
        # 因为Query(动作序列)只有一个尺度，所以我们只需要一个
        self.query_patch_avg = PoolEmbedding(
            pooling=nn.AdaptiveAvgPool2d,
            patch=self.patch
        )
        self.query_avg_map = depthwise_projection(
            in_features=query_dim,
            out_features=query_dim,
            groups=query_dim,
            # 确保其他参数与DCA中的设置一致
            kernel_size=(1, 1),
            padding=(0, 0),
        )
        # -----------------

    def forward(self, query, memory):
        # query: 动作序列特征 x, 形状 [bs, seq_len, query_dim]
        # memory: 多尺度的场景特征 cont_emb, 一个包含4个特征图的元组或列表
        
        # --- 1. 准备 Query 和 Memory ---
        # 对多尺度的 Memory (场景特征) 进行处理 (这部分逻辑和您原来DCA的代码一样)
        mem_x = self.m_apply(memory, self.patch_avg)
        mem_x = self.m_apply(mem_x, self.avg_map)
        
        # 对单尺度的 Query (动作特征) 进行处理
        # 首先需要将 [bs, seq_len, C] -> [bs, C, H, W] 的格式
        B, L, C = query.shape
        # 假设序列长度可以开方得到一个方形的patch，如果不行需要padding或调整patch大小
        H = W = int(L**0.5) 
        if H * W != L:
            # 如果序列长度不是平方数，这里需要更复杂的处理，例如padding
            # 为简化起见，我们先假设它是
            pass
        query_reshaped = einops.rearrange(query, 'B (H W) C -> B C H W', H=H, W=W)
        
        # 将Query也转换成token序列的形式
        q_x = self.query_patch_avg(query_reshaped)
        q_x = self.query_avg_map(q_x)
        
        # --- 2. 执行注意力 ---
        # 这是核心区别：现在Q来自query，而K和V的来源是memory
        for block in self.attention:
            # 我们需要稍微修改一下CCSABlock的逻辑，或者在这里手动构建输入
            # 这里我们选择手动构建，以避免修改您已有的CCSABlock
            
            # 分别对 q 和 mem 进行归一化
            x_c_q_list = block.m_apply([q_x], block.spatial_norm)
            x_c_mem_list = block.m_apply(mem_x, block.spatial_norm)
            
            # 用于生成Key和Value的源，由所有尺度的memory拼接而成
            kv_source = block.cat(*x_c_mem_list)
            
            # 构建交叉注意力的输入: [[Query, Key_Source, Value_Source]]
            # 这里的输入格式是为 s_attention 准备的
            x_in_spatial = [[x_c_q_list[0], kv_source, kv_source]]
            
            # 执行空间交叉注意力
            q_x = block.m_apply(x_in_spatial, block.s_attention)[0] # 输出只有一个元素
            
            # （可选）如果也想进行通道交叉注意力，可以类似地构建输入
            # x_in_channel = [[x_c_q_list[0], kv_source, kv_source]]
            # q_x_c = block.m_apply(x_in_channel, block.c_attention)[0]
            # q_x = q_x + q_x_c

        # --- 3. 后处理并返回 ---
        # 此时的q_x是经过场景信息增强后的动作token
        # 将其变回 [B, C, H, W] 的图像格式
        x = self.reshape(q_x)
        
        # 上采样回原始分辨率
        # 注意：这里的upconvs可能需要为query单独设计，因为只有一个尺度
        # 为简化，我们假设第一个upconvs就是为query准备的
        x = self.upconvs[0](x)
        
        # 残差连接：将原始的query(reshaped)与经过场景增强后的结果相加
        x_out = x + query_reshaped 
        x_out = self.bn_relu[0](x_out) # 使用第一个bn_relu
        
        # 最后，重新变回 [B, L, C] 的序列格式
        x_out = einops.rearrange(x_out, 'B C H W -> B (H W) C')
        
        return x_out

if __name__ == '__main__':
    features = [32, 64, 128, 256]
    dca_model = DCA(features=features)
    input1 = torch.randn(1, features[0], 224, 224)
    input2 = torch.randn(1, features[1], 112, 112)
    input3 = torch.randn(1, features[2], 56, 56)
    input4 = torch.randn(1, features[3], 28, 28)

    print("Input Shapes:")
    print("input1:", input1.shape)
    print("input2:", input2.shape)
    print("input3:", input3.shape)
    print("input4:", input4.shape)

    output1, output2, output3, output4 = dca_model((input1, input2, input3, input4))

    print("\nOutput Shapes:")
    print("output1:", output1.shape)
    print("output2:", output2.shape)
    print("output3:", output3.shape)
    print("output4:", output4.shape)