# 文件: models/scene_mamba/pointmamba.py
# (最终版 - 完美适配 PointTransformer 逻辑)

import torch
import torch.nn as nn
from einops import rearrange

# --- 导入我们需要的“零件” ---

# 1. 导入 Mamba 的核心实现
#    请确保您已经安装了 mamba_ssm 包: pip install mamba_ssm
from mamba_ssm.modules.mamba_simple import Mamba

# 2. 导入我们刚刚放入此目录的“新工具箱”
from .serialization import Point  # 这是一个方便的数据结构

# 3. 导入并复用 Point Transformer 的“旧工具箱”和通用模块
#    这些模块与核心架构无关，完全可以复用，非常方便！
from ..scene_models.pointops import furthestsampling, queryandgroup, interpolation
from ..scene_models.pointtransformer import TransitionDown, TransitionUp


# -----------------------------------------------------------------------------
# 步骤 1: 设计一个新的、基于Mamba的核心处理块 (PointMambaBlock) - 【适配版】
# -----------------------------------------------------------------------------
# 这个模块将代替原来的 PointTransformerBlock
class PointMambaBlock(nn.Module):
    expansion = 1  # 保持与PointTransformerBlock的兼容性

    def __init__(self, in_planes, planes, share_planes=8, nsample=16, grid_size=0.02):
        """
        __init__函数签名与PointTransformerBlock完全兼容。
        它现在可以正确接收 share_planes 和 nsample 参数，即使Mamba层本身不直接使用它们。
        """
        super(PointMambaBlock, self).__init__()
        self.grid_size = grid_size  # 存储 grid_size 以便在forward中使用
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        # --- Mamba核心处理单元 ---
        self.mamba_layer = Mamba(
            d_model=planes,
            d_state=16,  # SSM状态H的维度 (可以作为超参数调整)
            d_conv=4,  # 1D卷积核宽度 (可以作为超参数调整)
            expand=2  # MLP扩展因子 (可以作为超参数调整)
        )

        self.linear2 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn2 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        """
        pxo: 一个列表 [p, x, o]，分别代表点坐标、点特征和批次偏移量
        p: (N_total, 3)  点坐标
        x: (N_total, C)  点特征
        o: (B,)          批次中每个点云的结束索引
        """
        p, x, o = pxo

        identity = x
        x_after_mlp1 = self.relu(self.bn1(self.linear1(x)))

        # --- 关键的Mamba处理流程 ---
        bs = o.shape[0]

        # 序列化
        point_data = Point(coord=p, feat=x_after_mlp1, offset=o, grid_size=self.grid_size)
        point_data.serialization(order=["hilbert"])

        x_ordered = point_data.feat[point_data.serialized_order[0]]
        inverse_order = point_data.serialized_inverse[0]

        # Mamba处理 (需要处理批次内点数不一致的情况)
        bincounts = torch.diff(o, prepend=torch.tensor([0], device=o.device))
        x_padded = nn.utils.rnn.pad_sequence(torch.split(x_ordered, bincounts.tolist()), batch_first=True)
        x_processed_padded = self.mamba_layer(x_padded)
        x_processed = torch.cat([x_processed_padded[i, :count, :] for i, count in enumerate(bincounts)])

        # 逆序列化
        x_restored = x_processed[inverse_order]

        # 残差连接
        x = self.bn2(self.linear2(x_restored))
        x += identity
        x = self.relu(x)

        return [p, x, o]


# -----------------------------------------------------------------------------
# 步骤 2: 组装成完整的 PointMamba 编码器 (PointMambaEnc)
# -----------------------------------------------------------------------------
# 这个类将完美替换掉 `afford-motion` 中的 `SceneMapEncoder`
class PointMambaEnc(nn.Module):
    def __init__(self, block, blocks, c=6, num_points=8192, grid_size=0.02):
        super().__init__()
        self.num_points = num_points
        self.c = c
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]

        self.enc1 = self._make_enc(block, planes[0], blocks[0], stride=stride[0], nsample=nsample[0],
                                   grid_size=grid_size)
        self.enc2 = self._make_enc(block, planes[1], blocks[1], stride=stride[1], nsample=nsample[1],
                                   grid_size=grid_size)
        self.enc3 = self._make_enc(block, planes[2], blocks[2], stride=stride[2], nsample=nsample[2],
                                   grid_size=grid_size)
        self.enc4 = self._make_enc(block, planes[3], blocks[3], stride=stride[3], nsample=nsample[3],
                                   grid_size=grid_size)
        self.enc5 = self._make_enc(block, planes[4], blocks[4], stride=stride[4], nsample=nsample[4],
                                   grid_size=grid_size)

    def _make_enc(self, block, planes, blocks, stride=1, nsample=16, grid_size=0.02):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, nsample=nsample, grid_size=grid_size))
        return nn.Sequential(*layers)

    def forward(self, p, x=None):
        """
        适配 CMDM 的调用方式: forward 函数现在可以接收两个参数 p 和 x。
        """
        if x is not None:
            pxo = self._prepare_pxo_from_batch(p, x)
        else:
            pxo = p

        p0, x0, o0 = pxo
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])

        bs = p.shape[0] if x is not None else len(torch.unique(o0))
        # 将 (N_total // 256, C_out) 重排回 (B, N // 256, C_out)
        return rearrange(x5, '(b n) d -> b n d', b=bs)

    def _prepare_pxo_from_batch(self, p, x):
        """ 辅助函数，将批次数据转换为扁平化的pxo格式 """
        offset, count = [], 0
        for item in p:
            count += item.shape[0]
            offset.append(count)

        p0 = rearrange(p, 'b n d -> (b n) d')
        x0 = rearrange(x, 'b n d -> (b n) d')
        o0 = torch.IntTensor(offset).to(p0.device)
        return [p0, x0, o0]


# -----------------------------------------------------------------------------
# 步骤 3: 组装成完整的 PointMamba 编码器-解码器 (PointMambaSeg)
# -----------------------------------------------------------------------------
# 这个类将替换掉 `afford-motion` 中的 `SceneMapEncoderDecoder`
class PointMambaSeg(nn.Module):
    def __init__(self, block, blocks, c=6, num_points=8192, grid_size=0.02):
        super().__init__()
        self.num_points = num_points
        self.c = c
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]

        # 编码器部分
        self.enc1 = self._make_enc(block, planes[0], blocks[0], stride=stride[0], nsample=nsample[0],
                                   grid_size=grid_size)
        self.enc2 = self._make_enc(block, planes[1], blocks[1], stride=stride[1], nsample=nsample[1],
                                   grid_size=grid_size)
        self.enc3 = self._make_enc(block, planes[2], blocks[2], stride=stride[2], nsample=nsample[2],
                                   grid_size=grid_size)
        self.enc4 = self._make_enc(block, planes[3], blocks[3], stride=stride[3], nsample=nsample[3],
                                   grid_size=grid_size)
        self.enc5 = self._make_enc(block, planes[4], blocks[4], stride=stride[4], nsample=nsample[4],
                                   grid_size=grid_size)

        # 解码器部分
        self.dec5 = self._make_dec(block, planes[4], 2, nsample=nsample[4], is_head=True, grid_size=grid_size)
        self.dec4 = self._make_dec(block, planes[3], 2, nsample=nsample[3], grid_size=grid_size)
        self.dec3 = self._make_dec(block, planes[2], 2, nsample=nsample[2], grid_size=grid_size)
        self.dec2 = self._make_dec(block, planes[1], 2, nsample=nsample[1], grid_size=grid_size)
        self.dec1 = self._make_dec(block, planes[0], 2, nsample=nsample[0], grid_size=grid_size)

    def _make_enc(self, block, planes, blocks, stride=1, nsample=16, grid_size=0.02):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, nsample=nsample, grid_size=grid_size))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, nsample=16, is_head=False, grid_size=0.02):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, nsample=nsample, grid_size=grid_size))
        return nn.Sequential(*layers)

    def forward(self, p, x=None):
        if x is not None:
            pxo = self._prepare_pxo_from_batch(p, x)
        else:
            pxo = p

        p0, x0, o0 = pxo
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])

        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]

        return [x4, x3, x2, x1]

    def _prepare_pxo_from_batch(self, p, x):
        offset, count = [], 0
        for item in p:
            count += item.shape[0]
            offset.append(count)

        p0 = rearrange(p, 'b n d -> (b n) d')
        x0 = rearrange(x, 'b n d -> (b n) d')
        o0 = torch.IntTensor(offset).to(p0.device)
        return [p0, x0, o0]