import torch
import torch.nn as nn
from einops import rearrange

from mamba_ssm.modules.mamba_simple import Mamba
from .serialization import Point
from ..scene_models.pointtransformer import TransitionDown, TransitionUp


class PointMambaBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16, grid_size=0.02):
        super(PointMambaBlock, self).__init__()
        self.grid_size = grid_size
        self.linear_in = nn.Linear(in_planes, planes, bias=False)
        self.bn_in = nn.BatchNorm1d(planes)
        self.mixer = Mamba(d_model=planes, d_state=16, d_conv=4, expand=2)
        self.linear_out = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn_out = nn.BatchNorm1d(planes * self.expansion)
        self.act = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo
        identity = x

        x = self.linear_in(x)
        x = self.bn_in(x)
        x = self.act(x)

        point = Point(coord=p, feat=x, offset=o, grid_size=self.grid_size)
        point.serialization(order=["hilbert"])

        x_seq = point.feat[point.serialized_order[0]]
        inv = point.serialized_inverse[0]

        counts = torch.diff(o, prepend=o.new_zeros(1))
        parts = torch.split(x_seq, counts.tolist())
        x_pad = nn.utils.rnn.pad_sequence(parts, batch_first=True)
        x_pad = self.mixer(x_pad)
        x_seq = torch.cat([x_pad[i, :n, :] for i, n in enumerate(counts)])

        x = x_seq[inv]
        x = self.linear_out(x)
        x = self.bn_out(x)
        x = x + identity
        x = self.act(x)
        return [p, x, o]


class PointMambaEnc(nn.Module):
    def __init__(self, block, blocks, c=6, num_points=8192, grid_size=0.02, use_original_serialization=True):
        super().__init__()
        self.num_points = num_points
        self.c = c
        self.grid_size = grid_size
        self.use_original_serialization = use_original_serialization
        
        if use_original_serialization and ORIGINAL_POINTMAMBA_AVAILABLE:
            # 使用原始PointMamba的架构
            self._init_original_pointmamba()
        else:
            # 使用当前的层次化架构
            self._init_hierarchical_architecture(block, blocks)

    def _init_original_pointmamba(self):
        """初始化原始PointMamba架构"""
        self.trans_dim = 256
        self.depth = 4
        
        # 点云分组
        self.group_size = 32
        self.num_group = 64
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        
        # 编码器
        self.encoder = Encoder(encoder_channel=256)
        
        # 位置编码
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        
        # Mamba编码器
        self.blocks = MixerModel(
            d_model=self.trans_dim,
            n_layer=self.depth,
            rms_norm=True
        )
        
        # 输出适配器
        self.output_adapter = nn.Linear(self.trans_dim, 256)

    def _init_hierarchical_architecture(self, block, blocks):
        """初始化层次化架构（原始实现）"""
        self.in_planes, planes = self.c, [32, 64, 128, 256, 512]
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]

        self.enc1 = self._make_enc(block, planes[0], blocks[0], stride=stride[0], nsample=nsample[0], grid_size=self.grid_size)
        self.enc2 = self._make_enc(block, planes[1], blocks[1], stride=stride[1], nsample=nsample[1], grid_size=self.grid_size)
        self.enc3 = self._make_enc(block, planes[2], blocks[2], stride=stride[2], nsample=nsample[2], grid_size=self.grid_size)
        self.enc4 = self._make_enc(block, planes[3], blocks[3], stride=stride[3], nsample=nsample[3], grid_size=self.grid_size)
        self.enc5 = self._make_enc(block, planes[4], blocks[4], stride=stride[4], nsample=nsample[4], grid_size=self.grid_size)

    def _make_enc(self, block, planes, blocks, stride=1, nsample=16, grid_size=0.02):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, nsample=nsample, grid_size=grid_size))
        return nn.Sequential(*layers)

    def forward(self, p, x=None):
        if self.use_original_serialization and ORIGINAL_POINTMAMBA_AVAILABLE:
            return self._forward_original_pointmamba(p, x)
        else:
            return self._forward_hierarchical(p, x)

    def _forward_original_pointmamba(self, p, x=None):
        """使用原始PointMamba的前向传播"""
        # 1. 点云分组
        neighborhood, center = self.group_divider(p)
        
        # 2. 局部特征提取
        group_tokens = self.encoder(neighborhood)  # B G N
        pos_embed = self.pos_embed(center)  # B G C
        
        # 3. 双向序列化处理（原始PointMamba的核心特性）
        _, _, _, tokens_forward, pos_forward = serialization_func(
            center, group_tokens, pos_embed, 'hilbert'
        )
        _, _, _, tokens_backward, pos_backward = serialization_func(
            center, group_tokens, pos_embed, 'hilbert-trans'
        )
        
        # 4. 合并前向和后向序列
        tokens = torch.cat([tokens_forward, tokens_backward], dim=1)
        pos = torch.cat([pos_forward, pos_backward], dim=1)
        
        # 5. Mamba处理
        x = tokens + pos
        x = self.blocks(x)
        
        # 6. 输出适配
        output = self.output_adapter(x.mean(1))  # 全局平均池化
        return output

    def _forward_hierarchical(self, p, x=None):
        """使用层次化架构的前向传播（原始实现）"""
        if x is not None:
            pxo = self._pack_pxo(p, x)
        else:
            pxo = p

        p0, x0, o0 = pxo
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])

        b = p.shape[0] if x is not None else int(o0.numel())
        return rearrange(x5, '(b n) d -> b n d', b=b)

    def _pack_pxo(self, p, x):
        offset, acc = [], 0
        for item in p:
            acc += item.shape[0]
            offset.append(acc)
        p0 = rearrange(p, 'b n d -> (b n) d')
        x0 = rearrange(x, 'b n d -> (b n) d')
        o0 = torch.tensor(offset, device=p0.device, dtype=torch.int32)
        return [p0, x0, o0]


class PointMambaSeg(nn.Module):
    def __init__(self, block, blocks, c=6, num_points=8192, grid_size=0.02):
        super().__init__()
        self.num_points = num_points
        self.c = c
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]

        self.enc1 = self._make_enc(block, planes[0], blocks[0], stride=stride[0], nsample=nsample[0], grid_size=grid_size)
        self.enc2 = self._make_enc(block, planes[1], blocks[1], stride=stride[1], nsample=nsample[1], grid_size=grid_size)
        self.enc3 = self._make_enc(block, planes[2], blocks[2], stride=stride[2], nsample=nsample[2], grid_size=grid_size)
        self.enc4 = self._make_enc(block, planes[3], blocks[3], stride=stride[3], nsample=nsample[3], grid_size=grid_size)
        self.enc5 = self._make_enc(block, planes[4], blocks[4], stride=stride[4], nsample=nsample[4], grid_size=grid_size)

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
            pxo = self._pack_pxo(p, x)
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

    def _pack_pxo(self, p, x):
        offset, acc = [], 0
        for item in p:
            acc += item.shape[0]
            offset.append(acc)
        p0 = rearrange(p, 'b n d -> (b n) d')
        x0 = rearrange(x, 'b n d -> (b n) d')
        o0 = torch.tensor(offset, device=p0.device, dtype=torch.int32)
        return [p0, x0, o0]


