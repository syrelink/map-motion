import torch
import torch.nn as nn
from omegaconf import DictConfig

# 导入自定义模块和工具函数
from models.base import Model
from models.modules import PositionalEncoding, TimestepEmbedder
from models.modules import SceneMapEncoderDecoder, SceneMapEncoder
from models.functions import load_and_freeze_clip_model, encode_text_clip, \
    load_and_freeze_bert_model, encode_text_bert, get_lang_feat_dim_type
from utils.misc import compute_repr_dimesion
from models.MambaVision import MotionMambaMixer, Attention
from timm.models.layers import DropPath, Mlp
from models.motion_vrwkv import *


@Model.register()
class CMDM(nn.Module):

    def __init__(self, cfg: DictConfig, *args, **kwargs):
        """
        模型的初始化函数，负责搭建模型的全部网络结构。
        cfg: 来自Hydra的配置对象，包含了所有超参数。
        """
        super().__init__()
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

        # --- 1. 基本配置 ---
        self.motion_type = cfg.data_repr  # 动作数据的表示类型
        self.motion_dim = cfg.input_feats  # 动作特征的维度
        self.latent_dim = cfg.latent_dim  # Transformer内部工作的统一特征维度
        self.mask_motion = cfg.mask_motion  # 是否对动作序列的填充部分进行mask

        self.arch = cfg.arch  # 核心架构选择: 'trans_enc' 或 'trans_dec'

        # --- 2. 时间步嵌入模块 (Time Embedding) ---
        # 对于扩散模型，需要将时间步t编码成一个特征向量
        self.time_emb_dim = cfg.time_emb_dim
        self.timestep_embedder = TimestepEmbedder(self.latent_dim, self.time_emb_dim, max_len=1000)

        # --- 3. 场景接触信息编码模块 (Contact Encoder) ---
        self.contact_type = cfg.contact_model.contact_type
        self.contact_dim = compute_repr_dimesion(self.contact_type)
        self.planes = cfg.contact_model.planes
        # 根据架构选择不同的场景编码器
        if self.arch == 'trans_enc':
            # Encoder架构将场景信息作为序列的一部分，需要一个简单的编码器
            SceneMapModule = SceneMapEncoder
            # 定义一个线性层作为“适配器”，将场景特征维度映射到Transformer的内部维度
            self.contact_adapter = nn.Linear(self.planes[-1], self.latent_dim, bias=True)
        elif self.arch == 'trans_dec':
            # Decoder架构需要将场景信息作为memory，需要更复杂的编码器结构
            SceneMapModule = SceneMapEncoderDecoder
        elif self.arch == 'trans_DCA':
            # Decoder架构需要将场景信息作为memory，需要更复杂的编码器结构
            SceneMapModule = SceneMapEncoderDecoder
        else:
            raise NotImplementedError
        self.contact_encoder = SceneMapModule(
            point_feat_dim=self.contact_dim,
            planes=self.planes,
            blocks=cfg.contact_model.blocks,
            num_points=cfg.contact_model.num_points,
        )

        # --- 4. 文本指令编码模块 (Text Encoder) ---
        self.text_model_name = cfg.text_model.version
        self.text_max_length = cfg.text_model.max_length
        # 根据配置选择使用CLIP还是BERT
        self.text_feat_dim, self.text_feat_type = get_lang_feat_dim_type(self.text_model_name)
        if self.text_feat_type == 'clip':
            self.text_model = load_and_freeze_clip_model(self.text_model_name)
        elif self.text_feat_type == 'bert':
            self.tokenizer, self.text_model = load_and_freeze_bert_model(self.text_model_name)
        else:
            raise NotImplementedError
        # 同样定义一个“适配器”，将文本特征维度映射到Transformer的内部维度
        self.language_adapter = nn.Linear(self.text_feat_dim, self.latent_dim, bias=True)

        # --- 5. Transformer核心架构 ---
        # 动作数据本身的“适配器”
        self.motion_adapter = nn.Linear(self.motion_dim, self.latent_dim, bias=True)
        # 位置编码器，为序列中的每个token提供位置信息
        self.positional_encoder = PositionalEncoding(self.latent_dim, dropout=0.1, max_len=5000)

        self.num_layers = cfg.num_layers
        if self.arch == 'trans_enc':
            # 如果是Encoder架构，直接使用PyTorch官方的TransformerEncoder
            self.self_attn_layer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,  # 内部维度
                    nhead=cfg.num_heads,  # 多头注意力的头数
                    dim_feedforward=cfg.dim_feedforward,  # 前馈网络的隐藏层维度
                    dropout=cfg.dropout,
                    activation='gelu',
                    batch_first=True,  # 输入数据的格式为 (batch_size, seq_len, dim)
                ),
                enable_nested_tensor=False,
                num_layers=sum(cfg.num_layers),  # 总层数
            )
        elif self.arch == 'trans_dec':
            # 如果是Decoder架构，则需要手动搭建自注意力和交叉注意力的交替结构
            self.self_attn_layers = nn.ModuleList()
            self.kv_mappling_layers = nn.ModuleList()
            self.cross_attn_layers = nn.ModuleList()
            for i, n in enumerate(self.num_layers):
                # 1. 自注意力层 (Self-Attention)
                self.self_attn_layers.append(
                    nn.TransformerEncoder(  # 这里用Encoder实现自注意力
                        nn.TransformerEncoderLayer(
                            d_model=self.latent_dim,
                            nhead=cfg.num_heads,
                            dim_feedforward=cfg.dim_feedforward,
                            dropout=cfg.dropout,
                            activation='gelu',
                            batch_first=True,
                        ),
                        num_layers=n,
                    )
                )
                # 2. 交叉注意力层 (Cross-Attention)
                if i != len(self.num_layers) - 1:  # 最后一层之后不需要交叉注意力
                    # 将场景特征映射为K和V
                    self.kv_mappling_layers.append(
                        nn.Sequential(
                            nn.Linear(self.planes[-1 - i], self.latent_dim, bias=True),
                            nn.LayerNorm(self.latent_dim),
                        )
                    )
                    # 使用PyTorch官方的TransformerDecoderLayer实现交叉注意力
                    self.cross_attn_layers.append(
                        nn.TransformerDecoderLayer(
                            d_model=self.latent_dim,
                            nhead=cfg.num_heads,
                            dim_feedforward=cfg.dim_feedforward,
                            dropout=cfg.dropout,
                            activation='gelu',
                            batch_first=True,
                        )
                    )
# ==================== 新增的 trans_DCA 架构 ====================
        elif self.arch == 'trans_DCA':
            # --- DCA 架构初始化 ---
            # 本架构深度借鉴 MambaVision，采用 Mamba 和 Transformer 的混合设计。
            # 核心策略是 "Mamba First, Transformer Last"，即在网络的浅层使用高效的 Mamba 捕捉局部动态，
            # 在深层使用 Transformer 捕捉全局长距离依赖，这被证明是性能最佳的组合。

            # 我们将构建一个统一的层列表，每个层都是一个完整的块 (Mixer -> MLP)。
            # 为了实现这一点，我们需要为每个块的组件分别创建 ModuleList。
            self.dca_norm1s = nn.ModuleList()      # 每个块的第一个 LayerNorm
            self.dca_mixers = nn.ModuleList()      # 存储 MambaMixer 或 Attention
            self.dca_norm2s = nn.ModuleList()      # 每个块的第二个 LayerNorm
            self.dca_mlps = nn.ModuleList()        # 每个块的 MLP (前馈网络)
            self.dca_drop_paths = nn.ModuleList()  # 每个块的 DropPath

            # 计算随机深度 (Stochastic Depth) 的衰减率，用于 DropPath
            total_layers = sum(self.num_layers)
            dpr = [x.item() for x in torch.linspace(0, cfg.dropout, total_layers)]  # 线性衰减

            # MambaVision 的黄金法则：确定 Mamba 和 Transformer 的切换点
            # 我们将总层数的一半用作 Mamba，一半用作 Transformer。
            num_mamba_blocks = total_layers // 2

            # 循环构建每一层
            for i in range(total_layers):
                # 1. 添加 LayerNorms
                self.dca_norm1s.append(nn.LayerNorm(self.latent_dim))
                self.dca_norm2s.append(nn.LayerNorm(self.latent_dim))
                
                # 2. 根据当前层索引 i，决定使用 Mamba 还是 Transformer
                if i < num_mamba_blocks:
                    # 前半部分层：使用我们为动作序列定制的 MotionMambaMixer
                    mixer = MotionMambaMixer(d_model=self.latent_dim)
                else:
                    # 后半部分层：使用标准的自注意力模块 (Transformer)
                    mixer = Attention(dim=self.latent_dim, num_heads=cfg.num_heads)
                self.dca_mixers.append(mixer)

                # 3. 添加 MLP (前馈网络)
                self.dca_mlps.append(Mlp(in_features=self.latent_dim, hidden_features=int(self.latent_dim * 4), act_layer=nn.GELU, drop=cfg.dropout))

                # 4. 添加 DropPath
                self.dca_drop_paths.append(DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity())
# =============================================================
# ==================== 新增的 trans_wkv 架构 ====================
        elif self.arch == 'trans_wkv':
            # --- WKV 架构初始化 ---
            # 本架构基于 Vision-RWKV，用我们定制的 MotionWKVBlock 
            # 彻底取代了原有的自注意力和交叉注意力层。
            # 这样做旨在获得线性计算复杂度，从而在处理长动作序列时获得巨大的效率优势。

            self.wkv_layers = nn.ModuleList()
            
            # 计算随机深度 (Stochastic Depth) 的衰减率
            total_layers = sum(self.num_layers)
            dpr = [x.item() for x in torch.linspace(0, cfg.dropout, total_layers)]

            # 循环构建每一层 MotionWKVBlock
            for i in range(total_layers):
                self.wkv_layers.append(
                    MotionWKVBlock(
                        dim=self.latent_dim,
                        mlp_ratio=4., # FFN的扩展比例，通常为4
                        drop_path=dpr[i]
                    )
                )
# =================================================================
        else:
            raise NotImplementedError

        # --- 6. 输出层 ---
        # 将Transformer处理后的特征向量重新映射回原始的动作维度
        self.motion_layer = nn.Linear(self.latent_dim, self.motion_dim, bias=True)

    def forward(self, x, timesteps, **kwargs):
        """
        模型的前向传播函数，定义了数据如何在网络中流动。
        x: 带噪声的输入动作, [bs, seq_len, motion_dim]
        timesteps: 当前的扩散时间步
        kwargs: 其他条件信息，如文本、场景接触图等
        """
        # --- A. 准备各种条件的嵌入向量 ---

        # 1. 时间嵌入
        time_emb = self.timestep_embedder(timesteps)  # [bs, 1, latent_dim]
        time_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=self.device)

        # 2. 文本嵌入
        if self.text_feat_type == 'clip':
            text_emb = encode_text_clip(self.text_model, kwargs['c_text'], max_length=self.text_max_length,
                                        device=self.device)
            text_emb = text_emb.unsqueeze(1).detach().float()
            text_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=self.device)
        elif self.text_feat_type == 'bert':
            text_emb, text_mask = encode_text_bert(self.tokenizer, self.text_model, kwargs['c_text'],
                                                   max_length=self.text_max_length, device=self.device)
            text_mask = ~(text_mask.to(torch.bool))  # 转换mask标准
        else:
            raise NotImplementedError
        # (可选) 根据外部mask或擦除指令，进一步处理文本嵌入
        if 'c_text_mask' in kwargs:
            text_mask = torch.logical_or(text_mask, kwargs['c_text_mask'].repeat(1, text_mask.shape[1]))
        if 'c_text_erase' in kwargs:
            text_emb = text_emb * (1. - kwargs['c_text_erase'].unsqueeze(-1).float())
        # 将文本特征通过适配器映射到latent_dim
        text_emb = self.language_adapter(text_emb)  # [bs, text_len, latent_dim]

        # 3. 场景接触信息嵌入
        cont_emb = self.contact_encoder(kwargs['c_pc_xyz'], kwargs['c_pc_contact'])
        if hasattr(self, 'contact_adapter'):  # 仅当架构为trans_enc时存在
            cont_mask = torch.zeros((x.shape[0], cont_emb.shape[1]), dtype=torch.bool, device=self.device)
            if 'c_pc_mask' in kwargs:
                cont_mask = torch.logical_or(cont_mask, kwargs['c_pc_mask'].repeat(1, cont_mask.shape[1]))
            if 'c_pc_erase' in kwargs:
                cont_emb = cont_emb * (1. - kwargs['c_pc_erase'].unsqueeze(-1).float())
            cont_emb = self.contact_adapter(cont_emb)  # [bs, num_groups, latent_dim]

        # 4. 动作嵌入
        x = self.motion_adapter(x)  # [bs, seq_len, latent_dim]

        # --- B. 核心Transformer信息处理 ---

        if self.arch == 'trans_enc':
            # Encoder架构：将所有信息拼接成一个长序列
            x = torch.cat([time_emb, text_emb, cont_emb, x], dim=1)
            # 添加位置编码
            x = self.positional_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

            x_mask = None
            if self.mask_motion:
                x_mask = torch.cat([time_mask, text_mask, cont_mask, kwargs['x_mask']], dim=1)

            # 将拼接后的长序列送入Transformer Encoder
            x = self.self_attn_layer(x, src_key_padding_mask=x_mask)

            # 丢弃条件信息的输出部分，只保留动作部分的输出
            non_motion_token = time_mask.shape[1] + text_mask.shape[1] + cont_mask.shape[1]
            x = x[:, non_motion_token:, :]

        elif self.arch == 'trans_dec':

            # Decoder架构：将时间、文本和动作作为Query序列
            x = torch.cat([time_emb, text_emb, x], dim=1)
            x = self.positional_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

            x_mask = None
            if self.mask_motion:
                x_mask = torch.cat([time_mask, text_mask, kwargs['x_mask']], dim=1)

            # 逐层进行自注意力和交叉注意力
            for i in range(len(self.num_layers)):
                # 1. 自注意力
                x = self.self_attn_layers[i](x, src_key_padding_mask=x_mask)
                # 2. 交叉注意力（将场景信息cont_emb作为memory）
                if i != len(self.num_layers) - 1:
                    mem = cont_emb[i]  # memory
                    mem_mask = torch.zeros((x.shape[0], mem.shape[1]), dtype=torch.bool, device=self.device)
                    if 'c_pc_mask' in kwargs:
                        mem_mask = torch.logical_or(mem_mask, kwargs['c_pc_mask'].repeat(1, mem_mask.shape[1]))
                    if 'c_pc_erase' in kwargs:
                        mem = mem * (1. - kwargs['c_pc_erase'].unsqueeze(-1).float())

                    # 准备交叉注意力的K和V
                    mem = self.kv_mappling_layers[i](mem)
                    # 执行交叉注意力
                    x = self.cross_attn_layers[i](x, mem, tgt_key_padding_mask=x_mask, memory_key_padding_mask=mem_mask)

            # 丢弃条件信息的输出部分，只保留动作部分的输出
            non_motion_token = time_mask.shape[1] + text_mask.shape[1]
            x = x[:, non_motion_token:, :]
# ==================== 修正后的 trans_DCA 前向传播逻辑 ====================
        elif self.arch == 'trans_DCA':
            # --- DCA 前向传播 ---
            # 1. 准备输入序列 (与 trans_dec 相同)
            # 将时间、文本和带噪动作拼接成一个长序列
            x = torch.cat([time_emb, text_emb, x], dim=1)
            # 添加位置编码
            x = self.positional_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

            # 准备 mask (与 trans_dec 相同)
            x_mask = None
            if self.mask_motion:
                x_mask = torch.cat([time_mask, text_mask, kwargs['x_mask']], dim=1)
            
            # 2. 逐层通过 DCA 模块
            # 数据流遵循标准的 Pre-Norm 结构: x = x + DropPath(Module(Norm(x)))
            # 这与 mamba_vision.py 中的 Block 实现一致。
            for i in range(len(self.dca_mixers)):
                # 第一个残差连接：Mixer (Mamba 或 Attention)
                x = x + self.dca_drop_paths[i](self.dca_mixers[i](self.dca_norm1s[i](x)))
                
                # 第二个残差连接：MLP
                x = x + self.dca_drop_paths[i](self.dca_mlps[i](self.dca_norm2s[i](x)))

            # 3. 丢弃条件信息的输出部分，只保留动作部分的输出 (与 trans_dec 相同)
            non_motion_token = time_mask.shape[1] + text_mask.shape[1]
            x = x[:, non_motion_token:, :]
# =================================================================
# ==================== 新增的 trans_wkv 前向传播逻辑 ====================
        elif self.arch == 'trans_wkv':
            # --- WKV 前向传播 ---
            # 1. 准备输入序列 (与 trans_dec/trans_DCA 相同)
            # 将时间、文本、场景接触和带噪动作全部拼接成一个长序列
            x = torch.cat([time_emb, text_emb, cont_emb, x], dim=1)
            # 添加位置编码
            x = self.positional_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

            # WKV 架构不需要 mask，但我们保留它以防未来扩展
            x_mask = None # (可选)
            
            # 2. 逐层通过 WKV 模块
            # 每个 MotionWKVBlock 都是一个完整的处理单元，包含了 Motion-Mix 和 Channel-Mix
            for block in self.wkv_layers:
                x = block(x)

            # 3. 丢弃条件信息的输出部分，只保留动作部分的输出
            non_motion_token = time_mask.shape[1] + text_mask.shape[1] + cont_emb.shape[1]
            x = x[:, non_motion_token:, :]
# =================================================================
        else:
            raise NotImplementedError

        # --- C. 输出结果 ---
        # 将处理后的特征映射回动作维度，得到去噪后的动作
        x = self.motion_layer(x)
        return x