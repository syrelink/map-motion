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
from models.DCA import CrossDCA


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
            # 1. 自注意力模块 (Self-Attention)
            #    负责处理动作序列自身的时间依赖。我们仍然可以使用一个标准的TransformerEncoder。
            self.motion_self_attention = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=cfg.num_heads,
                    dim_feedforward=cfg.dim_feedforward,
                    dropout=cfg.dropout,
                    activation='gelu',
                    batch_first=True,
                ),
                num_layers=sum(cfg.num_layers) # 使用总层数作为深度
            )

            # 2. 交叉注意力模块 (Cross-Attention)
            #    实例化我们刚刚修正好的 CrossDCA 模块。
            self.dca_cross_attention = CrossDCA(
                query_dim=self.latent_dim,
                features=self.planes, # <-- 关键修改：将 memory_features 改为 features
                # 以下参数需要您在配置文件中定义
                patch=cfg.dca.patch,
                strides=cfg.dca.strides,
                n=cfg.dca.n_blocks,
                channel_head=cfg.dca.channel_head,
                spatial_head=cfg.dca.spatial_head
            )
        # =============================================================
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
            # 1. 准备 Query 序列 (同 trans_dec)
            #    将时间、文本和动作序列拼接在一起
            x = torch.cat([time_emb, text_emb, x], dim=1)
            x = self.positional_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)
            
            x_mask = None
            if self.mask_motion:
                x_mask = torch.cat([time_mask, text_mask, kwargs['x_mask']], dim=1)

            # 2. 先进行自注意力
            #    让包括条件在内的所有 token 互相交互，处理整体的时间/上下文逻辑
            x = self.motion_self_attention(x, src_key_padding_mask=x_mask)

            # --- 关键修正：在交叉注意力之前，分离出动作序列 ---
            
            # 3. 分离出需要与场景进行交叉注意的动作部分
            #    只有动作序列 x 才需要去查询场景地图 cont_emb
            non_motion_token = time_mask.shape[1] + text_mask.shape[1]
            motion_x = x[:, non_motion_token:, :]

            # 4. 对动作序列执行交叉注意力
            #    现在传入 dca_cross_attention 的是纯粹的 motion_x，
            #    其序列长度是固定的 motion_seq_len，可以被设计成一个完美的平方数。
            #    cont_emb 是 SceneMapEncoderDecoder 输出的特征列表
            attended_motion_x = self.dca_cross_attention(query=motion_x, memory=cont_emb)

            # 5. 直接返回经过场景增强后的动作序列
            #    因为我们已经提前分离了动作序列，所以不再需要最后一步的切片操作。
            x = attended_motion_x
        # =================================================================
        else:
            raise NotImplementedError

        # --- C. 输出结果 ---
        # 将处理后的特征映射回动作维度，得到去噪后的动作
        x = self.motion_layer(x)
        return x