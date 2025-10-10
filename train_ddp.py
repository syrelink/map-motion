import os
import hydra
import torch
import random
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.distributed import DistributedSampler

# 导入自定义的模块
from datasets.base import create_dataset
from datasets.misc import collate_fn_general
from models.base import create_model_and_diffusion
from utils.io import mkdir_if_not_exists, Board
from utils.training import TrainLoop
from utils.misc import compute_repr_dimesion

# @hydra.main 是一个装饰器，表示这个 main 函数是程序的入口
# 它会自动读取 configs 文件夹下的配置文件（默认是 default.yaml），
# 并将所有配置项解析成一个 cfg 对象传入 main 函数。
@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    主训练函数

    Args:
        cfg: Hydra 从 .yaml 配置文件中读取并创建的配置对象
    """
    # 根据模型配置的数据表示类型（data_repr），计算出模型实际的输入特征维度
    cfg.model.input_feats = compute_repr_dimesion(cfg.model.data_repr)
    
    # --- 1. 分布式（多GPU）环境设置 ---
    # 从环境变量 "LOCAL_RANK" 中获取当前进程应该使用哪块GPU。这是由 torch.distributed.launch 或 torchrun 自动设置的。
    cfg.gpu = int(os.environ["LOCAL_RANK"])
    # 告诉 PyTorch 当前进程要使用这块GPU
    torch.cuda.set_device(cfg.gpu)
    # 定义当前进程的设备
    device = torch.device('cuda', cfg.gpu)
    # 初始化分布式通信组，'nccl' 是NVIDIA GPU之间高效通信的推荐后端
    torch.distributed.init_process_group(backend='nccl')

    # --- 2. 日志、检查点和评估目录设置 (仅在主进程中执行) ---
    # if cfg.gpu == 0: 这个判断非常重要，确保只有主进程（通常是0号GPU对应的进程）
    # 才执行创建文件夹、打印日志等操作，避免多进程冲突和重复记录。
    if cfg.gpu == 0:
        # 移除默认的logger，以使用我们自定义的设置
        logger.remove(handler_id=0)
        # 创建日志、模型检查点和评估结果的保存目录
        mkdir_if_not_exists(cfg.log_dir)
        mkdir_if_not_exists(cfg.ckpt_dir)
        mkdir_if_not_exists(cfg.eval_dir)

        # 添加一个新的日志文件处理器，将日志信息写入到 runtime.log 文件中
        logger.add(cfg.log_dir + '/runtime.log')
        # 初始化可视化工具（如 TensorBoard 或 WandB）
        Board().create_board(cfg.platform, project=cfg.project, log_dir=cfg.log_dir)

        ## 打印配置信息，开始训练
        logger.info('[Configuration]\n' + OmegaConf.to_yaml(cfg) + '\n')
        logger.info('[Train] ==> Beign training..')
    
    # --- 3. 准备训练数据集 ---
    # 调用工厂函数创建数据集实例
    train_dataset = create_dataset(cfg.task.dataset, cfg.task.train.phase, gpu=cfg.gpu)
    if cfg.gpu == 0:
        logger.info(f'Load train dataset size: {len(train_dataset)}')
    
    # 创建分布式采样器，它会智能地将数据集分片，确保每个GPU进程在每个epoch拿到不重复的数据
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    
    # 创建数据加载器 (DataLoader)
    train_dataloader = train_dataset.get_dataloader(
        sampler=train_sampler,
        batch_size=cfg.task.train.batch_size, # 每批次的数据量
        collate_fn=collate_fn_general,       # 自定义的函数，用来将单个数据样本打包成一个batch
        num_workers=cfg.task.train.num_workers, # 使用多少个子进程来预加载数据，加快速度
        pin_memory=True,                     # 将数据锁在内存中，可以加速从CPU到GPU的传输
    )

    # --- 4. 创建模型与扩散器 (这是理解扩散模型的关键！) ---
    # 调用一个工厂函数，这个函数会根据cfg中的配置：
    # 1. 创建我们之前深入分析的神经网络模型 `model` (比如 CDM 或 CMDM 类的实例，即“引擎”)
    # 2. 创建一个 `diffusion` 对象，这个对象包含了扩散模型核心的数学逻辑，如噪声调度、加噪过程、去噪采样步等 (即“驾驶员/操作手册”)
    model, diffusion = create_model_and_diffusion(cfg, device=device)
    # 将模型移动到指定的GPU上
    model.to(device)
    # 将模型中的 BatchNorm 层转换为 SyncBatchNorm，这在多GPU训练中是必要的，可以同步各个GPU上的BN统计数据
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # 使用 DistributedDataParallel (DDP) 对模型进行包装，这是实现多GPU数据并行的标准方法。
    # 它负责将数据分发到不同GPU，并在反向传播时同步梯度。
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[cfg.gpu], output_device=cfg.gpu, find_unused_parameters=True, broadcast_buffers=False)

    # --- 5. 实例化并启动训练循环 ---
    # 创建 TrainLoop 类的实例。这个类封装了整个训练过程的循环逻辑。
    # 它就像一个‘驾校教练’，接收了‘学员’(model)、‘驾驶手册’(diffusion)和‘训练场地’(dataloader)，
    # 并在指定的设备(device)上进行训练。
    TrainLoop(
        cfg=cfg.task.train,
        model=model,          # 去噪网络 (引擎)
        diffusion=diffusion,  # 扩散过程的逻辑 (驾驶员)
        dataloader=train_dataloader, # 数据提供者
        device=device,
        save_dir=cfg.ckpt_dir, # 模型保存路径
        gpu=cfg.gpu,
        is_distributed=True,
    ).run_loop() # 调用 run_loop() 方法，正式开始漫长的训练循环

    ## 训练结束
    if cfg.gpu == 0:
        Board().close() # 关闭可视化记录
        logger.info('[Train] ==> End training..')

if __name__ == '__main__':
    # --- 设置随机种子以保证实验的可复现性 ---
    # 为了让每次运行代码的结果都完全一样，需要在所有可能产生随机性的地方固定一个种子。
    SEED = 2023
    # cuDNN的确定性设置
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    # PyTorch的随机种子
    torch.manual_seed(SEED)
    # PyTorch CUDA的随机种子
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # Python内置random库的种子
    random.seed(SEED)
    # NumPy的随机种子
    np.random.seed(SEED)
    
    # 运行主函数
    main()