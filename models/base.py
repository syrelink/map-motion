import torch.nn as nn
from typing import Dict, List
from omegaconf import DictConfig

from utils.registry import Registry

# 创建一个名为 'model' 的注册表 (Registry)。
# 注册表是一种设计模式，可以把它想象成一个“产品目录”。
# 任何被 @Model.register() 装饰的类，都会被自动添加到这个目录中，
# 之后就可以通过字符串名字来方便地查找和创建实例。
Model = Registry('model')

def create_model(cfg: DictConfig, *args, **kwargs) -> nn.Module:
    """
    根据配置文件(cfg)创建一个神经网络模型实例。
    这是一个模型工厂函数。

    Args:
        cfg: Hydra的配置对象
    
    Return:
        一个PyTorch模型 (nn.Module)
    """
    # 1. 从配置中获取模型的名字 (例如 'CDM' 或 'CMDM')。
    # 2. Model.get(cfg.model.name) 会从“产品目录”中找到这个名字对应的类。
    # 3. (...)(cfg.model, *args, **kwargs) 会使用模型相关的配置来实例化这个类。
    return Model.get(cfg.model.name)(cfg.model, *args, **kwargs)

def create_gaussian_diffusion(cfg: DictConfig, *args, **kwargs):
    """
    创建并配置高斯扩散过程处理器。
    这个函数负责组装扩散模型的所有数学逻辑和超参数。

    Args:
        cfg: Hydra的配置对象
    
    Return:
        一个配置好的 Diffusion 对象
    """
    # 从本地的 diffusion 文件夹中导入所需模块
    from diffusion import gaussian_diffusion as gd
    from diffusion.respace import SpacedDiffusion, space_timesteps

    # 只关注配置文件中 'diffusion' 部分的内容
    cfg = cfg.diffusion
    # 获取总的扩散步数，例如 1000
    steps = cfg.steps

    # 设置'时间步重采样'策略。这是一种加速技巧，允许在推理时跳步采样，
    # 例如，从1000步中只采样50步，以加快生成速度。
    if not cfg.timestep_respacing:
        timestep_respacing = [steps]
    else:
        timestep_respacing = cfg.timestep_respacing
    
    # --- 核心步骤 1: 创建噪声调度表 (Noise Schedule) ---
    # 根据指定的调度名称（如 'linear', 'cosine'）和总步数，生成一个 betas 数组。
    # betas 数组定义了在正向加噪过程中，每一步噪声的方差大小。
    betas = gd.get_named_beta_schedule(cfg.noise_schedule, steps)

    # --- 核心步骤 2: 确定模型的预测目标 ---
    # 配置文件决定了我们的神经网络（如CDM）应该预测什么。
    if not cfg.predict_xstart:
        # 如果 predict_xstart 为 False，则模型预测的是每一步添加的噪声 ε (epsilon)。
        model_type = gd.ModelMeanType.EPSILON
    else:
        # 如果为 True，则模型直接预测最终的干净图像 x_0 (start_x)。
        model_type = gd.ModelMeanType.START_X
    
    # --- 核心步骤 3: 确定损失函数类型 ---
    # 根据配置文件选择用于训练的损失函数。
    if cfg.loss_type == 'MSE':
        loss_type = gd.LossType.MSE
    elif cfg.loss_type == 'RESCALED_MSE':
        loss_type = gd.LossType.RESCALED_MSE
    elif cfg.loss_type == 'KL':
        loss_type = gd.LossType.KL
    elif cfg.loss_type == 'RESCALED_KL':
        loss_type = gd.LossType.RESCALED_KL
    
    # --- 核心步骤 4: 实例化并返回最终的 Diffusion 对象 ---
    # SpacedDiffusion 是一个封装了完整扩散逻辑的类。
    # 它接收所有配置好的参数，并提供加噪、去噪采样等核心方法。
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing), # 使用的时间步序列
        betas=betas, # 噪声调度表
        model_mean_type= model_type, # 模型的预测目标
        model_var_type=( # 模型方差的类型（通常是固定的，也可以学习）
            (
                gd.ModelVarType.FIXED_LARGE
                if not cfg.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not cfg.learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type, # 损失函数类型
        rescale_timesteps=cfg.rescale_timesteps,
    )

def create_model_and_diffusion(cfg: DictConfig, *args, **kwargs) -> nn.Module:
    """
    一个便捷的封装函数，同时创建模型和扩散器。
    这是主训练脚本 `train.py` 直接调用的函数。

    Args:
        cfg: Hydra的配置对象
    
    Return:
        一个包含 (model, diffusion) 的元组
    """
    # 调用上面的函数，分别创建“引擎”和“驾驶手册”
    model = create_model(cfg, *args, **kwargs)
    diffusion = create_gaussian_diffusion(cfg, *args, **kwargs)
    # 将创建好的两者一起返回
    return model, diffusion