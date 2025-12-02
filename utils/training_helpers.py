"""
训练辅助工具：EMA 和学习率调度器

这个文件提供了用于改进训练的工具：
1. EMA (Exponential Moving Average)：指数移动平均，用于稳定模型性能
2. 改进的学习率调度器：余弦退火 + warmup
"""

import torch
import torch.nn as nn
from copy import deepcopy
import math


class EMA:
    """
    Exponential Moving Average (EMA) for model parameters.

    EMA 维护模型参数的移动平均，可以提高模型的稳定性和泛化能力。
    在评估和推理时使用 EMA 参数通常能获得更好的性能。

    Args:
        model: 要应用 EMA 的模型
        decay: 衰减率，通常设置为 0.999 或 0.9999
        device: 设备
    """

    def __init__(self, model, decay=0.9999, device=None):
        self.decay = decay
        self.device = device if device is not None else torch.device('cpu')

        # 创建模型参数的 shadow copy
        self.shadow = {}
        self.original = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)

    def update(self, model):
        """
        更新 EMA 参数

        Args:
            model: 当前训练的模型
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        """
        将 EMA 参数应用到模型（用于评估）

        Args:
            model: 要应用 EMA 参数的模型
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        """
        恢复原始参数（评估后恢复训练参数）

        Args:
            model: 要恢复参数的模型
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.original
                param.data = self.original[name]
        self.original = {}

    def state_dict(self):
        """返回 EMA 的状态字典"""
        return {
            'decay': self.decay,
            'shadow': self.shadow
        }

    def load_state_dict(self, state_dict):
        """加载 EMA 的状态字典"""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']


class CosineAnnealingWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """
    带 Warmup 的余弦退火学习率调度器

    学习率变化：
    1. Warmup 阶段 (0 to warmup_epochs)：线性增长从 0 到 base_lr
    2. 余弦退火阶段 (warmup_epochs to max_epochs)：余弦衰减从 base_lr 到 min_lr

    Args:
        optimizer: 优化器
        warmup_epochs: warmup 的 epoch 数
        max_epochs: 总的 epoch 数
        min_lr: 最小学习率
        warmup_start_lr: warmup 起始学习率
        last_epoch: 上一个 epoch 的索引
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        min_lr=1e-6,
        warmup_start_lr=1e-6,
        last_epoch=-1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """计算当前学习率"""
        if self.last_epoch < self.warmup_epochs:
            # Warmup 阶段：线性增长
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * self.last_epoch / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # 余弦退火阶段
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


def create_optimizer_and_scheduler(model, cfg):
    """
    创建优化器和学习率调度器的便捷函数

    Args:
        model: 模型
        cfg: 配置对象，应包含：
            - optimizer.type: 优化器类型 ('adam' 或 'adamw')
            - optimizer.lr: 学习率
            - optimizer.weight_decay: 权重衰减
            - scheduler.type: 调度器类型 ('cosine' 或 None)
            - scheduler.warmup_epochs: warmup epochs
            - scheduler.max_epochs: 最大 epochs
            - scheduler.min_lr: 最小学习率

    Returns:
        optimizer, scheduler
    """
    # 创建优化器
    optimizer_type = cfg.optimizer.type.lower()
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=getattr(cfg.optimizer, 'weight_decay', 0.0)
        )
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=getattr(cfg.optimizer, 'weight_decay', 0.01)
        )
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")

    # 创建学习率调度器
    scheduler = None
    if hasattr(cfg, 'scheduler') and cfg.scheduler.type is not None:
        scheduler_type = cfg.scheduler.type.lower()
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingWarmupLR(
                optimizer,
                warmup_epochs=getattr(cfg.scheduler, 'warmup_epochs', 10),
                max_epochs=cfg.scheduler.max_epochs,
                min_lr=getattr(cfg.scheduler, 'min_lr', 1e-6),
                warmup_start_lr=getattr(cfg.scheduler, 'warmup_start_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=getattr(cfg.scheduler, 'step_size', 30),
                gamma=getattr(cfg.scheduler, 'gamma', 0.1)
            )
        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")

    return optimizer, scheduler


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    ema,
    epoch,
    save_path,
    best_metric=None
):
    """
    保存训练检查点

    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        ema: EMA 对象
        epoch: 当前 epoch
        save_path: 保存路径
        best_metric: 最佳指标值
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()

    if best_metric is not None:
        checkpoint['best_metric'] = best_metric

    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    scheduler=None,
    ema=None,
    device='cpu'
):
    """
    加载训练检查点

    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        ema: EMA 对象
        device: 设备

    Returns:
        epoch, best_metric
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', None)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if ema is not None and 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])

    return epoch, best_metric
