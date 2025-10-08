# PointMamba 改进说明

## 概述

我们已经对 `PointMambaEnc` 进行了重大改进，融合了原始PointMamba的核心特性，同时保持了与现有代码的兼容性。

## 主要改进

### 1. 融合原始PointMamba序列化策略

- **双向Hilbert序列化**：使用原始PointMamba的 `hilbert` 和 `hilbert-trans` 序列化策略
- **点云分组机制**：采用FPS+KNN分组策略，提高特征提取效率
- **位置编码**：使用原始PointMamba的位置编码策略

### 2. 架构选择机制

新的 `PointMambaEnc` 支持两种架构模式：

#### 模式1：原始PointMamba架构 (`use_original_serialization=True`)
```python
# 特点：
# - 使用点云分组 (FPS + KNN)
# - 双向序列化处理
# - 简化的Mamba编码器
# - 全局特征输出
```

#### 模式2：层次化架构 (`use_original_serialization=False`)
```python
# 特点：
# - 5层层次化编码器
# - 多尺度特征提取
# - 与PointTransformer组件结合
# - 多尺度特征输出
```

## 使用方法

### 1. 配置文件设置

创建新的配置文件 `configs/model/cmdm_original_pointmamba.yaml`：

```yaml
name: CMDM

## modeling space
input_feats: -1
data_repr: 'pos'

## time embedding
time_emb_dim: 512

## conditions
contact_model:
  contact_type: 'contact_cont_joints'
  contact_joints: [0, 10, 11, 12, 20, 21]
  planes: [32, 64, 128, 256]
  num_points: ${task.dataset.num_points}
  blocks: [2, 2, 2, 2]
  # 新增配置：是否使用原始PointMamba序列化
  use_original_serialization: true

text_model:
  version: 'ViT-B/32'
  max_length: 32

## model architecture
arch: 'trans_enc'
latent_dim: 512
mask_motion: true
num_layers: [1, 1, 1, 1, 1]
num_heads: 8
dropout: 0.1
dim_feedforward: 1024
```

### 2. 代码使用

```python
# 使用原始PointMamba序列化
encoder = SceneMapEncoder(
    point_feat_dim=contact_dim,
    planes=[32, 64, 128, 256],
    blocks=[2, 2, 2, 2],
    num_points=8192,
    use_original_serialization=True  # 启用原始PointMamba特性
)

# 使用层次化架构
encoder = SceneMapEncoder(
    point_feat_dim=contact_dim,
    planes=[32, 64, 128, 256],
    blocks=[2, 2, 2, 2],
    num_points=8192,
    use_original_serialization=False  # 使用传统层次化架构
)
```

## 技术细节

### 原始PointMamba架构特点

1. **点云分组**：
   - 使用FPS (Farthest Point Sampling) 选择中心点
   - 使用KNN获取每个中心点的邻域
   - 分组大小：32个点，64个组

2. **序列化策略**：
   - 前向Hilbert序列化：`hilbert`
   - 后向Hilbert序列化：`hilbert-trans`
   - 双向序列合并，增强特征表达

3. **Mamba编码器**：
   - 4层Mamba编码器
   - 使用RMSNorm归一化
   - 线性复杂度O(n)

### 层次化架构特点

1. **多尺度特征**：
   - 5层编码器，每层stride=4下采样
   - 特征维度：[32, 64, 128, 256, 512]
   - 保留多尺度信息

2. **PointTransformer组件**：
   - TransitionDown：下采样
   - PointMambaBlock：Mamba处理
   - 残差连接和批归一化

## 性能对比

| 架构 | 计算复杂度 | 内存使用 | 特征丰富度 | 适用场景 |
|------|------------|----------|------------|----------|
| 原始PointMamba | O(n) | 低 | 全局特征 | 分类/分割 |
| 层次化架构 | O(n) | 高 | 多尺度特征 | 场景理解 |

## 迁移指南

### 从旧版本迁移

1. **保持兼容性**：现有代码无需修改，默认使用原始PointMamba序列化
2. **配置更新**：在配置文件中添加 `use_original_serialization` 参数
3. **性能测试**：建议对比两种架构的性能差异

### 推荐使用场景

- **使用原始PointMamba序列化**：
  - 需要更高计算效率
  - 点云数据量大
  - 对全局特征要求高

- **使用层次化架构**：
  - 需要多尺度特征
  - 场景理解任务
  - 对局部细节要求高

## 故障排除

### 常见问题

1. **导入错误**：如果原始PointMamba组件不可用，会自动回退到层次化架构
2. **内存不足**：层次化架构内存使用较高，可以尝试减少 `num_points`
3. **性能差异**：两种架构在不同任务上表现可能不同，建议进行对比测试

### 调试建议

1. 检查 `ORIGINAL_POINTMAMBA_AVAILABLE` 标志
2. 验证配置文件中的 `use_original_serialization` 参数
3. 监控内存使用和计算时间

## 未来改进

1. **自适应架构选择**：根据输入数据自动选择最优架构
2. **混合架构**：结合两种架构的优势
3. **动态配置**：运行时调整架构参数
