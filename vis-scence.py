import numpy as np
import cv2
import trimesh
import os

print("--- 开始生成热力图 ---")

# --- 1. 定义你的文件路径 ---

# (请修改) 场景点云文件 (包含 'points' 的 .npz 文件)
# *** 假设 000019 号预测对应 000019 号场景 ***
scene_file = 'data/H3D/contacts/000019.npz'

# (请修改) 你的模型预测 .npy 文件
pred_file = 'outputs/2025-10-31_16-16-04_CDM-Perceiver-H3D/eval/test-1102-204431/H3D/pred_contact/000019-0.npy'

# (可选) 定义你希望保存的热力图文件路径
save_path = './heatmap_contact_000019.ply'

# --- 2. 准备数据 ---

try:
    # 2.1 加载场景 (xyz)
    print(f"加载场景: {scene_file}")
    scene_data = np.load(scene_file)
    xyz = scene_data['points']  # 形状 (8192, 3)
    scene_data.close()

    # 2.2 加载并处理你的预测数据 (pred_contact)
    print(f"加载预测数据: {pred_file}")
    pred_logits = np.load(pred_file)  # 形状 (1, 8192, 6)

    # a. 去掉 '1' (Batch) 维度
    pred_logits = pred_logits.squeeze(0)  # 形状 (8192, 6)

    # b. (重要) 将 Logits 转换为 [0, 1] 范围的概率
    # 使用 Sigmoid 函数: 1 / (1 + exp(-x))
    pred_probs = 1 / (1 + np.exp(-pred_logits))  # 形状 (8192, 6)

    # c. (重要) 将 6 个通道合成为 1 个 affordance 值
    # 我们取 6 个身体部位中，概率最高的那一个作为这个点的 affordance 值
    _map = pred_probs.max(axis=1)  # 形状 (8192,)

    print(f"数据处理完成。 xyz 维度: {xyz.shape}, _map 维度: {_map.shape}")

    # --- 3. 运行你的可视化代码 ---

    print("应用色彩映射 (colormap)...")

    # _map = np.uint8(255 * _map)
    # 你的原始代码：将 [0, 1] 浮点数转为 [0, 255] 整数
    _map_uint8 = np.uint8(255 * _map)

    # heatmap = cv2.applyColorMap(_map, cv2.COLORMAP_PARULA)
    # 将灰度图应用 PARULA 色彩方案，输出是 BGR 格式
    heatmap_bgr = cv2.applyColorMap(_map_uint8, cv2.COLORMAP_PARULA)

    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR).reshape(-1, 3)
    # (注意：这里做了个小修正)
    # applyColorMap 输出是 BGR, trimesh 需要 RGB。
    # 所以我们应该用 BGR2RGB (蓝绿红 -> 红绿蓝)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3)

    # trimesh.PointCloud(vertices=xyz, colors=heatmap).export(save_path)
    # 创建带颜色的点云对象并导出
    print(f"导出点云到: {save_path}")
    trimesh.PointCloud(vertices=xyz, colors=heatmap_rgb).export(save_path)

    print("--- 成功！热力图已生成。 ---")

except FileNotFoundError:
    print(f"错误: 找不到文件！请仔细检查你的 'scene_file' 和 'pred_file' 路径。")
except KeyError:
    print(f"错误: 在 .npz 文件中找不到 'points' 键。请检查 {scene_file} 是否正确。")
except Exception as e:
    print(f"发生了一个错误: {e}")