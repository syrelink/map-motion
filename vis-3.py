import numpy as np
import cv2
import trimesh
import os
from numpy.linalg import norm # L2 模长计算

# --- 关键路径 (用户需要替换为实际路径) ---
# 预测的 Affordance 编码特征 (1, 8192, 6)
PREDICTED_NPY_PATH = "outputs/CDM-Perceiver-HUMANISE-step200k/eval/test-1114-125307/HUMANISE/pred_contact/02000.npy"

# 场景点 XYZ 坐标路径 (从 GT npz 文件中获取)
# 假设场景 XYZ 坐标在 GT npz 文件中，我们将从 'points' 键中提取前 3 维
SCENE_XYZ_PATH = "data/HUMANISE/contact_motion/contacts/00000.npz" 

# 输出 PLY 文件路径
OUTPUT_PLY_PATH = "visualization_output/adm_affordance_heatmap.ply"
os.makedirs(os.path.dirname(OUTPUT_PLY_PATH) or '.', exist_ok=True)


def generate_adm_heatmap(predicted_npy_path: str, scene_data_path: str, save_path: str):
    """
    将模型预测的 (1, 8192, 6) 维 Affordance 编码特征转换为彩色热力点云。
    热力值通过 L2 模长计算，并使用 cv2.COLORMAP_PARULA 进行映射。
    """
    print(f"--- 正在处理预测文件: {predicted_npy_path} ---")
    
    # 1. 加载预测的 Affordance 编码特征 (1, 8192, 6)
    try:
        predicted_features_npy = np.load(predicted_npy_path)
        feature_map = predicted_features_npy.squeeze(0) # 形状: (8192, 6)
        
        # 2. 加载场景 XYZ 坐标 (从 GT npz 文件中获取)
        with np.load(scene_data_path) as data:
            # 提取 XYZ 坐标 (8192, 3)
            xyz = data['points'][:, :3] 
            print(f"场景 XYZ 形状: {xyz.shape}")
    except FileNotFoundError as e:
        print(f"错误: 无法找到文件。请检查路径: {e}")
        return
    
    # --- 3. 核心转换：L2 模长计算 (Affordance Score) ---
    # L2 模长 (norm) 将 6 维特征抽象为单维度热力值
    affordance_scores = norm(feature_map, axis=1) # 形状: (8192,)

    # 4. 归一化 Affordance Score 到 [0, 1]
    if np.max(affordance_scores) > 0:
        normalized_affordance = affordance_scores / np.max(affordance_scores)
    else:
        normalized_affordance = np.zeros_like(affordance_scores)

    # --- 5. 应用作者的颜色映射逻辑 ---
    # a) 归一化到 [0, 255] 并转换为 uint8 类型
    # _map 现在是 0-255 的 Affordance 强度
    _map = np.uint8(255 * normalized_affordance)
    
    # b) 应用颜色映射
    heatmap = cv2.applyColorMap(_map, cv2.COLORMAP_PARULA) 
    
    # c) 颜色空间转换 (BGR to RGB) 和重塑 (N x 3)
    # 注意：trimesh 通常使用 RGB，cv2 默认输出 BGR，需要转换。
    # 作者的代码使用了 cv2.COLOR_RGB2BGR，但从功能上看，我们希望得到 RGB，所以使用 BGR2RGB 进行标准转换。
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).reshape(-1, 3) 
    
    # 6. 创建并导出彩色点云 (PLY)
    pc = trimesh.PointCloud(vertices=xyz, colors=heatmap)
    pc.export(save_path)
    
    print(f"成功生成 Affordance 热力点云文件: {save_path}")

# 执行分析
# generate_adm_heatmap(PREDICTED_NPY_PATH, SCENE_XYZ_PATH, OUTPUT_PLY_PATH)