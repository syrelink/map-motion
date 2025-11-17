import numpy as np
import cv2
import trimesh
import os

# --- 1. 定义文件路径 ---

# !! 重要 !!
# 我们已经确认 00010.npz 是异常数据。
# 请在这里换上您 "contacts" 文件夹中的 "另一个" 文件，例如 00001.npz
npz_file_path = 'data/HUMANISE/contact_motion/contacts/00001.npz'

# --------------------------------------------------------------------

# 自动创建输出文件名
file_basename = os.path.splitext(os.path.basename(npz_file_path))[0]
save_dir = './visualizations'
save_path = os.path.join(save_dir, f'{file_basename}_heatmap.ply')

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# --- 2. 加载与处理数据 ---
try:
    data = np.load(npz_file_path)
    print(f"成功加载文件: {npz_file_path}")
    print("文件中包含的键 (Keys):", data.files)

    # --- 3. 提取 XYZ 坐标 ---
    # 'points' 维度为 (N, 6)，我们只取前 3 列作为 XYZ 坐标
    if 'points' not in data:
        raise KeyError("文件中未找到 'points' 键。")
    
    xyz = data['points'][:, :3]
    
    # [诊断] 检查坐标范围，这次它不应该是一条直线
    print(f"坐标 X 范围: {xyz[:, 0].min():.4f} to {xyz[:, 0].max():.4f}")
    print(f"坐标 Y 范围: {xyz[:, 1].min():.4f} to {xyz[:, 1].max():.4f}")
    print(f"坐标 Z 范围: {xyz[:, 2].min():.4f} to {xyz[:, 2].max():.4f}")

    # --- 4. 提取、聚合、归一化 Affordance 数据 ---
    if 'dist' not in data:
        raise KeyError("文件中未找到 'dist' 键。")
        
    # 'dist' 维度为 (N, 22)
    dist_map = data['dist']
    
    # 步骤 A: 聚合 (Aggregation)
    # 沿 22 个关节的维度(axis=1)取最小值，得到 (N,) 形状的数组
    _map = np.min(dist_map, axis=1) 
    
    print(f"原始 Affordance 值的范围: Min={_map.min():.4f}, Max={_map.max():.4f}")

    # 步骤 B: 归一化 (Normalization) - [关键修复]
    # 将 [Min, Max] 映射到 [0, 1] 范围
    _map_normalized = (_map - _map.min()) / (_map.max() - _map.min())
    
    # 步骤 C: 转换为 [0, 255] 的 uint8
    _map_uint8 = np.uint8(255 * _map_normalized)

    # --- 5. 颜色映射 (Color Mapping) ---
    
    # 应用 PARULA 颜色映射
    # _map_uint8 必须是 (N, 1) 的形状
    heatmap_bgr = cv2.applyColorMap(_map_uint8.reshape(-1, 1), cv2.COLORMAP_PARULA)
    
    # 将 BGR 转换为 RGB (Trimesh 需要 RGB)，并重塑为 (N, 3)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3)

    # --- 6. 创建并导出 Trimesh 点云 ---
    
    # 使用 xyz 坐标和 heatmap_rgb 颜色创建点云对象
    point_cloud = trimesh.PointCloud(vertices=xyz, colors=heatmap_rgb)
    
    # 导出到文件
    point_cloud.export(save_path)
    
    print(f"\n可视化成功！")
    print(f"已保存热力图点云到: {save_path}")
    print("请用 MeshLab 打开此文件查看。")

except FileNotFoundError:
    print(f"错误: 找不到文件 {npz_file_path}")
except KeyError as e:
    print(f"错误: {e}")
except Exception as e:
    print(f"发生未知错误: {e}")