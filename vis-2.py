import numpy as np
import cv2
import trimesh
import os

# --- 1. 定义文件路径 ---
npz_file_path = 'data/HUMANISE/contact_motion/contacts/00010.npz'
save_dir = './visualizations'
save_path = os.path.join(save_dir, '00010_affordance_map.ply')

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# --- 2. 加载数据 ---
try:
    data = np.load(npz_file_path)
    print(f"成功加载文件: {npz_file_path}")
    print("文件中包含的键 (Keys):", data.files)

    # --- 3. 根据您的文件结构提取数据 ---
    
    # 'points' 维度为 (8192, 6)，我们只取前 3 列作为 XYZ 坐标
    xyz = data['points'][:, :3]
    
    # 'dist' 维度为 (8192, 22)，代表每个点到 22 个关节的距离
    # 我们取每个点到所有关节的 "最小" 距离，将其聚合为 (8192,)
    dist_map = data['dist']
    _map = np.min(dist_map, axis=1) # 关键聚合操作！
    
    # 检查值范围，Silverster98 的代码假设值在 [0, 1] 区间
    print(f"Affordance 值的范围: Min={_map.min()}, Max={_map.max()}")
    
    # [可选] 如果你的 _map 值不在 [0, 1] 区间，可能需要手动归一化
    # if _map.max() > 1.0 or _map.min() < 0.0:
    #     print("注意：值不在 [0, 1] 范围，正在执行归一化...")
    #     _map = (_map - _map.min()) / (_map.max() - _map.min())
        
    # --- 4. 可视化 (来自 Silverster98 的代码) ---
    
    # 归一化到 [0, 255] 并转换为 uint8
    _map_uint8 = np.uint8(255 * _map)
    
    # 应用 PARULA 颜色映射
    # _map_uint8 必须是 (N,) 或 (N, 1) 的形状
    heatmap = cv2.applyColorMap(_map_uint8.reshape(-1, 1), cv2.COLORMAP_PARULA)
    
    # cv2.applyColorMap 的输出是 BGR 格式 (N, 1, 3)
    # 转换为 RGB (Trimesh 需要) 并且形状为 (N, 3)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).reshape(-1, 3)

    # --- 5. 创建并导出 Trimesh 点云 ---
    
    # 使用 xyz 坐标和 heatmap 颜色创建点云对象
    point_cloud = trimesh.PointCloud(vertices=xyz, colors=heatmap_rgb)
    
    # 导出到文件
    point_cloud.export(save_path)
    
    print(f"\n可视化成功！")
    print(f"已保存到: {save_path}")
    print("您现在可以使用 3D 查看器 (如 MeshLab) 打开该文件。")

except FileNotFoundError:
    print(f"错误: 找不到文件 {npz_file_path}")
except KeyError as e:
    print(f"错误: 文件中找不到键 {e}。请确保 .npz 文件包含 'points' 和 'dist'。")
except Exception as e:
    print(f"发生错误: {e}")