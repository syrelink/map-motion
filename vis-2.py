import numpy as np
import cv2
import trimesh
import os

# --- 1. 定义文件路径 ---
npz_file_path = 'data/HUMANISE/contact_motion/contacts/00010.npz'
save_dir = './visualizations'
save_path = os.path.join(save_dir, '00010_affordance_map_NORMALIZED.ply') # 改个新名字

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
    
    # 'dist' 维度为 (8192, 22)
    _map = np.min(data['dist'], axis=1) # 形状 (8192,)
    
    print(f"原始 Affordance 值的范围: Min={_map.min()}, Max={_map.max()}")
    
    # --- 4. 可视化 (关键修复) ---
    
    # ！！！！！！！！！！！！！！！！！！！！！！！！
    # 关键修复：归一化 (Normalization)
    # 将 [Min, Max] (例如 [0.03, 2.05]) 映射到 [0, 1] 范围
    # ！！！！！！！！！！！！！！！！！！！！！！！！
    _map_normalized = (_map - _map.min()) / (_map.max() - _map.min())
    
    # 现在 _map_normalized 的范围是 [0, 1]，可以安全地乘以 255
    _map_uint8 = np.uint8(255 * _map_normalized)
    
    # 应用 PARULA 颜色映射
    heatmap = cv2.applyColorMap(_map_uint8.reshape(-1, 1), cv2.COLORMAP_PARULA)
    
    # 转换为 RGB (Trimesh 需要)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).reshape(-1, 3)

    # --- 5. 创建并导出 Trimesh 点云 ---
    
    point_cloud = trimesh.PointCloud(vertices=xyz, colors=heatmap_rgb)
    
    # 导出到文件
    point_cloud.export(save_path)
    
    print(f"\n可视化成功！")
    print(f"已保存到: {save_path}")
    print("请检查新生成的文件，这次应该有彩色热力图了。")

except Exception as e:
    print(f"发生错误: {e}")