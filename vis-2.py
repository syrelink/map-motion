import numpy as np
import trimesh
import os

# --- 1. 定义文件路径 ---
npz_file_path = 'data/HUMANISE/contact_motion/contacts/00010.npz'
save_dir = './visualizations'
# 我们给它一个新名字，以便区分
save_path = os.path.join(save_dir, '00010_SCENE_ONLY.ply') 

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# --- 2. 加载数据 ---
try:
    data = np.load(npz_file_path)
    print(f"成功加载文件: {npz_file_path}")
    print("文件中包含的键 (Keys):", data.files)

    # --- 3. 提取 XYZ 坐标 ---
    
    # 'points' 维度为 (8192, 6)，我们只取前 3 列 (XYZ)
    xyz = data['points'][:, :3]
    
    print(f"加载的 xyz 坐标维度: {xyz.shape}")
    
    # [可选的诊断] 检查坐标范围
    print(f"xyz 坐标 X 范围: {xyz[:, 0].min()} to {xyz[:, 0].max()}")
    print(f"xyz 坐标 Y 范围: {xyz[:, 1].min()} to {xyz[:, 1].max()}")
    print(f"xyz 坐标 Z 范围: {xyz[:, 2].min()} to {xyz[:, 2].max()}")

    # --- 4. 创建并导出 Trimesh 点云 (仅坐标) ---
    
    # 创建点云对象，注意：这次我们 "不" 传递 colors 参数
    point_cloud = trimesh.PointCloud(vertices=xyz)
    
    # 导出到文件
    point_cloud.export(save_path)
    
    print(f"\n可视化成功！")
    print(f"已保存 "仅场景" 点云到: {save_path}")
    print("请用 MeshLab 打开此文件，检查场景是否看起来正常。")

except FileNotFoundError:
    print(f"错误: 找不到文件 {npz_file_path}")
except KeyError as e:
    print(f"错误: 文件中找不到键 {e}。")
except Exception as e:
    print(f"发生错误: {e}")