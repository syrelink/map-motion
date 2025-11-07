import os
import glob
import numpy as np
import cv2
import trimesh
from tqdm import tqdm
from typing import Optional, List

def visualize_affordance_heatmap_on_pointcloud(
    xyz_data: np.ndarray,
    affordance_value: np.ndarray,
    save_path: str,
    map_range: Optional[Tuple[float, float]] = None
) -> None:
    """
    将Affordance值转换为热力图颜色，并应用于点云进行可视化导出。

    Args:
        xyz_data: 场景点的 (N, 3) 坐标数组。
        affordance_value: 每个场景点的 Affordance 值 (N,) 或 (N, 1) 数组。
        save_path: 导出的3D点云文件路径（例如：'output/heatmap.ply'）。
        map_range: Affordance 值的归一化范围 [min_val, max_val]。
                   如果为 None，则使用 Affordance 值的 [min, max] 进行归一化。
    """
    # 确保 Affordance 值是 (N, ) 形状
    if affordance_value.ndim > 1:
        affordance_value = affordance_value.squeeze()
        
    if affordance_value.ndim != 1 or affordance_value.shape[0] != xyz_data.shape[0]:
        raise ValueError(
            f"Affordance shape mismatch: expected ({xyz_data.shape[0]},), got {affordance_value.shape}"
        )

    # 1. 归一化 Affordance 值到 [0, 1]
    if map_range is None:
        min_val = affordance_value.min()
        max_val = affordance_value.max()
    else:
        min_val, max_val = map_range
        
    # 防止分母为零
    if max_val - min_val < 1e-6:
        normalized_map = np.zeros_like(affordance_value, dtype=np.float32)
    else:
        normalized_map = (affordance_value - min_val) / (max_val - min_val)
    
    # 将 [0, 1] 的浮点数归一化到 [0, 255] 的 uint8 整数
    # 注意：这里使用了对话中提到的 '255 * _map'
    _map = np.uint8(255 * normalized_map)

    # 2. 颜色映射 (Colormapping)
    # 使用 cv2.COLORMAP_PARULA 生成热力图
    heatmap = cv2.applyColorMap(_map, cv2.COLORMAP_PARULA)

    # 3. 颜色空间转换和重塑
    # 将 BGR (OpenCV默认) 转换为 RGB，并调整形状为 (N, 3)
    # heatmap 原始形状是 (N, 1, 3)，需要 reshape(-1, 3)
    # cv2.COLOR_BGR2RGB: 将 BGR 顺序转为 RGB 顺序
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).reshape(-1, 3)

    # 4. 导出 3D 点云
    # trimesh.PointCloud 期望颜色是 [0, 255] 的 RGB 整数
    pc = trimesh.PointCloud(vertices=xyz_data, colors=heatmap_rgb)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    pc.export(save_path)
    print(f"✅ 3D 热力图点云已成功保存到: {save_path}")


def process_affordance_files(
    affordance_folder: str, 
    xyz_template_path: str,
    output_folder: str
):
    """
    遍历 Affordance 文件，生成对应的 3D 热力图点云。

    Args:
        affordance_folder: 包含 Affordance NPY 文件的目录。
        xyz_template_path: 包含 xyz 坐标的 NPZ 文件路径模板（需替换文件名）。
        output_folder: 保存 PLY 点云文件的目标目录。
    """
    affordance_files = glob.glob(os.path.join(affordance_folder, "*.npy"))
    if not affordance_files:
        print(f"❌ 找不到 Affordance 文件: {affordance_folder}")
        return

    # 通常 Affordance 值是归一化的 (0, 1) 距离或概率。
    # 如果不确定 Affordance 值的实际 min/max 范围，可以设置一个固定的范围进行可视化对比。
    # 例如：Affordance Map 的值通常表示距离或接触概率，可能在 [0, 1] 或 [0, 某最大距离]
    # 假设 Affordance NPY 文件中保存的值是**距离**，且我们希望颜色映射范围是 [0, 0.5]
    # map_range = [0.0, 0.5] 

    print(f"找到 {len(affordance_files)} 个 Affordance 文件，开始处理...")
    
    for file_path in tqdm(affordance_files):
        # 1. 提取文件名和对应的数据索引
        base_name = os.path.basename(file_path).replace('.npy', '')
        
        # 假设文件名格式是 {data_id}-{caption_idx}.npy (例如: M001-0.npy)
        data_id = base_name.split('-')[0]
        
        # 2. 加载 Affordance Map
        # 假设 Affordance NPY 文件只包含一个 (N, 1) 或 (N,) 的 Affordance 值数组
        affordance_value = np.load(file_path) 
        
        # Affordance Map 可能包含多个样本 (K, N, 1)，如果是，我们只取第一个
        if affordance_value.ndim == 3 and affordance_value.shape[0] > 1:
            affordance_value = affordance_value[0]
            
        # 3. 加载对应的 XYZ 坐标
        # 根据您提供的 ContactMotionHumanML3DDataset 代码，xyz 坐标来自原始的 .npz 文件
        # 原始 contact 文件名应该是 {data_id}.npz
        contact_npz_path = os.path.join(
            os.path.dirname(os.path.dirname(affordance_folder)), 
            'contacts', 
            f'{data_id}.npz'
        )
        
        if not os.path.exists(contact_npz_path):
            print(f"⚠️ 找不到对应的 XYZ 文件: {contact_npz_path}，跳过。")
            continue
            
        contact_data = np.load(contact_npz_path)
        # xyz = contact_data['points'][:, 0:3]
        # 注意：这里需要确保Affordance Map的帧数N与XYZ点的数量匹配。
        # 由于我们只关心Affordance Map，我们假设这里的XYZ是针对整个点云/身体网格的点的XYZ。
        # 在 HumanML3D 中，通常只有少量关节的 XYZ，而 Contact Map 是关于这些关节到场景点的距离。
        # 假设 Affordance Map 的每一行对应一个关节或预定义的点。
        
        # 在原始 HumanML3D Contact 任务中，'points' 存储的是接触点（可能是身体或场景点）
        # 假设我们可视化的是**身体点（关节或顶点）**。
        # HumanML3D 的 'new_joint_vecs' 是 263 维，可能对应 22 关节 * 3 坐标 * 4 (root/velocity/rotation等)
        # 鉴于 ContactMap 的可视化目标是 **Affordance on SCENE POINTS**，
        # 我们假设 Affordance 的 N 对应场景点云的 N。
        
        # ***由于我们无法确定 XYZ 的具体来源和 Affordance Map 的维度对应关系，
        # 我们暂时使用 Affordance Map 的维度作为 N，假设 XYZ 对应 Affordance Map 作用的点。***
        # 如果 Affordance 维度是 (N_frames, N_points, D), 我们需要 N_points 的 XYZ 坐标。
        
        # 简化的处理（假设 Affordance 是针对每一帧的，我们取第一帧的 Affordance 均值进行可视化）
        
        # 假设 Affordance 是 (N_points, 1) 或 (N_points) 的数组
        # 并且假设我们已经知道 XYZ 坐标，这里使用一个占位符。
        # **您需要根据实际的 XYZ 数据加载方式进行修改！**
        
        # --- 临时 XYZ 坐标占位符 (假设 Affordance 值对应 Body Joints 的 Affordance) ---
        # 如果 Affordance 对应人体关节，则 N=22 或 N=24
        # 这里的 contact_data['points'] 可能是 N_points x 3
        # Affordance 值来自 pred_contact，它预测的是 Contact Map，通常是 (N_frames, N_joints) 或 (N_frames, N_joints, N_points)
        
        # 为了运行代码，我们假设 Affordance NPY 文件是 (N_points,)，且 contact_data['points'] 提供了 (N_points, 3) 的 XYZ。
        
        try:
            # 简化：使用 contact.npz 中的 'points' 作为 XYZ (场景点或身体点)
            xyz_points = contact_data['points'].astype(np.float32)
        except KeyError:
             print(f"⚠️ {contact_npz_path} 中不包含 'points' 键，无法获取 XYZ 坐标。跳过。")
             continue
        
        # 4. 生成热力图并导出
        output_file_path = os.path.join(output_folder, f"{base_name}_heatmap.ply")
        
        # 这里需要解决维度问题：Affordance Map 的帧数与 xyz 的点数可能不匹配。
        # Affordance (M, N)，XYZ (K, 3)。我们需要将 M x N 的 Affordance 映射到 K 个点上。
        # 简化处理：假设 Affordance 是针对 K 个点的单个值 (K,) 或 (K, 1)。
        
        if affordance_value.ndim == 2 and affordance_value.shape[0] != xyz_points.shape[0]:
             # 如果 Affordance 是 (N_frames, N_points)，取所有帧的均值作为最终热力图值
             print(f"⚠️ Affordance shape is {affordance_value.shape}, using mean over frames.")
             affordance_for_vis = np.mean(affordance_value, axis=0)
        elif affordance_value.shape[0] == xyz_points.shape[0]:
             # Affordance 是 (N_points,) 或 (N_points, 1)，正好匹配点数
             affordance_for_vis = affordance_value
        else:
             print(f"❌ Affordance shape {affordance_value.shape} 与 XYZ 点数 {xyz_points.shape[0]} 不匹配。跳过。")
             continue

        visualize_affordance_heatmap_on_pointcloud(
            xyz_data=xyz_points,
            affordance_value=affordance_for_vis,
            save_path=output_file_path,
            # map_range=map_range # 如果需要固定范围，请取消注释
        )

# --- 运行示例 ---
# 假设配置和路径
DATA_DIR = './' 
# Affordance 文件路径 (与 ContactMotionHumanML3DDataset 的 test 阶段设置一致)
PRED_CONTACT_FOLDER = os.path.join(DATA_DIR, 'outputs/2025-10-31_16-16-04_CDM-Perceiver-H3D/eval/test-1102-204431/H3D/pred_contact')
OUTPUT_VIS_FOLDER = './Visualization' # 最终保存 PLY 文件的目录

# 启动处理
process_affordance_files(
    affordance_folder=PRED_CONTACT_FOLDER, 
    xyz_template_path=os.path.join(DATA_DIR, 'data/H3D/contacts/{data_id}.npz'),
    output_folder=OUTPUT_VIS_FOLDER
)

print("\n--- 示例运行提示 ---")
print("要实际运行，请取消注释最后三行，并确保 `DATA_DIR` 路径正确。")
print("请务必检查您的 Affordance NPY 文件（pred_contact）和 Contact NPZ 文件（H3D/contacts）中的数据维度，以确保它们能够正确匹配。")