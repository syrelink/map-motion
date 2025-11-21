import numpy as np
import trimesh
import cv2
import os


def visualize_affordance(npz_path, output_filename="affordance_vis.ply"):
    print(f"正在处理文件: {npz_path}")

    # 1. 加载数据
    data = np.load(npz_path)

    # 获取场景点坐标 (XYZ)
    # shape: (8192, 6) -> 取前3列作为坐标
    points_raw = data['points']
    xyz = points_raw[:, :3]

    # 获取距离场数据
    # shape: (8192, 22)
    dists = data['dist']

    # 2. 计算 Affordance (可供性)
    # 论文公式 (1): c = exp(-0.5 * d / sigma^2)
    # 原始数据是距离(distance)，我们需要将其转换为热度(affinity)。
    # 场景中某一点的 affordance 取决于它离人体"最近"的那个关节的距离。

    # 第一步：对22个关节取最小距离，得到该点离人体的最近距离
    # Shape 变为 (8192,)
    min_dist = np.min(dists, axis=1)

    # 第二步：应用高斯核进行转换 (Distance -> Affordance)
    # sigma 控制热力图的扩散程度，论文中提到 sigma 是归一化因子
    sigma = 0.5  # 可以根据视觉效果调整这个参数
    affordance_val = np.exp(-0.5 * (min_dist ** 2) / (sigma ** 2))

    # 3. 归一化到 [0, 255]
    # 将 0.0-1.0 的浮点数映射到 0-255 的整数
    # 为了增强对比度，这里使用了基于数据最大最小值的归一化
    _map = (affordance_val - affordance_val.min()) / (affordance_val.max() - affordance_val.min() + 1e-8)
    _map = np.uint8(255 * _map)

    # 4. 应用颜色映射 (Color Mapping)
    # 注意: cv2.COLORMAP_PARULA 在标准 OpenCV 中可能不可用。
    # 如果报错，通常使用 cv2.COLORMAP_JET 或 cv2.COLORMAP_VIRIDIS 代替。
    try:
        # 尝试使用 Parula (如果你的 opencv 支持)
        colormap_mode = cv2.COLORMAP_PARULA
    except AttributeError:
        print("警告: 当前 OpenCV 版本不支持 COLORMAP_PARULA，切换为 COLORMAP_JET")
        colormap_mode = cv2.COLORMAP_JET

    # applyColorMap 需要输入维度为 (N, 1) 或 (H, W)
    heatmap = cv2.applyColorMap(_map, colormap_mode)

    # applyColorMap 输出是 BGR 格式 (8192, 1, 3)，我们需要 (8192, 3) RGB
    heatmap = heatmap.squeeze()
    heatmap_rgb = cv2.cvtColor(heatmap.reshape(1, -1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)

    # 5. 导出点云
    # 创建 Trimesh 点云对象：包含几何位置(vertices)和颜色(colors)
    pcd = trimesh.PointCloud(vertices=xyz, colors=heatmap_rgb)

    # 保存文件
    pcd.export(output_filename)
    print(f"可视化文件已保存至: {output_filename}")


# --- 运行示例 ---
# 请修改为你的实际文件路径
input_file = "data/HUMANISE/contact_motion/contacts/00002.npz"

if os.path.exists(input_file):
    visualize_affordance(input_file)
else:
    print(f"找不到文件: {input_file}，请检查路径。")

# # 如果你想可视化那个预测文件 (pred_contact/02000.npy):
# data = np.load("outputs/CDM-Perceiver-HUMANISE-step200k/eval/test-1118-194403/HUMANISE/pred_contact/02000.npy")
# xyz = data[0, :, :3]  # 取第一个 batch
# colors = data[0, :, 3:] # 假设后3维是预测的颜色
# trimesh.PointCloud(vertices=xyz, colors=colors).export("pred_vis.ply")