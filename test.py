import numpy as np
import cv2
import trimesh

# --- 1. 准备你的输入数据 (这是你需要自己提供的) ---

# 假设你的场景点云有 1000 个点
num_points = 1000

# _map: 你的可供性地图，是一个 [0.0, 1.0] 范围内的 NumPy 数组
#         这里我们用随机数来模拟。
#         根据我们之前的讨论，高值(接近1.0)代表“强交互/距离近”
_map = np.random.rand(num_points)

# xyz: 点云的 3D 坐标，形状为 (N, 3)
#      这里我们用随机数来模拟
xyz = np.random.rand(num_points, 3)

# save_path: 你想保存文件的路径
save_path = 'Visualization/colored_point_cloud.ply'


# --- 2. 运行截图中的代码 ---

# 将 [0.0, 1.0] 的值 缩放到 [0, 255] 的整数范围
_map_uint8 = np.uint8(255 * _map)

# 应用色谱，将灰度值转换为 BGR 彩色图像
# _map_uint8 的形状需要是 (N, 1) 才能被 applyColorMap 正确处理
heatmap_bgr = cv2.applyColorMap(_map_uint8.reshape(-1, 1), cv2.COLORMAP_PARULA)

# 将 BGR 格式转换为 trimesh 需要的 RGB 格式
# (注意：截图中的 cv2.COLOR_RGB2BGR 也能实现 BGR->RGB 的效果，但 cv2.COLOR_BGR2RGB 是更标准、更清晰的写法)
heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

# 将颜色数组的形状从 (N, 1, 3) 调整为 (N, 3)
heatmap_flat = heatmap_rgb.reshape(-1, 3)

# 使用 trimesh 创建点云对象，并赋给它坐标(vertices)和颜色(colors)
point_cloud = trimesh.PointCloud(vertices=xyz, colors=heatmap_flat)

# 导出为 3D 文件 (例如 .ply 格式)
point_cloud.export(save_path)

print(f"成功将彩色点云保存到: {save_path}")