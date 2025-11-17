import numpy as np
import cv2  # 需要 opencv-python
import trimesh  # 需要 trimesh
import os

# --- 1. (请修改) 定义您的文件路径 ---

# 场景文件: 包含 'points' 数据的 .npz 文件
# (这个文件提供了 xyz 坐标)
scene_npz_file = 'data/HUMANISE/contact_motion/contacts/00019.npz'

# 预测文件: ADM 模型输出的 .npy 文件
# (这个文件提供了 (1, 8192, 6) 的 affordance 预测)
pred_npy_file = 'outputs/2025-10-31_16-16-04_CDM-Perceiver-H3D/eval/test-1102-204431/H3D/pred_contact/000019-0.npy'

# 输出文件: 您希望保存的彩色 .ply 文件名
save_path = 'heatmap_visualization_000019.ply'


# --- 2. 脚本主程序 ---

def visualize_affordance(scene_path, pred_path, output_path):
    print("--- 开始生成热力图 ---")

    # --- 步骤 A: 加载输入数据 (xyz) ---
    try:
        print(f"加载场景 (xyz): {scene_path}")
        scene_data = np.load(scene_path)

        # 检查 'points' 键是否存在
        if 'points' not in scene_data:
            print(f"错误: 在 {scene_path} 中找不到 'points' 键。")
            print(f"找到的键: {scene_data.files}")
            return

        xyz = scene_data['points']
        scene_data.close()

        if xyz.shape[1] != 3:
            print(f"错误: 'points' 数据的维度 {xyz.shape} 不正确，应为 (N, 3)")
            return

        print(f"  -> 'xyz' 维度: {xyz.shape}")

    except FileNotFoundError:
        print(f"错误: 找不到场景文件: {scene_path}")
        return
    except Exception as e:
        print(f"加载场景时出错: {e}")
        return

    # --- 步骤 B: 加载并处理预测数据 (_map) ---
    try:
        print(f"加载预测数据: {pred_path}")
        pred_logits = np.load(pred_path)  # 形状 (1, 8192, 6)

        print(f"  -> 原始预测维度: {pred_logits.shape}")

        # 检查维度是否匹配
        if pred_logits.ndim != 3 or pred_logits.shape[0] != 1 or pred_logits.shape[1] != xyz.shape[0]:
            print("错误: 预测维度与 'xyz' 维度不匹配。")
            print(f"预期 'xyz' 点数 {xyz.shape[0]}，但预测文件是 {pred_logits.shape}")
            return

        # 1. 去掉 Batch 维度: (1, 8192, 6) -> (8192, 6)
        pred_logits_squeezed = pred_logits.squeeze(0)

        # 2. 将 Logits (原始分数) 转换为 [0, 1] 的概率
        # 使用 Sigmoid 函数: 1 / (1 + exp(-x))
        pred_probs = 1 / (1 + np.exp(-pred_logits_squeezed))

        # 3. 创建 _map: 从 6 个关节通道中获取最大概率
        # (8192, 6) -> (8192,)
        # 这代表了“在这一点上，与‘任何’关节接触的最高概率”
        _map = pred_probs.max(axis=1)

        print(f"  -> 处理后 _map 维度: {_map.shape}")

    except FileNotFoundError:
        print(f"错误: 找不到预测文件: {pred_path}")
        return
    except Exception as e:
        print(f"加载或处理预测数据时出错: {e}")
        return

    # --- 步骤 C: 运行您的可视化代码 ---

    print("应用色彩映射 (colormap)...")

    # _map = np.uint8(255 * _map)
    # 将 [0, 1] 浮点数转为 [0, 255] 整数
    _map_uint8 = np.uint8(255 * _map)

    # heatmap = cv2.applyColorMap(_map, cv2.COLORMAP_PARULA)
    # 将灰度图应用 PARULA 色彩方案，输出是 BGR 格式
    heatmap_bgr = cv2.applyColorMap(_map_uint8, cv2.COLORMAP_PARULA)

    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR).reshape(-1, 3)
    # *** (重要修正) ***
    # applyColorMap 输出是 BGR, trimesh 需要 RGB。
    # 我们使用 BGR2RGB 将其转换为正确的颜色顺序
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3)

    # trimesh.PointCloud(vertices=xyz, colors=heatmap).export(save_path)
    # 创建带颜色的点云对象
    point_cloud = trimesh.PointCloud(vertices=xyz, colors=heatmap_rgb)

    # 导出到文件
    print(f"导出点云到: {output_path}")
    point_cloud.export(output_path)

    print("--- 成功！热力图已生成。 ---")


# --- 3. 运行主程序 ---
if __name__ == "__main__":
    visualize_affordance(scene_npz_file, pred_npy_file, save_path)