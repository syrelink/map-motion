import numpy as np
import trimesh
import cv2
import os
import glob


def auto_match_and_visualize(pred_dir, gt_dir, output_dir="./map-vis-matched"):
    # 1. 准备输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📂 创建输出目录: {output_dir}")

    # 2. 获取所有预测文件 (.npy)
    # 假设文件名格式是 "02000.npy"
    pred_files = glob.glob(os.path.join(pred_dir, "*.npy"))

    if len(pred_files) == 0:
        print("❌ 未找到任何 .npy 文件，请检查 pred_dir 路径！")
        return

    print(f"🔍 找到 {len(pred_files)} 个预测文件，开始自动匹配...")

    success_count = 0

    for pred_path in pred_files:
        try:
            # --- A. 提取核心 ID ---
            # 从 "path/to/02000.npy" 提取 "02000"
            file_name = os.path.basename(pred_path)  # "02000.npy"
            file_id = os.path.splitext(file_name)[0]  # "02000"

            # --- B. 构造对应的 GT 路径 ---
            # 只要 ID 一样，它们就是一对！
            gt_path = os.path.join(gt_dir, f"{file_id}.npz")

            # 检查 GT 是否存在
            if not os.path.exists(gt_path):
                print(f"⚠️  跳过: 找到了预测 {file_id}.npy，但在 data 文件夹没找到对应的 {file_id}.npz")
                continue

            # --- C. 开始“移花接木” ---
            # 1. 读取预测 (取热力值 - 后3列)
            pred_data = np.load(pred_path)  # (1, 8192, 6)
            pred_colors = pred_data[0, :, 3:]

            # 2. 读取 GT (取几何 XYZ - 前3列)
            gt_data = np.load(gt_path)
            # 兼容不同的 key 写法
            if 'points' in gt_data:
                real_xyz = gt_data['points'][:, :3]
            elif 'scene' in gt_data:
                real_xyz = gt_data['scene'][:, :3]
            else:
                real_xyz = gt_data['arr_0'][:, :3]

            # --- D. 生成热力图 ---
            # 计算强度并归一化
            intensity = np.mean(pred_colors, axis=1)
            # 动态归一化以增强每张图的对比度
            norm_intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
            norm_uint8 = np.uint8(255 * norm_intensity)

            # 颜色映射
            try:
                colormap = cv2.COLORMAP_PARULA
            except AttributeError:
                colormap = cv2.COLORMAP_JET

            heatmap = cv2.applyColorMap(norm_uint8, colormap).squeeze()
            heatmap_rgb = cv2.cvtColor(heatmap.reshape(1, -1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)

            # --- E. 保存 ---
            save_name = f"vis_{file_id}.ply"
            save_path = os.path.join(output_dir, save_name)

            trimesh.PointCloud(vertices=real_xyz, colors=heatmap_rgb).export(save_path)

            print(f"✅ 已生成: {save_name} (匹配 ID: {file_id})")
            success_count += 1

        except Exception as e:
            print(f"❌ 处理 {file_id} 时出错: {e}")

    print(f"\n🎉 处理完成！成功匹配并生成了 {success_count} 个文件。")
    print(f"结果保存在: {output_dir}")


# --- 🚀 运行配置 ---
if __name__ == "__main__":
    # 1. 预测文件文件夹 (包含 02000.npy 等)
    pred_folder = "outputs/CDM-Perceiver-HUMANISE-step200k/eval/test-1114-125307/HUMANISE/pred_contact/"

    # 2. 真实数据文件夹 (包含 02000.npz 等)
    # 请务必确认这个路径下有大量 .npz 文件
    gt_folder = "data/HUMANISE/contact_motion/contacts/"

    auto_match_and_visualize(pred_folder, gt_folder)