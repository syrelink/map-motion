import numpy as np

# 假设文件路径如下 (请根据您的实际环境修改基础路径)
gt_contact_path = "data/HUMANISE/contact_motion/contacts/00000.npz"
pred_contact_path = "data/HUMANISE/pred_contact/00000.npy"
pred_contact_path2 = "outputs/CDM-Perceiver-HUMANISE-step200k/eval/test-1114-125307/HUMANISE/pred_contact/02000.npy"


def analyze_npz_file(file_path):
    """
    加载并分析地面真值 (GT) contact/affordance 数据 (.npz 文件)。

    预期结构: 包含 'points', 'mask', 'dist' 三个键的字典。
    """
    print(f"\n--- 分析文件: {file_path} ---")
    try:
        # npz 文件使用 np.load 加载，并需要通过键访问内部数组
        with np.load(file_path) as data:
            print(f"数据结构: NumPy NpzFile (包含多个数组)")
            print(f"包含的键 (Keys): {list(data.keys())}")

            # 预期键值分析
            for key in data.keys():
                arr = data[key]
                print(f"\n  - 数组名: '{key}'")
                print(f"    -> 维度 (Shape): {arr.shape}")
                print(f"    -> 数据类型: {arr.dtype}")

                # 基于您之前分析的推断进行解释
                if key == 'points' and arr.shape == (8192, 6):
                    print("    -> 含义推断: 3D 场景点云 (N=8192) 的 XYZRGB 坐标或特征。")
                elif key == 'mask' and arr.shape == (8192,):
                    print("    -> 含义推断: 场景点的有效性或目标对象二值掩码。")
                elif key == 'dist' and len(arr.shape) == 2 and arr.shape[0] == 8192:
                    print(f"    -> 含义推断: 地面真值 Affordance/距离场 (N=8192 场景点到 J={arr.shape[1]} 关节)。")

                if arr.size > 0:
                    print(f"    -> 前 5 个值示例: {arr.flatten()[:5]}")

    except FileNotFoundError:
        print(f"错误：文件未找到，请检查路径: {file_path}")
    except Exception as e:
        print(f"加载或分析文件时发生错误: {e}")


def analyze_npy_file(file_path):
    """
    加载并分析模型预测的 Affordance/Contact 数据 (.npy 文件)。

    预期结构: (1, N, D_C) 的 NumPy 数组。
    """
    print(f"\n--- 分析文件: {file_path} ---")
    try:
        data = np.load(file_path)

        print(f"数据类型: {type(data)}")
        print(f"数据结构 (Shape): {data.shape}")
        print(f"数据元素类型 (Dtype): {data.dtype}")

        # 基于您之前分析的推断进行解释
        if data.shape == (1, 8192, 6):
            print(f"\n含义推断:")
            print(f"  -> 这是一个 Affordance Diffusion Model (ADM) 的预测输出。")
            print(f"  -> Shape (1, 8192, 6) 代表 (Batch Size, 场景点数 N, 编码特征维度 D_C)。")
            print(f"  -> 它是一个 6 维的、编码后的场景可供性（Affordance）特征图，用于指导第二阶段运动生成。")

        if data.size > 0:
            print(f"前 5 个元素示例: {data.flatten()[:5]}")

    except FileNotFoundError:
        print(f"错误：文件未找到，请检查路径: {file_path}")
    except Exception as e:
        print(f"加载或分析文件时发生错误: {e}")


analyze_npz_file(gt_contact_path)
analyze_npy_file(pred_contact_path)
analyze_npy_file(pred_contact_path2)