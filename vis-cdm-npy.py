import numpy as np

# 假设文件路径如下所示 (请根据您的实际路径进行修改)
file_path = "outputs/CDM-Perceiver-HUMANISE-step200k/eval/test-1114-125307/HUMANISE/pred_contact/02000.npy"


def analyze_npy_file(file_path):
    """
    加载并分析模型生成的可供性图数据（.npy 文件）。
    """
    try:
        # 1. 加载数据
        data = np.load(file_path)

        print(f"--- 成功加载文件: {file_path} ---")
        print(f"数据类型: {type(data)}")
        print(f"数据结构 (Shape): {data.shape}")
        print(f"数据元素类型 (Dtype): {data.dtype}")

        # 2. 打印部分内容示例
        if data.size > 0:
            print("-" * 30)
            print("前 5 个元素（扁平化后）示例:")
            print(data.flatten()[:5])

    except FileNotFoundError:
        print(f"错误：文件未找到，请检查路径: {file_path}")
    except Exception as e:
        print(f"加载或分析文件时发生错误: {e}")


# 执行分析
analyze_npy_file(file_path)