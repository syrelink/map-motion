import pickle
import numpy as np

# 假设文件路径如下所示 (请根据您的实际路径进行修改)
file_path = "outputs/CMDM-Enc-HUMANISE-step400k/eval/test-1118-155603/joints/02000.pkl"


def analyze_pkl_file(file_path):
    """
    加载并分析模型生成的动作数据（.pkl 文件）。
    """
    try:
        # 1. 加载数据
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print(f"--- 成功加载文件: {file_path} ---")
        print(f"数据类型: {type(data)}")

        # 2. 检查数据结构（根据您提供的渲染脚本推断，它是一个字典）
        if isinstance(data, dict):
            print("文件内容结构: 字典 (Dict)")
            print("-" * 30)
            print("字典包含的键 (Keys):")

            for key, value in data.items():
                # 打印每个键的类型、形状和前几个值（如果是数组）
                key_info = f"  - '{key}': Type={type(value).__name__}"

                if isinstance(value, np.ndarray):
                    key_info += f", Shape={value.shape}, dtype={value.dtype}"
                    print(key_info)
                    if value.size > 0:
                        print(f"    Sampled values: {value.flatten()[:5]}")
                elif isinstance(value, list):
                    key_info += f", Length={len(value)}"
                    print(key_info)
                    if len(value) > 0:
                        print(f"    First item type: {type(value[0]).__name__}")
                elif isinstance(value, str):
                    key_info += f", Content (partial): '{value[:100]}...'"
                    print(key_info)
                else:
                    print(key_info)
        else:
            print("文件内容结构: 非字典类型 (例如:", type(data).__name__, ")")
            print("原始内容示例:", str(data)[:200], "...")

    except FileNotFoundError:
        print(f"错误：文件未找到，请检查路径: {file_path}")
    except Exception as e:
        print(f"加载或分析文件时发生错误: {e}")


# 执行分析
analyze_pkl_file(file_path)