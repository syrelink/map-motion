import numpy as np
'''
outputs/CDM-Perceiver-HUMANISE-step200k/eval/test-1114-125307/HUMANISE/pred_contact/02000.npy: (1, 8192, 6)
正在检查文件: data/HUMANISE/contact_motion/contacts/00000.npz 
包含的键 (Keys): ['points', 'mask', 'dist']

数组名: 'points'
  -> 维度 (Shape): (8192, 6)
  -> 数据类型: float32
------------------------------
数组名: 'mask'
  -> 维度 (Shape): (8192,)
  -> 数据类型: int64
------------------------------
数组名: 'dist'
  -> 维度 (Shape): (8192, 22)
  -> 数据类型: float32
------------------------------

'''
# 你要检查的文件路径
file_path3 = 'data/H3D/contacts/000000.npz'
file_path = 'data/HUMANISE/contact_motion/contacts/00000.npz'

print(f"--- 正在分析文件: {file_path} ---")

data_ary = np.load(file_path)
print(data_ary)
print(data_ary.shape)

# try:
#     # 1. 使用 np.load() 加载文件
#     # 这会返回一个 NpzFile 对象，它像一个字典
#     data = np.load(file_path)
#
#     # 2. 打印文件内所有数组的“键” (Keys)
#     # .files 属性会列出所有存储在文件中的数组名称
#     keys = data.files
#     print(f"文件包含的数组 (Keys): {keys}\n")
#
#     # 3. 遍历每一个键，打印详细信息
#     if not keys:
#         print("此 .npz 文件为空。")
#     else:
#         for key in keys:
#             # 3.1 通过键名从 data 对象中获取数组
#             array = data[key]
#
#             print(f"--- 数组: '{key}' ---")
#
#             # 3.2 打印数组的维度 (Shape)
#             print(f"  -> 维度 (Shape): {array.shape}")
#
#             # 3.3 打印数组的数据类型
#             print(f"  -> 数据类型 (DType): {array.dtype}")
#
#             # 3.4 打印数组内容的摘要 (例如前5个元素)
#             #     使用 .flatten() 将多维数组展平，以便轻松查看
#             print(f"  -> 数据内容 (前5个值):")
#             if array.size > 5:
#                 print(f"     {array.flatten()[:5]} ... (及更多数据)")
#             else:
#                 print(f"     {array}")
#             print("-" * 20)
#
#     # 4. (重要) 关闭文件句柄
#     data.close()
#
# except FileNotFoundError:
#     print(f"错误: 找不到文件！请检查路径是否正确: {file_path}")
# except Exception as e:
#     print(f"读取文件时发生错误: {e}")