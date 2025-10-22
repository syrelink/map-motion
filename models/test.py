# test.py (增强版)

import torch
import sys

# --- 增强的诊断部分 ---
# 1. 尝试导入模块，如果失败则给出明确提示
try:
    print("正在尝试从 'models.vrwkv' 导入模块...")
    # 确保您的文件名是 vrwkv.py 并且在 models 文件夹下
    from models.vrwkv import Block_time, RUN_CUDA
    print("模块导入成功！")
except ImportError as e:
    print(f"\n❌ 导入失败！请检查文件路径和名称。")
    print(f"   错误信息: {e}")
    sys.exit(1) # 导入失败，直接退出

def test_block_dimension():
    # 2. 检查 CUDA 内核是否成功加载
    if RUN_CUDA is None:
        print("\n❌ 测试无法执行：Bi-WKV CUDA kernel 未能加载。")
        print("   这通常是由于 C++/CUDA 编译环境问题导致的。请检查：")
        print("   - 是否已安装 g++ 和 NVIDIA CUDA Toolkit (nvcc)。")
        print("   - 编译器路径是否在系统 PATH 中。")
        print("   - PyTorch 和 CUDA 版本是否兼容。")
        print("   - load() 函数中的源文件路径是否正确。")
        return

    # --- 定义模拟的超参数 ---
    batch_size = 4      
    seq_len = 60        
    latent_dim = 512    

    # --- 实例化我们的 Block_time 模块 ---
    print("\n正在实例化 Block_time 模块...")
    try:
        rwkv_block = Block_time(
            n_embd=latent_dim,
            n_layer=12,
            layer_id=5,
        ).cuda()
        rwkv_block.eval()
        print("模块实例化成功！")
    except Exception as e:
        print(f"\n❌ 模块实例化失败: {e}")
        return

    # --- 创建一个模拟的输入张量 ---
    dummy_input = torch.randn(batch_size, seq_len, latent_dim).cuda()
    print(f"\n创建了一个模拟输入张量:")
    print(f"  - 输入形状: {dummy_input.shape}")
    print(f"  - 预期输出形状: {dummy_input.shape}")

    # --- 执行前向传播并检查输出 ---
    print("\n正在执行前向传播...")
    try:
        with torch.no_grad():
            output = rwkv_block(dummy_input)
        
        print("前向传播完成！")
        print(f"  - 实际输出形状: {output.shape}")

        # --- 验证维度是否符合预期 ---
        assert output.shape == dummy_input.shape, "输出形状与输入形状不匹配！"
        
        print("\n✅ 测试通过！输入和输出的维度完全符合预期。")

    except Exception as e:
        print(f"\n❌ 测试失败！在前向传播过程中发生错误。这很可能是 CUDA 内核执行时的问题。")
        print(f"   错误信息: {e}")

# --- 运行测试 ---
if __name__ == "__main__":
    test_block_dimension()