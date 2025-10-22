import torch
# 确保从您存放新模块的文件中正确导入
# 假设您的文件名为 rwkv_modules.py
from models.vrwkv import Block_time, RUN_CUDA

def test_block_dimension():
    """
    一个简单的单元测试函数，用于验证 Block_time 的输入输出维度。
    """
    # 检查 CUDA 内核是否成功加载
    if RUN_CUDA is None:
        print("无法执行测试：Bi-WKV CUDA kernel 未能加载。")
        return

    # --- 1. 定义模拟的超参数 ---
    batch_size = 4      # 批大小
    seq_len = 60        # 序列长度 (例如60帧动作)
    latent_dim = 512    # 特征维度 (必须和您的主模型 CMDM.latent_dim 一致)

    # --- 2. 实例化我们的 Block_time 模块 ---
    # 这里的 n_layer 和 layer_id 仅用于权重初始化，可以设置任意合理值
    print("正在实例化 Block_time 模块...")
    try:
        rwkv_block = Block_time(
            n_embd=latent_dim,
            n_layer=12,  # 假设总共有12层
            layer_id=5,  # 当前是第5层
        ).cuda() # 将模型移动到 GPU
        rwkv_block.eval() # 设置为评估模式
        print("模块实例化成功！")
    except Exception as e:
        print(f"模块实例化失败: {e}")
        return

    # --- 3. 创建一个模拟的输入张量 ---
    # 这模拟了您的动作序列在进入 Block 之前的状态
    dummy_input = torch.randn(batch_size, seq_len, latent_dim).cuda()
    print(f"\n创建了一个模拟输入张量:")
    print(f"  - 输入形状: {dummy_input.shape}")
    print(f"  - 预期输出形状: {dummy_input.shape}")

    # --- 4. 执行前向传播并检查输出 ---
    print("\n正在执行前向传播...")
    try:
        with torch.no_grad(): # 在测试时不需要计算梯度
            output = rwkv_block(dummy_input)
        
        print("前向传播完成！")
        print(f"  - 实际输出形状: {output.shape}")

        # --- 5. 验证维度是否符合预期 ---
        assert output.shape == dummy_input.shape, "输出形状与输入形状不匹配！"
        
        print("\n✅ 测试通过！输入和输出的维度完全符合预期。")

    except Exception as e:
        print(f"\n❌ 测试失败！在前向传播过程中发生错误: {e}")

# --- 运行测试 ---
if __name__ == "__main__":
    test_block_dimension()