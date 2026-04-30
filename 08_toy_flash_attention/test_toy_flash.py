import torch
import toy_flash_attn
import math

# 设置参数 (配合 C++ 中的宏定义 d=32)
seq_len = 2048 # 尝试 2048 长度
d = 32

print(f"初始化张量... 序列长度: {seq_len}, 维度: {d}")
# 必须使用 FP32 以匹配我们写的教学版 C++
Q = torch.randn(seq_len, d, device='cuda', dtype=torch.float32).contiguous()
K = torch.randn(seq_len, d, device='cuda', dtype=torch.float32).contiguous()
V = torch.randn(seq_len, d, device='cuda', dtype=torch.float32).contiguous()

# ==========================================
# 1. 运行 PyTorch 标准注意力 (标准答案)
# ==========================================
print("计算 PyTorch 标准 Attention...")
scores = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(d)
probs = torch.nn.functional.softmax(scores, dim=-1)
O_std = torch.matmul(probs, V)

# ==========================================
# 2. 运行我们手写的 Toy FlashAttention
# ==========================================
print("计算自定义 Toy FlashAttention...")
O_custom = toy_flash_attn.forward(Q, K, V)

# ==========================================
# 3. 结果验证
# ==========================================
# 允许有一定的浮点数累加误差
is_correct = torch.allclose(O_std, O_custom, atol=1e-3, rtol=1e-3)
print(f"\n✅ 结果是否正确? {'是' if is_correct else '否'}")

if is_correct:
    print("太棒了！你的 C++ Online Softmax 在线更新逻辑与标准全局 Softmax 结果完全一致！")
    print("这意味着你的代码真正在没有生成完整 NxN 矩阵的情况下，算出了正确的注意力机制！")
else:
    print("最大误差:", (O_std - O_custom).abs().max().item())
