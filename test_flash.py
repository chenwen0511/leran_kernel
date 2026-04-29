import torch
import time

# 模拟一个 8K 长度，隐式维度为 4096 的注意力机制参数
# Batch=1, Heads=32, Seq_len=8192, Head_dim=128
batch, heads, seq_len, head_dim = 1, 32, 8192, 128

q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
v = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)

# ==========================================
# 1. 传统朴素实现 (会产生巨大中间矩阵)
# ==========================================
def standard_attention(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    probs = torch.nn.functional.softmax(scores, dim=-1)
    output = torch.matmul(probs, v)
    return output

torch.cuda.synchronize()
start = time.time()
out_std = standard_attention(q, k, v)
torch.cuda.synchronize()
print(f"❌ 标准 Attention 耗时: {(time.time() - start)*1000:.2f} ms")

# ==========================================
# 2. FlashAttention (通过 F.scaled_dot_product_attention 触发)
# ==========================================
torch.cuda.synchronize()
start = time.time()
# PyTorch 2.0+ 内置了 FlashAttention
out_flash = torch.nn.functional.scaled_dot_product_attention(q, k, v)
torch.cuda.synchronize()
print(f"🚀 FlashAttention 耗时: {(time.time() - start)*1000:.2f} ms")

print(f"结果是否一致? {torch.allclose(out_std, out_flash, atol=1e-2)}")
