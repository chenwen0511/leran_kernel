import torch
import custom_matmul
import time

M, K, N = 4096, 4096, 4096

# 关键：改为 Half 精度输入
print(f"正在创建 Float16 矩阵: A({M}x{K}) 和 B({K}x{N})...")
A = torch.randn(M, K, device='cuda', dtype=torch.float16).contiguous()
B = torch.randn(K, N, device='cuda', dtype=torch.float16).contiguous()

# 预热
_ = custom_matmul.forward(A, B)

# 测试你的 Tensor Cores (WMMA) 版
torch.cuda.synchronize()
start = time.time()
C_custom = custom_matmul.forward(A, B)
torch.cuda.synchronize()
print(f"👉 你的 Tensor Cores (WMMA) 版耗时: {time.time() - start:.4f} 秒")

# 测试 PyTorch (cuBLAS) 版 (自动利用 Tensor Cores)
torch.cuda.synchronize()
start = time.time()
C_pt = torch.matmul(A, B).to(torch.float32) # cuBLAS FP16乘完，我们转 FP32 比较结果
torch.cuda.synchronize()
print(f"👉 PyTorch 官方 (cuBLAS) 耗时: {time.time() - start:.4f} 秒")
