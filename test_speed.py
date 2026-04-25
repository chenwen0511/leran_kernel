import torch
import custom_matmul # 你的 Shared Memory 版
import time

M, K, N = 4096, 4096, 4096
A = torch.randn(M, K, device='cuda', dtype=torch.float32).contiguous()
B = torch.randn(K, N, device='cuda', dtype=torch.float32).contiguous()

# 预热
_ = custom_matmul.forward(A, B)

# 测试你的 Shared Memory 版
torch.cuda.synchronize()
start = time.time()
C_custom = custom_matmul.forward(A, B)
torch.cuda.synchronize()
print(f"👉 你的 Shared Memory 版耗时: {time.time() - start:.4f} 秒")

# 测试 PyTorch (cuBLAS) 版
torch.cuda.synchronize()
start = time.time()
C_pt = torch.matmul(A, B)
torch.cuda.synchronize()
print(f"👉 PyTorch 官方 (cuBLAS) 耗时: {time.time() - start:.4f} 秒")
