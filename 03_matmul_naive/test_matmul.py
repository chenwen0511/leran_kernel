import torch
import custom_matmul
import time

# 矩阵维度设定为 4096 x 4096 (大模型中常见的维度)
M, K, N = 4096, 4096, 4096

print(f"正在创建矩阵: A({M}x{K}) 和 B({K}x{N})...")
# 使用 contiguous() 确保内存连续
A = torch.randn(M, K, device='cuda', dtype=torch.float32).contiguous()
B = torch.randn(K, N, device='cuda', dtype=torch.float32).contiguous()

# 预热 GPU (防止第一次执行包含了初始化的开销)
_ = custom_matmul.forward(A, B)
_ = torch.matmul(A, B)

# 1. 测试你的 CUDA 算子
torch.cuda.synchronize()
start_time = time.time()
C_custom = custom_matmul.forward(A, B)
torch.cuda.synchronize()
custom_time = time.time() - start_time
print(f"你的朴素 CUDA 算子耗时: {custom_time:.4f} 秒")

# 2. 测试 PyTorch 官方算子 (底层是 NVIDIA cuBLAS 库)
torch.cuda.synchronize()
start_time = time.time()
C_pytorch = torch.matmul(A, B)
torch.cuda.synchronize()
pytorch_time = time.time() - start_time
print(f"PyTorch 官方 (cuBLAS) 耗时: {pytorch_time:.4f} 秒")

# 3. 验证精度 (浮点运算会有微小误差，所以用 torch.allclose)
is_correct = torch.allclose(C_custom, C_pytorch, atol=1e-3)
print(f"结果是否正确? {'✅ 是' if is_correct else '❌ 否'}")
