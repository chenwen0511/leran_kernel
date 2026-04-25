import torch
import custom_ops # 导入你刚刚用 CUDA 写的包！

print("初始化 GPU Tensors...")
# 在 4090 上创建两个百万级别的数组
a = torch.ones(1000000, device='cuda', dtype=torch.float32)
b = torch.ones(1000000, device='cuda', dtype=torch.float32) * 2

print("调用自定义 CUDA 算子...")
# 执行你的 CUDA 内核！
c = custom_ops.forward(a, b)

print(f"C 的前 5 个元素: {c[:5]}")
print(f"结果是否全部正确 (期望值为3.0)? {torch.all(c == 3.0).item()}")
