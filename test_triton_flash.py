import torch
import triton
import triton.language as tl
import math
import time

# ==========================================
# 1. 核心算子：Triton 版 FlashAttention
# ==========================================
# @triton.jit 装饰器会在运行时将 Python 代码即时编译为 GPU 机器码
@triton.jit
def flash_attn_kernel(
    Q, K, V, O,              # 数据指针
    seq_len, d,              # 维度信息
    BLOCK_M: tl.constexpr,   # 块大小 (编译期常量)
    BLOCK_N: tl.constexpr, 
    BLOCK_D: tl.constexpr
):
    # 1. 确定当前 Block 负责的 Q 矩阵的行范围
    pid = tl.program_id(0)
    start_m = pid * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M) # 比如 [0, 1, 2... 15]
    offs_d = tl.arange(0, BLOCK_D)           # 比如 [0, 1... 31]
    
    # 2. 极其优雅的指针计算：广播机制自动生成 2D 矩阵指针
    # 等价于 C++ 中的 (row * d + col)
    q_ptrs = Q + (offs_m[:, None] * d + offs_d[None, :])
    mask_q = (offs_m[:, None] < seq_len) & (offs_d[None, :] < d)
    
    # 加载 Q 块到 SRAM（Triton 会自动分配 Shared Memory！）
    q = tl.load(q_ptrs, mask=mask_q, other=0.0)

    # 3. 初始化 Online Softmax 的状态变量 (自动分配在极速的寄存器中)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # 4. 沿着 Seq_len 维度滑动，遍历 K 和 V 块
    offs_n = tl.arange(0, BLOCK_N)
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n_curr = start_n + offs_n
        
        # 计算 K 和 V 的指针并加载
        k_ptrs = K + (offs_n_curr[:, None] * d + offs_d[None, :])
        v_ptrs = V + (offs_n_curr[:, None] * d + offs_d[None, :])
        mask_kv = (offs_n_curr[:, None] < seq_len) & (offs_d[None, :] < d)
        
        k = tl.load(k_ptrs, mask=mask_kv, other=0.0)
        v = tl.load(v_ptrs, mask=mask_kv, other=0.0)

        # 5. 魔法降临：Q * K^T (Triton 编译器会自动将其翻译为 Tensor Core MMA 指令！)
        qk = tl.dot(q, tl.trans(k))
        
        # 缩放并遮蔽越界部分
        qk = qk / 5.65685 # sqrt(32)
        qk = tl.where(offs_n_curr[None, :] < seq_len, qk, float("-inf"))

        # 6. Online Softmax 核心公式 (全是高阶张量操作，没有繁琐的 for 循环！)
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        
        l_new = l_i * alpha + tl.sum(p, 1)
        
        # 7. 累加计算结果 P * V
        # 将 p 转换为 V 的数据类型，防止类型不匹配
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        
        # 推进历史状态
        m_i = m_new
        l_i = l_new

    # 8. 归一化并写回极慢的 HBM
    acc = acc / l_i[:, None]
    o_ptrs = O + (offs_m[:, None] * d + offs_d[None, :])
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_q)


# ==========================================
# 2. PyTorch 调用接口
# ==========================================
def triton_flash_attention(q, k, v):
    seq_len, d = q.shape
    o = torch.empty_like(q)
    
    # 定义块大小 (硬件友好，必须是 2 的幂)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 32
    
    # 启动 Grid (告诉 Triton 需要多少个 Block 来处理行)
    grid = (triton.cdiv(seq_len, BLOCK_M), )
    
    # 发射！
    flash_attn_kernel[grid](
        q, k, v, o,
        seq_len, d,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return o

# ==========================================
# 3. 终极测试与速度对比
# ==========================================
if __name__ == "__main__":
    seq_len = 4096
    d = 32
    print(f"初始化 Float16 张量... 序列长度: {seq_len}, 维度: {d}")
    
    # Triton 在 FP16 混合精度下才能完美发挥 Tensor Cores 威力
    Q = torch.randn(seq_len, d, device='cuda', dtype=torch.float16)
    K = torch.randn(seq_len, d, device='cuda', dtype=torch.float16)
    V = torch.randn(seq_len, d, device='cuda', dtype=torch.float16)

    # PyTorch 标准算子
    torch.cuda.synchronize()
    start = time.time()
    scores = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(d)
    probs = torch.nn.functional.softmax(scores, dim=-1)
    O_std = torch.matmul(probs, V)
    torch.cuda.synchronize()
    print(f"❌ PyTorch 标准 Attention 耗时: {(time.time() - start)*1000:.2f} ms")

    # Triton 自定义算子
    torch.cuda.synchronize()
    start = time.time()
    O_triton = triton_flash_attention(Q, K, V)
    torch.cuda.synchronize()
    print(f"🚀 Triton FlashAttention 耗时: {(time.time() - start)*1000:.2f} ms")

    is_correct = torch.allclose(O_std, O_triton, atol=1e-2, rtol=1e-2)
    print(f"\n✅ 结果是否正确? {'是' if is_correct else '否'}")
