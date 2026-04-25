#include <torch/extension.h>

// 1. 核心算子：二维矩阵乘法
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // blockIdx.y / threadIdx.y 控制行 (Row)
    // blockIdx.x / threadIdx.x 控制列 (Col)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查：确保没有超出矩阵的实际尺寸
    if (row < M && col < N) {
        float value = 0.0f;
        // 点积运算：A 的第 row 行 乘以 B 的第 col 列
        for (int k = 0; k < K; ++k) {
            // 注意二维数组在内存中是一维展开的：index = row * width + col
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

// 2. PyTorch 接口函数
torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B) {
    // 安全检查：确保张量在 GPU 上且内存在物理上是连续的
    TORCH_CHECK(A.device().is_cuda(), "A 必须是 CUDA Tensor");
    TORCH_CHECK(B.device().is_cuda(), "B 必须是 CUDA Tensor");
    TORCH_CHECK(A.is_contiguous(), "A 必须在内存中连续");
    TORCH_CHECK(B.is_contiguous(), "B 必须在内存中连续");

    // 获取矩阵维度 M, K, N
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(A.size(1) == B.size(0), "矩阵维度不匹配：A 的列数必须等于 B 的行数");

    // 初始化输出矩阵 C，形状为 M x N
    auto C = torch::empty({M, N}, A.options());

    // 3. 配置二维线程模型
    // 定义每个 Block 为 16x16=256 个线程 (这是显卡很喜欢的配置)
    dim3 threadsPerBlock(16, 16);
    
    // 计算需要多少个 Block 才能覆盖整个 M x N 的矩阵
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x, // X轴覆盖 N (列)
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y  // Y轴覆盖 M (行)
    );

    // 4. 启动算子
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

// 5. 注册到 Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_forward, "Naive Matrix Multiplication (CUDA)");
}
