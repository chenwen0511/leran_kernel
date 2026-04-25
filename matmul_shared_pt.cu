#include <torch/extension.h>

// 定义分块大小 (Tile Size)，16x16=256个线程，非常契合硬件
#define TILE_SIZE 16

__global__ void matmul_shared_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // 1. 声明共享内存 (Shared Memory)
    // __shared__ 关键字意味着这块内存存在于 SM 的 L1 Cache 中，同一个 Block 内的所有线程共享它！
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    // 计算当前线程负责的 C 矩阵的全局行号和列号
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // 寄存器变量，用于累加当前线程的最终结果
    float value = 0.0f;

    // 2. 沿着 K 维度滑动分块 (Tiling)
    // 每次滑动 TILE_SIZE 这么多的距离
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        
        // 3. 团队协作搬砖：每个线程负责把 A 和 B 的一个元素从全局显存读到共享内存
        // 记得做边界检查，如果越界就补 0 (Padding)
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            sA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (t * TILE_SIZE + threadIdx.y < K && col < N) {
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 4. 屏障同步 (Barrier Synchronization)
        // 必须等 Block 里所有线程都把自己的那块砖搬完，才能开始下一步计算
        __syncthreads();

        // 5. 在极速的共享内存中计算这一小块的点积
        for (int i = 0; i < TILE_SIZE; ++i) {
            value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        // 6. 再次同步！
        // 必须等所有人算完，才能进入下一个 for 循环去覆盖 sA 和 sB 的内容
        __syncthreads();
    }

    // 7. 所有分块滑动完毕，将最终累加结果写回慢速的全局显存 C
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// 绑定到 PyTorch 的接口函数 (逻辑和上次一模一样)
torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A 必须是 CUDA Tensor");
    TORCH_CHECK(B.device().is_cuda(), "B 必须是 CUDA Tensor");
    TORCH_CHECK(A.is_contiguous(), "A 必须连续");
    TORCH_CHECK(B.is_contiguous(), "B 必须连续");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(A.size(1) == B.size(0), "维度不匹配");

    auto C = torch::empty({M, N}, A.options());

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    matmul_shared_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_forward, "Shared Memory Matrix Multiplication");
}
