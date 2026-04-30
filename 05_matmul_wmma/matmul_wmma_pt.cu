#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h> // 核心 WMMA 头文件

using namespace nvcuda;

// WMMA 要求的特定块大小 (在 4090 上，16x16x16 非常通用且高性能)
// M_TILE x N_TILE x K_TILE
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void matmul_wmma_kernel(const half* A, const half* B, float* C, int M, int N, int K) {
    // 1. 依然使用 Tiling 策略，但分块大小必须配合 WMMA Tile
    // 注意：这里的 BLOCK 大小对应一个 Warp 的协作范围，通常我们一个 Block 设为 128 或 256 个线程。
    // 我们让一个 Block 负责更大的区域，让 Block 内部的 Warp 分头算不同的 16x16 块。
    // 这里简单设为一个 Block 一个 Warp (32个线程) 来教学演示。
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / WMMA_M;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / WMMA_N;

    // 2. 声明 WMMA 片段 (Fragments)
    // 它们是不透明类型，数据在 Warp 间分布。
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // 累加器片段初始化为 0
    wmma::fill_fragment(acc_frag, 0.0f);

    // 3. 沿着 K 维度滑动分块
    int numTiles = K / WMMA_K;
    for (int t = 0; t < numTiles; ++t) {
        // 计算全局内存指针
        int aRow = warpM * WMMA_M;
        int aCol = t * WMMA_K;
        int bRow = t * WMMA_K;
        int bCol = warpN * WMMA_N;

        // 4. Warp 协作从全局内存同步加载数据到片段
        // wmma::load_matrix_sync 非常方便，省去了自定义 Shared Memory 搬运和同步
        // 参数 K 和 N 是全局矩阵的宽度，用于计算 Strides。
        if (aRow < M && aCol < K) {
            wmma::load_matrix_sync(a_frag, A + (aRow * K + aCol), K);
        }
        if (bRow < K && bCol < N) {
            wmma::load_matrix_sync(b_frag, B + (bRow * N + bCol), N);
        }

        // 5. 束矩阵乘加 (WMMA) 的魔法核心
        // 执行 D = A * B + D (acc_frag)
        // 这一步 32 个线程瞬间协作，利用专用硬件算出一个 16x16x16 的块。
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // 6. 将最终的累加片段同步存储回全局内存
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + (cRow * N + cCol), acc_frag, N, wmma::mem_row_major);
    }
}

// PyTorch 绑定 (注意精度变化)
torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(A.scalar_type() == torch::ScalarType::Half, "A must be Float16 (half)");
    TORCH_CHECK(B.scalar_type() == torch::ScalarType::Half, "B must be Float16 (half)");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(A.size(1) == B.size(0), "Dimensions mismatch");

    // 输出是 FP32，累加更高精
    auto C = torch::empty({M, N}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat));

    // 配置线程模型。这里简单让一个块对应一个 Warp 负责一个 16x16 块的完整生命周期。
    // 在真实生产环境，我们会使用 256 或 128 个线程的 Block，并让 Warp 分工算更大的 Tile。
    dim3 threadsPerBlock(WMMA_N, WMMA_M); // 这里设为 16x16=256个线程，方便边界计算，内部只有一个主 Warp 计算，其余空闲演示。真实的 WMMA 只需要 Warp Size(32) 个线程。
    dim3 blocksPerGrid(
        (N + WMMA_N - 1) / WMMA_N,
        (M + WMMA_M - 1) / WMMA_M
    );

    matmul_wmma_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_forward, "Tensor Cores Matrix Multiplication (WMMA)");
}
