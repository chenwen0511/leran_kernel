#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA 调用失败: " << #call                            \
                      << " | 错误: " << cudaGetErrorString(err__)              \
                      << " | 行号: " << __LINE__ << std::endl;                 \
            return 1;                                                          \
        }                                                                      \
    } while (0)

// 1. 定义 CUDA 算子 (Kernel)
// __global__ 表示这个函数在 GPU 上运行，可以被 CPU 调用
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    // 使用核心公式计算当前线程要处理的数据索引
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // 边界检查：确保没有越界访问
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int numElements = 100000;
    size_t size = numElements * sizeof(float);
    int driverVersion = 0, runtimeVersion = 0;
    CHECK_CUDA(cudaDriverGetVersion(&driverVersion));
    CHECK_CUDA(cudaRuntimeGetVersion(&runtimeVersion));
    std::cout << "CUDA Driver API 版本: " << driverVersion
              << ", Runtime 版本: " << runtimeVersion << std::endl;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(0));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "当前设备: " << prop.name
              << " | Compute Capability: " << prop.major << "." << prop.minor
              << std::endl;

    // 2. 在 Host (CPU) 上分配内存并初始化
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        std::cerr << "Host 内存分配失败！" << std::endl;
        free(h_A); free(h_B); free(h_C);
        return 1;
    }
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 3. 在 Device (GPU) 上分配显存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));

    // 4. 将数据从 Host 拷贝到 Device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // 5. 配置线程并发模型并启动算子 (Kernel Launch)
    int threadsPerBlock = 256; // 通常设为 128, 256 或 512
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    // <<<Grid尺寸, Block尺寸>>> 是 CUDA 独有的启动语法
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 6. 将结果从 Device 拷贝回 Host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // 7. 验证结果
    bool success = true;
    for (int i = 0; i < numElements; ++i) {
        if (std::fabs(h_C[i] - 3.0f) > 1e-6f) {
            success = false;
            break;
        }
    }
    if (success) std::cout << "算子执行成功！所有结果均为 3.0" << std::endl;
    else std::cout << "算子执行失败！" << std::endl;

    // 8. 释放内存
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
