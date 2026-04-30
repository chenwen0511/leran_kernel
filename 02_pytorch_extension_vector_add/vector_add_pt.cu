#include <torch/extension.h> // 引入 PyTorch C++ API

// 1. 依然是你的那个 CUDA 算子，没有任何改变！
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// 2. 编写 C++ 接口函数，接收 PyTorch Tensor，返回 PyTorch Tensor
torch::Tensor vector_add_forward(torch::Tensor A, torch::Tensor B) {
    // 安全检查：确保输入是在 GPU 上的连续 float32 张量
    TORCH_CHECK(A.device().is_cuda(), "Tensor A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "Tensor B must be a CUDA tensor");
    
    // 创建一个空的 Tensor 来装结果，形状和 A 一样
    auto C = torch::empty_like(A);
    int numElements = A.numel(); // 获取元素总数

    // 配置线程模型
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // 3. 启动算子！
    // A.data_ptr<float>() 的作用是提取 PyTorch Tensor 底层的裸显存指针喂给 CUDA
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        numElements
    );

    return C; // 返回给 Python
}

// 4. Pybind11 绑定代码：告诉 Python 这个模块叫什么，里面有什么函数
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 在 Python 里它将被调用为: module.forward
    m.def("forward", &vector_add_forward, "Custom Vector Add on CUDA");
}
