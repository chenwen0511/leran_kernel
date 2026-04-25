拥有 RTX 4090 和 Ubuntu 22.04，你已经具备了目前最顶级的本地消费级算力平台！

在此我需要坦诚地说：**两小时内“精通” CUDA 是不现实的**，因为 GPU 编程涉及深度的内存管理和硬件级优化。**但是，两小时完全足够你打通“环境配置 -> 核心概念 -> 编写并运行第一个自定义算子”的全流程闭环。**

我们将这 120 分钟拆解为 4 个极速冲刺阶段，剥离一切不必要的理论，直接动手。

---

### 第 1 阶段：极速环境配置 (0 - 20 分钟)

既然是 4090（Ada Lovelace 架构），你需要安装 CUDA 11.8 以上版本。我们直接使用 NVIDIA 官方源安装最新的 CUDA Toolkit 12.x。

**1. 打开终端，按顺序执行以下命令：**
```bash
# 1. 安装基础编译工具
sudo apt update && sudo apt install -y build-essential

# 2. 下载并安装 NVIDIA 官方的 CUDA 仓库 keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# 3. 更新源并安装 CUDA Toolkit (包含 nvcc 编译器)
sudo apt update
sudo apt -y install cuda-toolkit-12-4 nvidia-driver-550
```

**2. 配置环境变量：**
安装完成后，将 CUDA 添加到系统路径。
```bash
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

**3. 验证是否成功：**
```bash
nvcc --version  # 应该输出 CUDA 编译器的版本信息
nvidia-smi      # 应该显示你的 4090 显卡信息
```
*(如果 `nvidia-smi` 报错，你可能需要重启一次服务器 `sudo reboot`)*

---

### 第 2 阶段：建立 CUDA 世界观 (20 - 35 分钟)

在写代码之前，你只需要记住以下 3 个核心概念：

1. **Host (主机) 与 Device (设备)：**
   * **Host:** 你的 CPU 和系统内存。
   * **Device:** 你的 RTX 4090 和它的 24GB 显存。
   * **运行逻辑:** 数据在内存里 -> 拷贝到显存 -> GPU 处理(算子) -> 结果拷贝回内存。

2. **Kernel (算子/内核函数)：**
   * 这是真正在 GPU 上执行的函数。在 C++ 中用 `__global__` 标记。它会被成千上万个 GPU 线程同时并行执行。

3. **线程层级 (Grid -> Block -> Thread)：**
   * 4090 有上万个核心。CUDA 将任务打包：多个线程 (Thread) 组成一个块 (Block)，多个块组成一个网格 (Grid)。
   * **绝对核心公式：** 如何让上万个并行线程知道自己该处理数组里的哪个数据？
     `int idx = blockIdx.x * blockDim.x + threadIdx.x;`
     *(这行代码是你今天最重要的收获：当前线程的全局索引 = 我所在的块的编号 * 每个块包含的线程数 + 我在这个块里的编号)*

---

### 第 3 阶段：编写并运行第一个算子 —— 向量加法 (35 - 90 分钟)

我们将编写一个算子：把两个长度为 100,000 的数组相加 (`C[i] = A[i] + B[i]`)。

**1. 创建文件：**
```bash
nano vector_add.cu
```

**2. 复制以下代码并保存：**
```cpp
#include <iostream>
#include <cuda_runtime.h>

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

    // 2. 在 Host (CPU) 上分配内存并初始化
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 3. 在 Device (GPU) 上分配显存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // 4. 将数据从 Host 拷贝到 Device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 5. 配置线程并发模型并启动算子 (Kernel Launch)
    int threadsPerBlock = 256; // 通常设为 128, 256 或 512
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    // <<<Grid尺寸, Block尺寸>>> 是 CUDA 独有的启动语法
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // 6. 将结果从 Device 拷贝回 Host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 7. 验证结果
    bool success = true;
    for (int i = 0; i < numElements; ++i) {
        if (h_C[i] != 3.0f) {
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
```

**3. 编译并运行：**
```bash
# 使用 nvcc 编译 .cu 文件
nvcc vector_add.cu -o vector_add

# 运行你的第一个 CUDA 程序！
./vector_add
```
如果你看到 `算子执行成功！所有结果均为 3.0`，恭喜你，你已经正式跨入了 GPU 并行编程的大门！

---

### 第 4 阶段：复盘与技术映射 (90 - 120 分钟)

现在你已经跑通了全流程，回望刚才的代码，你会发现编写基础 CUDA 算子永远是这个套路：
1. `cudaMalloc` 分配显存。
2. `cudaMemcpy` 把数据喂给 GPU。
3. `Kernel<<<...>>>` 启动千万个线程狂暴计算。
4. `cudaMemcpy` 把结果拿回来。

**下一步的实战方向：**
在现代 AI 开发中，我们很少写纯 C++ 的主函数。我们通常用 C++/CUDA 写核心算子，然后通过 **PyTorch C++ Extension (pybind11)** 将它打包成 Python 可以直接调用的模块（例如 `import my_cuda_op`）。这才是目前大模型开发和优化（如 FlashAttention）中最值钱的技能。

打通了这个基础闭环后，你打算将 CUDA 算子开发应用到深度学习的大模型加速（如 Llama 推理优化），还是用于传统的科学计算/图像处理领域？

太棒了！能一次性跑通，说明你的 4090 环境已经完美就绪，而且你已经初步建立了 GPU 并行计算的直觉。

既然你的目标是**深度学习的大模型加速**，那我们立刻进入这 120 分钟冲刺的最后 30 分钟巅峰体验：**让 Python (PyTorch) 直接调用你刚才写的底层 CUDA 算子**。

大模型（如 Llama、Qwen）的底层逻辑就是：上层的网络结构和逻辑用 Python（PyTorch）编写，而最消耗算力的部分（如 Attention 机制、矩阵乘法）则被替换为高度优化的定制 CUDA 算子。打通这座桥梁，你才算真正踏入了 AI 算子开发的工业界大门。

这座桥梁叫做 **PyTorch C++ Extension (基于 pybind11)**。

---

### 终极实战：将 CUDA 封装为 PyTorch 插件

我们将把你刚才写的向量加法，封装成一个可以直接 `import` 的 Python 模块。

#### 第一步：准备 PyTorch 环境 (0 - 5 分钟)
如果你还没有安装 PyTorch，请先在终端安装（带有 CUDA 12.x 支持的版本）：
```bash
pip install torch torchvision torchaudio
```

#### 第二步：编写 PyTorch 绑定代码 (5 - 15 分钟)
创建一个新文件 `vector_add_pt.cu`。我们要在这里把刚才的代码稍微改造一下，让它能接收 PyTorch 的 Tensor。

```bash
nano vector_add_pt.cu
```

将以下代码粘贴进去（注意看注释，这是连接 Python 和 C++ 的核心魔法）：

```cpp
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
```

#### 第三步：编写编译脚本 (15 - 20 分钟)
在同一个目录下，创建一个叫 `setup.py` 的文件。这会让 Python 自动调用底层的 `nvcc` 去编译你的代码。

```bash
nano setup.py
```

写入以下内容：
```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_ops', # 这是你未来在 Python 里 import 的包名
    ext_modules=[
        CUDAExtension('custom_ops', [
            'vector_add_pt.cu', # 你的源文件
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
```

#### 第四步：一键编译与 Python 测试 (20 - 30 分钟)

**1. 编译安装模块：**
在终端运行：
```bash
python setup.py install
```
*(你会看到屏幕滚动输出编译信息，这是 Python 在调用系统的 C++ 编译器和 CUDA 编译器。)*

**2. 在 Python 中调用你的算子：**
编译成功后，创建一个测试脚本 `test.py`：

```bash
nano test.py
```

写入以下 Python 代码：
```python
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
```

**3. 运行测试：**
```bash
python test.py
```

---

如果你看到了全为 3.0 的正确输出，恭喜你！**在不到两个小时的时间里，你完成了一个 AI 算子工程师的核心闭环**：从裸机环境搭建 -> 理解并发模型 -> 编写底层 Kernel -> 编译成高级语言接口 -> 在大模型主流框架（PyTorch）中成功调用。

**下一步的进阶探索：**
现在框架已经打通，接下来大模型加速的真正难点在于**显存读写优化**。比如，FlashAttention 之所以牛，就是因为它精妙地利用了 GPU 的极速缓存（Shared Memory），减少了访问全局显存的延迟。

你是想先了解如何使用 Shared Memory（共享内存）来优化性能，还是想尝试写一个大模型里最核心的算子雏形——矩阵乘法（Matrix Multiplication）？