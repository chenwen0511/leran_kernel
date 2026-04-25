# CUDA Vector Add 排障记录（RTX 4090 + Ubuntu 22.04）

本文记录 `vector_add.cu` 从“编译通过但运行失败”到“最终运行成功”的完整调试过程，方便后续复现和排错。

## 1. 初始现象

- 编译成功：`nvcc vector_add.cu -o vector_add`
- 运行失败：`./vector_add` 输出“算子执行失败”
- 环境信息：
  - `nvidia-smi` 正常，GPU 为 `RTX 4090`
  - `nvcc --version` 最初是 `11.5`

## 2. 第一次定位：给代码加 CUDA 错误检查

在 `vector_add.cu` 中加入了：

- `CHECK_CUDA(...)` 宏，用于检查每个 CUDA API 的返回值
- 在 kernel launch 后增加：
  - `cudaGetLastError()`
  - `cudaDeviceSynchronize()`
- 在程序启动处增加：
  - `cudaDriverGetVersion`
  - `cudaRuntimeGetVersion`
  - `cudaSetDevice(0)`
  - `cudaGetDeviceProperties`

这样可以把“静默失败”转成“明确失败点 + 错误信息”。

## 3. 关键报错与结论

出现过两类关键报错：

1. `cudaMalloc(...)` 失败（unknown error / OS call failed）
2. `cudaSetDevice(0)` 失败（unknown error）

这说明问题不是 kernel 算法本身，而是 CUDA 运行时初始化层面的问题（工具链版本与环境状态）。

## 4. 根因

重启后检查发现：

- `which nvcc` 指向 `/usr/bin/nvcc`
- `nvcc --version` 仍是 `11.5`

即：虽然安装了 CUDA 12.4，但当前 shell 实际使用的是旧版编译器，导致运行时行为异常。

## 5. 修复步骤

先在当前终端切换到 CUDA 12.4：

```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64
```

然后确认版本：

```bash
which nvcc
nvcc --version
```

应显示 `/usr/local/cuda-12.4/bin/nvcc` 和 `release 12.4`。

最后编译运行：

```bash
nvcc -arch=sm_89 vector_add.cu -o vector_add
./vector_add
```

## 6. 最终结果

程序输出：

- `CUDA Driver API 版本: 13010, Runtime 版本: 12040`
- `当前设备: NVIDIA GeForce RTX 4090 | Compute Capability: 8.9`
- `算子执行成功！所有结果均为 3.0`

说明 CUDA 程序已在 4090 上正常执行。

## 7. 建议（避免下次踩坑）

- 把 CUDA 12.4 的 `PATH/LD_LIBRARY_PATH` 写入 `~/.bashrc`，避免重启后回退到 `nvcc 11.5`
- Ada 架构（4090）编译时建议显式指定：`-arch=sm_89`
- 新写 CUDA 程序时保留 `CHECK_CUDA` 和 `cudaDeviceSynchronize`，可显著缩短排障时间

## 8. `vector_add.cu` 核心逻辑解读

这份代码本质上是在演示 CUDA 程序的标准执行闭环：Host 准备数据 -> Device 并行计算 -> Host 验证结果。

### 8.1 Kernel：GPU 上真正执行的函数

`vectorAdd` 用 `__global__` 声明，表示它由 CPU 发起调用、在 GPU 上并行执行。  
每个线程先计算自己的全局索引：

`i = blockDim.x * blockIdx.x + threadIdx.x`

然后执行边界判断，防止越界访问：

- 若 `i < numElements`，执行 `C[i] = A[i] + B[i]`
- 否则该线程不做事

这就是“一个线程负责一个元素”的并行模式。

### 8.2 Host 侧流程：分配、拷贝、启动、回传

`main()` 里按顺序做了 8 件事：

1. 设定数据规模：`numElements = 100000`
2. 在 CPU 内存分配并初始化 `h_A/h_B/h_C`
3. 在 GPU 显存分配 `d_A/d_B/d_C`（`cudaMalloc`）
4. 把输入数据从 Host 拷到 Device（`cudaMemcpy H2D`）
5. 设置并行配置并启动 kernel（`vectorAdd<<<grid, block>>>`）
6. 把结果从 Device 拷回 Host（`cudaMemcpy D2H`）
7. 在 CPU 侧验证结果是否都接近 `3.0f`
8. 释放 GPU 和 CPU 内存

### 8.3 为什么选择 `threadsPerBlock = 256`

`256` 是常见默认值，原因是：

- 通常能较好匹配多数 GPU 的调度粒度
- 在示例程序里足够稳定，便于教学和调试

网格大小由下面公式自动覆盖全部元素：

`blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock`

这样即使 `numElements` 不是 `threadsPerBlock` 的整数倍，也能保证每个元素都有线程处理。

### 8.4 错误检查为什么重要

本项目里新增的 `CHECK_CUDA(...)`、`cudaGetLastError()`、`cudaDeviceSynchronize()` 是关键调试手段：

- CUDA API 出错会立刻定位到具体调用点
- kernel 异步错误会在同步点被明确暴露

没有这些检查时，程序可能只表现为“结果不对”，但看不到真正失败原因。

## 9. 第二阶段（PyTorch 扩展）实战过程复盘

这一阶段目标是把 CUDA Kernel 封装成 Python 可直接调用的 `custom_ops` 模块。

### 9.1 现象与报错

执行 `python setup.py install` 时出现：

- `RuntimeError: The detected CUDA version mismatches the version that was used to compile PyTorch`
- 报错中出现 `detected CUDA: 12.8`、`torch compiled CUDA: 13.0`

这说明 `torch` 的 CUDA ABI 版本与本机扩展编译工具链不一致，`torch.utils.cpp_extension` 会直接拒绝编译。

### 9.2 现场诊断

通过 Python 脚本确认到：

- `torch: 2.11.0+cu130`
- `torch.version.cuda: 13.0`
- `nvcc` 在另一些 shell 里是 `12.x`（一度是 12.4）

结论：版本栈混用了，导致扩展构建阶段触发 mismatch。

### 9.3 方案选择与修复（方案 A）

你选择了方案 A：让 `torch` 对齐到 `cu128`。

实际修复动作：

1. 卸载旧包：`torch/torchvision/torchcodec`
2. 从 `cu128` 索引重装：`torch torchvision`
3. 验证版本对齐：
   - `torch: 2.11.0+cu128`
   - `torch.version.cuda: 12.8`
4. 处理“激活环境后回退旧 nvcc”的问题：
   - 在 `jax_env` 增加 conda `activate.d/deactivate.d` 脚本
   - 强制 `CUDA_HOME=/usr/local/cuda-12.8`
   - 强制 PATH 优先使用 `/usr/local/cuda-12.8/bin/nvcc`

### 9.4 闭环验证

`python setup.py install` 成功，关键日志显示：

- 使用 `/usr/local/cuda-12.8/bin/nvcc` 编译
- 正确生成并安装 `custom_ops` 动态库

`python test.py` 输出：

- `初始化 GPU Tensors...`
- `调用自定义 CUDA 算子...`
- `C 的前 5 个元素: tensor([3., 3., 3., 3., 3.], device='cuda:0')`
- `结果是否全部正确 (期望值为3.0)? True`

说明 PyTorch -> C++ Extension -> CUDA Kernel 的完整调用链已经跑通。

## 10. 新增代码详解：`vector_add_pt.cu`

### 10.1 Kernel 部分（与裸 CUDA 版本一致）

- `vectorAdd` 仍然是标准并行索引模型
- 每个线程计算一个元素：`C[i] = A[i] + B[i]`

这保证了你从裸 CUDA 迁移到 PyTorch 扩展时，核心计算逻辑不变，变化只在“接口层”。

### 10.2 C++ 接口函数：`vector_add_forward`

`vector_add_forward(torch::Tensor A, torch::Tensor B)` 做了几件关键事情：

1. `TORCH_CHECK` 约束输入必须是 CUDA Tensor（防止 CPU Tensor 误传）
2. 用 `torch::empty_like(A)` 分配输出 Tensor `C`
3. 通过 `A.numel()` 获取总元素个数
4. 计算 `threadsPerBlock / blocksPerGrid`
5. 把 Tensor 的底层显存指针（`data_ptr<float>()`）传入 kernel
6. 返回 `C` 给 Python

核心价值在于：把 Python 层的 Tensor 安全地映射到 CUDA 原生指针，再把结果回传到 Python Tensor 语义。

### 10.3 Pybind11 导出

`PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` 这一段把 C++ 函数导出为 Python 模块接口：

- Python 侧可以直接 `import custom_ops`
- 并调用 `custom_ops.forward(a, b)`

这就是 C++/CUDA 与 Python 框架之间的桥。

## 11. 新增代码详解：`setup.py`

`setup.py` 的作用是告诉 Python 如何构建你的 CUDA 扩展：

- `CUDAExtension('custom_ops', ['vector_add_pt.cu'])`：定义扩展名和源文件
- `BuildExtension`：调用 PyTorch 提供的构建流程（内部会调 nvcc/g++）

当你执行 `python setup.py install` 时，本质是在编译并安装 `custom_ops*.so` 到当前环境 `site-packages`。

## 12. 新增代码详解：`test.py`

`test.py` 是最小功能验证脚本，分三步：

1. 在 GPU 上构造输入 Tensor：
   - `a` 全 1
   - `b` 全 2
2. 调用自定义算子：`c = custom_ops.forward(a, b)`
3. 打印前 5 个元素并做整体正确性判断：`torch.all(c == 3.0)`

这个脚本验证了两件事：

- 功能正确（输出值对）
- 路径正确（确实走到了 CUDA 扩展而不是 Python fallback）

## 13. 本项目当前状态（已达成）

- 裸 CUDA 程序 `vector_add.cu` 可在 4090 成功运行
- PyTorch C++/CUDA 扩展 `custom_ops` 成功编译并安装
- Python 端调用 `custom_ops.forward` 成功且结果正确
- 环境版本一致性（`torch cu128` + `nvcc 12.8`）已固定到 `jax_env`

## 14. 第三步（Naive MatMul）学习过程复盘

你在第三步已经完成了“二维 CUDA Kernel + PyTorch 绑定 + 性能/精度对比测试”的完整闭环。

终端实测结果：

- 输入规模：`A(4096x4096)`、`B(4096x4096)`
- 朴素 CUDA 算子：`0.0270 s`
- PyTorch 官方（cuBLAS）：`0.0026 s`
- 正确性：`True`

这说明两件事都成立：

1. 算法实现是正确的（数值与 `torch.matmul` 对齐）
2. 性能仍有巨大优化空间（当前约慢一个数量级）

## 15. 新增代码解读：`matmul_pt.cu`

### 15.1 二维线程映射（核心升级点）

相比向量加法的一维索引，矩阵乘法采用二维索引：

- `row = blockIdx.y * blockDim.y + threadIdx.y`
- `col = blockIdx.x * blockDim.x + threadIdx.x`

含义是：每个线程负责输出矩阵 `C[row, col]` 的一个元素。  
这一步是从 1D 并行模型升级到 2D 并行模型的关键。

### 15.2 Kernel 计算逻辑（点积）

在 `if (row < M && col < N)` 保护下，线程执行：

- 初始化 `value = 0`
- 遍历 `k in [0, K)`，累计
  - `A[row * K + k] * B[k * N + col]`
- 写回 `C[row * N + col] = value`

这里显式展示了二维矩阵在显存中按一维地址访问（行主序展开）。

### 15.3 PyTorch 接口层：`matmul_forward`

`matmul_forward(torch::Tensor A, torch::Tensor B)` 里新增了三类保障：

1. 设备与内存布局检查
   - `A/B` 必须是 CUDA Tensor
   - `A/B` 必须 `contiguous`
2. 维度合法性检查
   - `A.size(1) == B.size(0)`
3. 输出分配与 kernel 启动
   - `C = torch::empty({M, N}, A.options())`
   - `threadsPerBlock(16, 16)`，总线程 256
   - `blocksPerGrid` 采用向上取整，确保覆盖全部元素

### 15.4 Python 绑定

通过：

- `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`
- `m.def("forward", &matmul_forward, ...)`

把 C++/CUDA 入口导出为 Python 可调用接口 `custom_matmul.forward(...)`。

## 16. 新增代码解读：`setup.py` 的变化

你把扩展编译从“单算子”升级为“多算子同仓”：

- 原有：`custom_ops <- vector_add_pt.cu`
- 新增：`custom_matmul <- matmul_pt.cu`

这意味着一次 `python setup.py install` 会同时构建两个 CUDA 扩展模块，后续实验管理更方便。

## 17. 新增代码解读：`test_matmul.py`

这个测试脚本设计得很标准，包含了性能测量的关键细节：

1. 构造大矩阵并 `contiguous()`
2. 先预热（避免首次调用包含初始化开销）
3. 用 `torch.cuda.synchronize()` 包裹计时区间，保证计时准确
4. 与 `torch.matmul`（底层 cuBLAS）同输入对比
5. 用 `torch.allclose(..., atol=1e-3)` 做浮点容差验证

这份脚本既是功能测试，也是基准测试（baseline benchmark）。

## 18. 为什么正确但比 cuBLAS 慢

当前 `matmul_kernel` 是“朴素全局内存版本”：

- 每个线程在 `K` 循环中频繁访问全局显存
- 没有使用 shared memory 做 tile 复用
- 没有做寄存器/访存合并/向量化等优化

而 cuBLAS 会做高度工程化优化（分块、共享内存、流水线、指令级并行等），所以大幅更快是预期结果，不是代码错误。

## 19. 第三步学习收获总结

通过这一步，你已经掌握了大模型算子开发中非常关键的工程能力：

- 从一维并行思维升级到二维线程映射
- 从“写 Kernel”升级到“写可被 PyTorch 调用的扩展”
- 从“只看正确性”升级到“正确性 + 性能基准”双验证

下一步如果继续优化 MatMul，最优先就是引入 **Shared Memory Tiling**，这是从“能跑”走向“能快”的第一道门槛。

## 20. 共享内存优化（Shared Memory）实战复盘

你已经完成了下一阶段：把 Naive MatMul 升级为 Shared Memory Tiling 版本，并成功跑通测试。

终端输出显示：

- `你的 Shared Memory 版耗时: 0.0207 秒`
- `PyTorch 官方 (cuBLAS) 耗时: 0.0343 秒`

这说明共享内存版本已经正确执行且性能有明显提升。  
由于单次计时受系统状态、GPU 动态频率、首次调用路径和样本次数影响，建议用多次迭代取平均值做最终结论（下文会给建议）。

## 21. 新增代码原理详解：`matmul_shared_pt.cu`

### 21.1 为什么 Shared Memory 能加速

Naive 版本的问题是：每个线程都从全局显存反复读取 A/B 元素，访存延迟高。  
Shared Memory 版本把计算拆成 `16x16` 的小块（tile）：

1. 先把 tile 从全局显存搬到共享内存
2. 块内线程同步后，在共享内存中做乘加
3. 重复滑动 tile，累计结果

核心收益是“数据复用”：同一块数据在共享内存中可被多个线程重复使用，显著减少全局显存访问。

### 21.2 关键结构一：`TILE_SIZE = 16`

```cpp
#define TILE_SIZE 16
```

含义：

- 每个线程块尺寸是 `16 x 16 = 256` 个线程
- 每次处理输出矩阵的一个 `16x16` 子块
- `256` 是常见、稳定的线程块规模，通常能较好兼顾并行度和资源占用

### 21.3 关键结构二：共享内存 tile

```cpp
__shared__ float sA[TILE_SIZE][TILE_SIZE];
__shared__ float sB[TILE_SIZE][TILE_SIZE];
```

含义：

- `sA`/`sB` 是 block 级共享缓存
- 一个 block 内所有线程可读写同一份 tile
- 这是“团队搬运 + 团队计算”的数据中心

### 21.4 关键结构三：全局坐标与线程职责

```cpp
int row = blockIdx.y * TILE_SIZE + threadIdx.y;
int col = blockIdx.x * TILE_SIZE + threadIdx.x;
```

每个线程只负责输出矩阵 `C[row, col]` 的一个元素；  
但这个元素需要跨越整个 `K` 维做点积，所以会在 `t` 循环里不断累加。

### 21.5 关键结构四：沿 K 维分块滑动（Tiling Loop）

```cpp
int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
for (int t = 0; t < numTiles; ++t) { ... }
```

本质是把长点积拆成多个短点积，每次只处理一个 tile 宽度。  
这个“滑窗”过程是 GPU MatMul 优化最核心的思想之一。

### 21.6 关键结构五：协同加载 + 越界补零

```cpp
if (row < M && t * TILE_SIZE + threadIdx.x < K) {
    sA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
} else {
    sA[threadIdx.y][threadIdx.x] = 0.0f;
}
```

`B` 的加载逻辑同理。  
意义：

- 让 tile 边界外的线程写入 `0`，避免非法访存
- 保持计算逻辑统一，不需要为尾块写复杂分支

### 21.7 关键结构六：两次 `__syncthreads()` 的必要性

第一次同步（加载后）：

- 确保所有线程都完成了 `sA/sB` 的写入
- 否则有线程可能读到未写完的数据

第二次同步（计算后）：

- 确保所有线程都用完当前 tile
- 才能安全进入下一轮并覆盖共享内存

这两个同步点是正确性关键，少一个都可能导致随机错误结果。

### 21.8 关键结构七：寄存器累加与最终写回

```cpp
float value = 0.0f;
...
value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
...
C[row * N + col] = value;
```

每个线程在寄存器中维护私有累加器 `value`，最后一次性写回全局显存。  
这减少了中间结果的显存往返开销。

### 21.9 PyTorch 接口层保持稳定

`matmul_forward` 与 Naive 版本接口语义一致：

- 检查 `CUDA + contiguous + 维度匹配`
- 分配输出 `C`
- 配置 `dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE)`
- 启动 `matmul_shared_kernel`

这意味着你只替换了“内核实现”，而没有破坏 Python 调用方式。

## 22. 测速脚本解读：`test_speed.py`

`test_speed.py` 目标是快速比较 Shared 版本与 `torch.matmul`：

1. 构造 `4096x4096` 输入
2. 预热一次自定义算子
3. 用 `torch.cuda.synchronize()` 包裹计时
4. 打印两者耗时

这能快速观察趋势，但若要更严谨，建议：

- 两边都预热多次（如 10 次）
- 正式测量多次（如 50~100 次）取均值和 P50/P90
- 保证同一进程、同一功耗状态下测试
- 增加正确性校验：`torch.allclose(C_custom, C_pt, atol=1e-3)`

## 23. 这一阶段你真正掌握了什么

你已经从“会写 CUDA 算子”升级到“会做第一层性能优化”：

- 理解并实现了 Tiling + Shared Memory
- 掌握了 `__syncthreads()` 在并行协作中的语义
- 知道了“正确性、性能、可复现基准”三者要同时看
- 保持了 Python 接口稳定，仅替换内核实现完成迭代

这已经是走向高性能算子开发的关键里程碑。下一步最自然的方向是：`double buffering`、`vectorized load`、`wmma/tensor core`。

## 24. Tensor Cores（WMMA）阶段实战复盘

你已经成功完成 WMMA 雏形算子的编译与运行，终端结果如下：

- `你的 Tensor Cores (WMMA) 版耗时: 0.0303 秒`
- `PyTorch 官方 (cuBLAS) 耗时: 0.0828 秒`

这说明当前 WMMA 扩展链路已经打通：`Python -> PyTorch Extension -> CUDA WMMA Kernel -> Tensor Cores`。

## 25. WMMA 阶段遇到的关键报错与修复

### 25.1 编译时报 `wmma::` 无法识别

报错特征：

- `name followed by "::" must be a class or namespace name`
- 多处出现在 `wmma::fragment / load_matrix_sync / mma_sync / store_matrix_sync`

根因：

- 代码中使用了 `using namespace nvidia;`
- WMMA 实际命名空间是 `nvcuda::wmma`

修复：

- 改为 `using namespace nvcuda;`

### 25.2 Python 导入时报 `.so undefined symbol`

报错特征：

- `undefined symbol: TensorBase::data_ptr<__half>()`

根因：

- 扩展中直接调用 `A.data_ptr<half>()`
- `half` 是 CUDA 的 `__half`，与 PyTorch C++ API 的模板实例符号不一致，导致运行时链接失败

修复：

- 使用 PyTorch 类型取指针，再转换为 CUDA half：
  - `A.data_ptr<at::Half>()`
  - `B.data_ptr<at::Half>()`
  - `reinterpret_cast<const half*>(...)`

这样能同时满足 PyTorch ABI 与 WMMA kernel 的参数类型要求。

## 26. 新增代码详解：`matmul_wmma_pt.cu`

### 26.1 头文件与命名空间

核心依赖：

- `#include <mma.h>`：WMMA API
- `#include <cuda_fp16.h>`：`half` 类型

命名空间：

- `nvcuda::wmma` 才是正确入口

### 26.2 WMMA tile 维度

```cpp
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
```

含义：

- 每次 `mma_sync` 计算一个 `16x16x16` 的矩阵乘加块
- 这是 Tensor Core 常见、硬件友好的基础粒度

### 26.3 Fragment（片段）模型

```cpp
wmma::fragment<wmma::matrix_a, ...> a_frag;
wmma::fragment<wmma::matrix_b, ...> b_frag;
wmma::fragment<wmma::accumulator, ..., float> acc_frag;
```

理解重点：

- fragment 不是普通数组，而是 Warp 级协作数据结构
- `accumulator` 用 `float`，实现 FP16 输入、FP32 累加，减少数值误差

### 26.4 核心计算流程

每轮 `t` 循环做三件事：

1. `wmma::load_matrix_sync` 加载 A/B tile 到 fragment
2. `wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag)` 在 Tensor Cores 上做乘加
3. 循环结束后 `wmma::store_matrix_sync` 写回 C

本质是：沿 K 维分块，逐块累加。

### 26.5 类型桥接（最关键修复点）

现在的调用方式：

```cpp
reinterpret_cast<const half*>(A.data_ptr<at::Half>())
reinterpret_cast<const half*>(B.data_ptr<at::Half>())
```

为什么正确：

- `data_ptr<at::Half>()` 保证走 PyTorch 已导出的 ABI
- `reinterpret_cast` 让 kernel 参数匹配 CUDA `half*`

这是解决 `.so undefined symbol` 的关键。

### 26.6 当前实现的工程特征

当前 WMMA kernel 是“教学雏形”：

- 逻辑清晰，便于理解 WMMA API
- 线程组织仍较朴素（`16x16` block，资源利用未最优化）
- 已具备进一步优化基础（warp mapping、多 warp/block、shared staging 等）

## 27. 新增代码详解：`test_tensor_cores.py`

脚本关键点：

1. 输入强制使用 `torch.float16`（WMMA 前提）
2. 先预热一次，避免首次执行开销污染
3. 计时前后用 `torch.cuda.synchronize()`，确保异步 CUDA 计时准确
4. 用 `torch.matmul` 作为 cuBLAS 对照组
5. 将 `torch.matmul` 结果转为 FP32（与自定义输出精度对齐）

当前脚本主要做性能对比；如果要更稳健，建议补充：

- `torch.allclose(C_custom, C_pt, atol=...)` 正确性校验
- 多轮统计（mean / p50 / p90）

## 28. WMMA 阶段学习收获

你已经掌握了 Tensor Core 路线的核心入口能力：

- 会写并编译 WMMA 基础 kernel
- 理解 fragment / load / mma / store 的计算流水
- 能定位并修复 C++ 扩展常见 ABI/符号问题
- 能把 FP16 输入 + FP32 累加接入 PyTorch 调用链

这一步是从“shared memory 优化”走向“硬件专用计算单元优化”的关键跨越。
