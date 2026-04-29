#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define Br 16
#define Bc 16
#define d  32

__global__ void flash_attention_forward_kernel(
    const float* Q, const float* K, const float* V, float* O, int seq_len) 
{
    int q_row_start = blockIdx.x * Br;
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 

    __shared__ float sQ[Br][d];
    __shared__ float sK[Bc][d];
    __shared__ float sV[Bc][d];
    __shared__ float S_ij[Br][Bc];

    // ✅ 修复 1：专门开辟共享内存用于跨线程传递 Softmax 参数，绝不污染 sQ
    __shared__ float s_decay[Br];
    __shared__ float s_l_new[Br];
    __shared__ float s_m_new[Br];

    float l_i = 0.0f;
    float m_i = -FLT_MAX;
    float O_row = 0.0f; 

    // ✅ 修复 2：越界保护与清零 (Padding)
    if (q_row_start + ty < seq_len && tx < d) {
        sQ[ty][tx] = Q[(q_row_start + ty) * d + tx];
    } else {
        sQ[ty][tx] = 0.0f; 
    }
    __syncthreads();

    int num_blocks_K = (seq_len + Bc - 1) / Bc;

    for (int k_idx = 0; k_idx < num_blocks_K; ++k_idx) {
        int k_row_start = k_idx * Bc;

        if (k_row_start + ty < seq_len && tx < d) {
            sK[ty][tx] = K[(k_row_start + ty) * d + tx];
            sV[ty][tx] = V[(k_row_start + ty) * d + tx];
        } else {
            sK[ty][tx] = 0.0f;
            sV[ty][tx] = 0.0f;
        }
        __syncthreads();

        if (tx == 0) {
            for (int k_col = 0; k_col < Bc; ++k_col) {
                float score = 0.0f;
                for (int i = 0; i < d; ++i) {
                    score += sQ[ty][i] * sK[k_col][i];
                }
                S_ij[ty][k_col] = score / sqrtf((float)d);
            }
        }
        __syncthreads();

        if (tx == 0) {
            float m_ij = -FLT_MAX;
            for (int c = 0; c < Bc; ++c) {
                if (k_row_start + c < seq_len) {
                    m_ij = fmaxf(m_ij, S_ij[ty][c]);
                }
            }

            float m_new = fmaxf(m_i, m_ij);
            float exponent_decay = expf(m_i - m_new);
            
            float l_ij = 0.0f;
            for (int c = 0; c < Bc; ++c) {
                if (k_row_start + c < seq_len) {
                    l_ij += expf(S_ij[ty][c] - m_new);
                }
            }

            float l_new = exponent_decay * l_i + l_ij;

            for (int c = 0; c < Bc; ++c) {
                S_ij[ty][c] = expf(S_ij[ty][c] - m_new); 
            }

            // ✅ 将参数写入专属共享内存
            s_decay[ty] = exponent_decay;
            s_l_new[ty] = l_new;
            s_m_new[ty] = m_new;
        }
        __syncthreads();

        // ✅ 从专属共享内存读取
        float exponent_decay = s_decay[ty];
        float l_new = s_l_new[ty];
        m_i = s_m_new[ty];

        float pv_sum = 0.0f;
        for (int c = 0; c < Bc; ++c) {
            if (k_row_start + c < seq_len) {
                pv_sum += S_ij[ty][c] * sV[c][tx];
            }
        }
        
        O_row = O_row * exponent_decay + pv_sum;
        l_i = l_new;
        __syncthreads();
    }

    if (q_row_start + ty < seq_len && tx < d) {
        O[(q_row_start + ty) * d + tx] = O_row / l_i;
    }
}

// (下面的 PyTorch 绑定代码无需改动，为了防止漏掉我再贴一遍)
torch::Tensor flash_attention_forward_pt(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.device().is_cuda() && K.device().is_cuda() && V.device().is_cuda(), "必须在 CUDA 上");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "内存必须连续");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "教学版仅支持 FP32");
    
    int seq_len = Q.size(0);
    TORCH_CHECK(Q.size(1) == d, "Head Dim 必须等于 d");

    auto O = torch::empty_like(Q);
    dim3 threadsPerBlock(d, Br); 
    dim3 blocksPerGrid((seq_len + Br - 1) / Br); 

    flash_attention_forward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), seq_len
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward_pt, "Toy FlashAttention Forward");
}
