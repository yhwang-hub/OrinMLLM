/**
 * Batched GEMV Kernel Implementation
 * 
 * Implements batched GEMV operations inspired by llama.cpp's ncols_dst approach.
 * The key insight is that during decode, we can process multiple tokens in parallel,
 * reading the weight matrix only once but computing outputs for all tokens.
 * 
 * For Qwen2.5-7B with ~28GB weights:
 * - Single token: 28GB memory read
 * - N tokens batched: 28GB memory read (shared across all tokens)
 * - Theoretical speedup: Up to N times (bandwidth limited)
 */

#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "batched_gemv_kernel.cuh"

namespace kernel {

/**
 * Batched GEMV kernel using template-based column count
 * 
 * Each block handles one output row, each thread accumulates partial sums
 * for all batch columns (ncols_dst tokens).
 * 
 * Template Parameters:
 * - BLOCK_SIZE: threads per block (typically 256)
 * - NCOLS_DST: number of output columns (batch size), typically 1-8
 */
template <int BLOCK_SIZE = 256, int NCOLS_DST = 1>
__global__ void batched_gemv_kernel(
    const float* __restrict__ input,    // [NCOLS_DST, M]
    const float* __restrict__ weight,   // [K, M]
    float* __restrict__ output,         // [NCOLS_DST, K]
    const int M,                        // input dimension
    const int K,                        // output dimension
    const int stride_input,             // stride between input columns (usually M)
    const int stride_output             // stride between output columns (usually K)
) {
    const int row = blockIdx.x;  // output row index
    const int tid = threadIdx.x;
    
    if (row >= K) return;
    
    const float* weight_row = weight + row * M;
    
    // Each thread maintains partial sums for all columns
    float sums[NCOLS_DST] = {0.0f};
    
    // Vectorized load using float4
    constexpr int vec_size = 4;
    const int num_vecs = M / vec_size;
    
    const float4* weight_vec = reinterpret_cast<const float4*>(weight_row);
    
    // Main loop: process elements in chunks of 4
    #pragma unroll 4
    for (int i = tid; i < num_vecs; i += BLOCK_SIZE) {
        float4 w = weight_vec[i];
        
        // For each column (token in batch), compute dot product contribution
        #pragma unroll
        for (int col = 0; col < NCOLS_DST; ++col) {
            const float4* input_vec = reinterpret_cast<const float4*>(input + col * stride_input);
            float4 x = input_vec[i];
            sums[col] += w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w;
        }
    }
    
    // Handle remainder
    const int base = num_vecs * vec_size;
    for (int i = base + tid; i < M; i += BLOCK_SIZE) {
        float w = weight_row[i];
        #pragma unroll
        for (int col = 0; col < NCOLS_DST; ++col) {
            sums[col] += w * input[col * stride_input + i];
        }
    }
    
    // Block-level reduction for each column
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    #pragma unroll
    for (int col = 0; col < NCOLS_DST; ++col) {
        float sum = BlockReduce(temp_storage).Sum(sums[col]);
        __syncthreads();  // Required between BlockReduce calls
        
        if (tid == 0) {
            output[col * stride_output + row] = sum;
        }
    }
}

/**
 * Batched Fused Gate-Up-SwiGLU kernel
 * 
 * Combines W1 @ input (gate), W3 @ input (up), and SwiGLU activation
 * for multiple tokens in a single pass.
 * 
 * output[col, row] = silu(sum(W1[row,:] * input[col,:])) * sum(W3[row,:] * input[col,:])
 */
template <int BLOCK_SIZE = 256, int NCOLS_DST = 1>
__global__ void batched_fused_gate_up_swiglu_kernel(
    const float* __restrict__ input,    // [NCOLS_DST, M]
    const float* __restrict__ w1,       // [K, M] gate projection
    const float* __restrict__ w3,       // [K, M] up projection
    float* __restrict__ output,         // [NCOLS_DST, K]
    const int M,                        // input dimension
    const int K,                        // output dimension (hidden_dim)
    const int stride_input,             // stride between input columns
    const int stride_output             // stride between output columns
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= K) return;
    
    const float* w1_row = w1 + row * M;
    const float* w3_row = w3 + row * M;
    
    // Each thread maintains partial sums for gate and up projections for all columns
    float sums_gate[NCOLS_DST] = {0.0f};
    float sums_up[NCOLS_DST] = {0.0f};
    
    // Vectorized load
    constexpr int vec_size = 4;
    const int num_vecs = M / vec_size;
    
    const float4* w1_vec = reinterpret_cast<const float4*>(w1_row);
    const float4* w3_vec = reinterpret_cast<const float4*>(w3_row);
    
    #pragma unroll 4
    for (int i = tid; i < num_vecs; i += BLOCK_SIZE) {
        float4 g = w1_vec[i];  // gate weights
        float4 u = w3_vec[i];  // up weights
        
        #pragma unroll
        for (int col = 0; col < NCOLS_DST; ++col) {
            const float4* input_vec = reinterpret_cast<const float4*>(input + col * stride_input);
            float4 x = input_vec[i];
            
            // Accumulate gate and up projections
            sums_gate[col] += g.x * x.x + g.y * x.y + g.z * x.z + g.w * x.w;
            sums_up[col] += u.x * x.x + u.y * x.y + u.z * x.z + u.w * x.w;
        }
    }
    
    // Handle remainder
    const int base = num_vecs * vec_size;
    for (int i = base + tid; i < M; i += BLOCK_SIZE) {
        float g = w1_row[i];
        float u = w3_row[i];
        
        #pragma unroll
        for (int col = 0; col < NCOLS_DST; ++col) {
            float x = input[col * stride_input + i];
            sums_gate[col] += g * x;
            sums_up[col] += u * x;
        }
    }
    
    // Block-level reduction
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage_gate;
    __shared__ typename BlockReduce::TempStorage temp_storage_up;
    
    #pragma unroll
    for (int col = 0; col < NCOLS_DST; ++col) {
        float sum_gate = BlockReduce(temp_storage_gate).Sum(sums_gate[col]);
        __syncthreads();
        float sum_up = BlockReduce(temp_storage_up).Sum(sums_up[col]);
        __syncthreads();
        
        if (tid == 0) {
            // SiLU activation: silu(x) = x / (1 + exp(-x))
            float gate_activated = sum_gate / (1.0f + expf(-sum_gate));
            output[col * stride_output + row] = gate_activated * sum_up;
        }
    }
}

// Dispatch helper to select the right kernel based on batch size
template <int BLOCK_SIZE>
void dispatch_batched_gemv_by_ncols(
    const float* input,
    const float* weight,
    float* output,
    int M, int K,
    int batch_size,
    int stride_input,
    int stride_output,
    cudaStream_t stream
) {
    dim3 grid(K);
    dim3 block(BLOCK_SIZE);
    
    switch (batch_size) {
        case 1:
            batched_gemv_kernel<BLOCK_SIZE, 1><<<grid, block, 0, stream>>>(
                input, weight, output, M, K, stride_input, stride_output);
            break;
        case 2:
            batched_gemv_kernel<BLOCK_SIZE, 2><<<grid, block, 0, stream>>>(
                input, weight, output, M, K, stride_input, stride_output);
            break;
        case 3:
            batched_gemv_kernel<BLOCK_SIZE, 3><<<grid, block, 0, stream>>>(
                input, weight, output, M, K, stride_input, stride_output);
            break;
        case 4:
            batched_gemv_kernel<BLOCK_SIZE, 4><<<grid, block, 0, stream>>>(
                input, weight, output, M, K, stride_input, stride_output);
            break;
        case 5:
            batched_gemv_kernel<BLOCK_SIZE, 5><<<grid, block, 0, stream>>>(
                input, weight, output, M, K, stride_input, stride_output);
            break;
        case 6:
            batched_gemv_kernel<BLOCK_SIZE, 6><<<grid, block, 0, stream>>>(
                input, weight, output, M, K, stride_input, stride_output);
            break;
        case 7:
            batched_gemv_kernel<BLOCK_SIZE, 7><<<grid, block, 0, stream>>>(
                input, weight, output, M, K, stride_input, stride_output);
            break;
        case 8:
            batched_gemv_kernel<BLOCK_SIZE, 8><<<grid, block, 0, stream>>>(
                input, weight, output, M, K, stride_input, stride_output);
            break;
        default:
            // For larger batch sizes, fall back to iterative processing or cuBLAS
            // This should not happen in typical decode scenarios
            for (int b = 0; b < batch_size; ++b) {
                batched_gemv_kernel<BLOCK_SIZE, 1><<<grid, block, 0, stream>>>(
                    input + b * stride_input, weight, output + b * stride_output,
                    M, K, stride_input, stride_output);
            }
            break;
    }
}

// Dispatch helper for fused gate-up-swiglu
template <int BLOCK_SIZE>
void dispatch_batched_fused_swiglu_by_ncols(
    const float* input,
    const float* w1,
    const float* w3,
    float* output,
    int M, int K,
    int batch_size,
    int stride_input,
    int stride_output,
    cudaStream_t stream
) {
    dim3 grid(K);
    dim3 block(BLOCK_SIZE);
    
    switch (batch_size) {
        case 1:
            batched_fused_gate_up_swiglu_kernel<BLOCK_SIZE, 1><<<grid, block, 0, stream>>>(
                input, w1, w3, output, M, K, stride_input, stride_output);
            break;
        case 2:
            batched_fused_gate_up_swiglu_kernel<BLOCK_SIZE, 2><<<grid, block, 0, stream>>>(
                input, w1, w3, output, M, K, stride_input, stride_output);
            break;
        case 3:
            batched_fused_gate_up_swiglu_kernel<BLOCK_SIZE, 3><<<grid, block, 0, stream>>>(
                input, w1, w3, output, M, K, stride_input, stride_output);
            break;
        case 4:
            batched_fused_gate_up_swiglu_kernel<BLOCK_SIZE, 4><<<grid, block, 0, stream>>>(
                input, w1, w3, output, M, K, stride_input, stride_output);
            break;
        case 5:
            batched_fused_gate_up_swiglu_kernel<BLOCK_SIZE, 5><<<grid, block, 0, stream>>>(
                input, w1, w3, output, M, K, stride_input, stride_output);
            break;
        case 6:
            batched_fused_gate_up_swiglu_kernel<BLOCK_SIZE, 6><<<grid, block, 0, stream>>>(
                input, w1, w3, output, M, K, stride_input, stride_output);
            break;
        case 7:
            batched_fused_gate_up_swiglu_kernel<BLOCK_SIZE, 7><<<grid, block, 0, stream>>>(
                input, w1, w3, output, M, K, stride_input, stride_output);
            break;
        case 8:
            batched_fused_gate_up_swiglu_kernel<BLOCK_SIZE, 8><<<grid, block, 0, stream>>>(
                input, w1, w3, output, M, K, stride_input, stride_output);
            break;
        default:
            // Fallback for larger batches
            for (int b = 0; b < batch_size; ++b) {
                batched_fused_gate_up_swiglu_kernel<BLOCK_SIZE, 1><<<grid, block, 0, stream>>>(
                    input + b * stride_input, w1, w3, output + b * stride_output,
                    M, K, stride_input, stride_output);
            }
            break;
    }
}

// Public API implementation
void batched_gemv_kernel_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& weight,
    tensor::Tensor& output,
    int batch_size,
    const CudaConfig* config
) {
    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA) << "Input must be on CUDA device";
    CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA) << "Weight must be on CUDA device";
    CHECK(output.device_type() == base::DeviceType::kDeviceCUDA) << "Output must be on CUDA device";
    
    const int M = weight.get_dim(1);  // input dimension
    const int K = weight.get_dim(0);  // output dimension
    
    const float* input_ptr = input.ptr<float>();
    const float* weight_ptr = weight.ptr<float>();
    float* output_ptr = output.ptr<float>();
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    // Stride is M for row-major layout [batch, M]
    const int stride_input = M;
    const int stride_output = K;
    
    // Choose block size based on M
    // Larger M benefits from larger block sizes
    if (M >= 4096) {
        dispatch_batched_gemv_by_ncols<256>(
            input_ptr, weight_ptr, output_ptr, M, K,
            batch_size, stride_input, stride_output, stream);
    } else if (M >= 1024) {
        dispatch_batched_gemv_by_ncols<128>(
            input_ptr, weight_ptr, output_ptr, M, K,
            batch_size, stride_input, stride_output, stream);
    } else {
        dispatch_batched_gemv_by_ncols<64>(
            input_ptr, weight_ptr, output_ptr, M, K,
            batch_size, stride_input, stride_output, stream);
    }
}

void batched_fused_gate_up_swiglu_kernel_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& w1,
    const tensor::Tensor& w3,
    tensor::Tensor& output,
    int batch_size,
    const CudaConfig* config
) {
    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA) << "Input must be on CUDA device";
    CHECK(w1.device_type() == base::DeviceType::kDeviceCUDA) << "W1 must be on CUDA device";
    CHECK(w3.device_type() == base::DeviceType::kDeviceCUDA) << "W3 must be on CUDA device";
    CHECK(output.device_type() == base::DeviceType::kDeviceCUDA) << "Output must be on CUDA device";
    
    const int M = w1.get_dim(1);  // input dimension
    const int K = w1.get_dim(0);  // output dimension (hidden_dim)
    
    CHECK_EQ(w3.get_dim(0), K);
    CHECK_EQ(w3.get_dim(1), M);
    
    const float* input_ptr = input.ptr<float>();
    const float* w1_ptr = w1.ptr<float>();
    const float* w3_ptr = w3.ptr<float>();
    float* output_ptr = output.ptr<float>();
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    const int stride_input = M;
    const int stride_output = K;
    
    // Choose block size based on M
    if (M >= 4096) {
        dispatch_batched_fused_swiglu_by_ncols<256>(
            input_ptr, w1_ptr, w3_ptr, output_ptr, M, K,
            batch_size, stride_input, stride_output, stream);
    } else if (M >= 1024) {
        dispatch_batched_fused_swiglu_by_ncols<128>(
            input_ptr, w1_ptr, w3_ptr, output_ptr, M, K,
            batch_size, stride_input, stride_output, stream);
    } else {
        dispatch_batched_fused_swiglu_by_ncols<64>(
            input_ptr, w1_ptr, w3_ptr, output_ptr, M, K,
            batch_size, stride_input, stride_output, stream);
    }
}

} // namespace kernel
