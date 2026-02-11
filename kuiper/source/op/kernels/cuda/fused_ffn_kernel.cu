#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "fused_ffn_kernel.cuh"

namespace kernel {

/**
 * Fused Gate-Up-SwiGLU GEMV Kernel
 * 
 * Combines W1 @ input (gate), W3 @ input (up), and SwiGLU activation into a single kernel.
 * This eliminates 2 kernel launches and avoids writing/reading intermediate results.
 *
 * Memory access pattern (per row):
 * - Read input once: M floats
 * - Read W1 row: M floats  
 * - Read W3 row: M floats
 * - Write output: 1 float
 *
 * Total memory: 3M + 1 floats per row (instead of 4M + 3 with separate kernels)
 * 
 * Optimization techniques:
 * - float4 vectorized loads for coalesced memory access
 * - Warp shuffle reduction (faster than shared memory for small reductions)
 * - Single pass through input vector
 */
template <int BLOCK_SIZE = 256>
__global__ void fused_gate_up_swiglu_kernel(
    const float* __restrict__ input,    // [M]
    const float* __restrict__ w1,       // [K, M] gate projection
    const float* __restrict__ w3,       // [K, M] up projection
    float* __restrict__ output,         // [K]
    const int M,                        // input dimension
    const int K                         // output dimension
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= K) return;
    
    // Pointers to current row of each weight matrix
    const float* w1_row = w1 + row * M;
    const float* w3_row = w3 + row * M;
    
    // Accumulate gate and up projections
    float sum_gate = 0.0f;
    float sum_up = 0.0f;
    
    // Vectorized load with float4 (16 bytes per load)
    constexpr int vec_size = 4;
    const int num_vecs = M / vec_size;
    
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    const float4* w1_vec = reinterpret_cast<const float4*>(w1_row);
    const float4* w3_vec = reinterpret_cast<const float4*>(w3_row);
    
    // Main loop: process 4 elements at a time
    #pragma unroll 4
    for (int i = tid; i < num_vecs; i += BLOCK_SIZE) {
        float4 x = input_vec[i];
        float4 g = w1_vec[i];  // gate weight
        float4 u = w3_vec[i];  // up weight
        
        // Fused multiply-add for both projections
        sum_gate += g.x * x.x + g.y * x.y + g.z * x.z + g.w * x.w;
        sum_up += u.x * x.x + u.y * x.y + u.z * x.z + u.w * x.w;
    }
    
    // Handle remainder (if M is not divisible by 4)
    const int base = num_vecs * vec_size;
    for (int i = base + tid; i < M; i += BLOCK_SIZE) {
        float x_val = input[i];
        sum_gate += w1_row[i] * x_val;
        sum_up += w3_row[i] * x_val;
    }
    
    // Block-level reduction using CUB
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage_gate;
    __shared__ typename BlockReduce::TempStorage temp_storage_up;
    
    sum_gate = BlockReduce(temp_storage_gate).Sum(sum_gate);
    __syncthreads();  // Required between two BlockReduce calls with different temp storage
    sum_up = BlockReduce(temp_storage_up).Sum(sum_up);
    
    // Thread 0 computes final result: silu(gate) * up
    if (tid == 0) {
        // SiLU activation: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        float gate_activated = sum_gate / (1.0f + expf(-sum_gate));
        output[row] = gate_activated * sum_up;
    }
}

/**
 * Warp-optimized version for smaller M dimensions
 * Uses warp shuffle for reduction (no shared memory needed)
 */
template <int WARP_SIZE = 32>
__global__ void fused_gate_up_swiglu_kernel_warp(
    const float* __restrict__ input,
    const float* __restrict__ w1,
    const float* __restrict__ w3,
    float* __restrict__ output,
    const int M,
    const int K
) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    
    // Each warp handles one output row
    const int row = blockIdx.x * warps_per_block + warp_id;
    
    if (row >= K) return;
    
    const float* w1_row = w1 + row * M;
    const float* w3_row = w3 + row * M;
    
    float sum_gate = 0.0f;
    float sum_up = 0.0f;
    
    constexpr int vec_size = 4;
    const int num_vecs = M / vec_size;
    
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    const float4* w1_vec = reinterpret_cast<const float4*>(w1_row);
    const float4* w3_vec = reinterpret_cast<const float4*>(w3_row);
    
    #pragma unroll 4
    for (int i = lane_id; i < num_vecs; i += WARP_SIZE) {
        float4 x = input_vec[i];
        float4 g = w1_vec[i];
        float4 u = w3_vec[i];
        
        sum_gate += g.x * x.x + g.y * x.y + g.z * x.z + g.w * x.w;
        sum_up += u.x * x.x + u.y * x.y + u.z * x.z + u.w * x.w;
    }

    // Handle remainder
    const int base = num_vecs * vec_size;
    for (int i = base + lane_id; i < M; i += WARP_SIZE) {
        float x_val = input[i];
        sum_gate += w1_row[i] * x_val;
        sum_up += w3_row[i] * x_val;
    }
    
    // Warp reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum_gate += __shfl_down_sync(0xffffffff, sum_gate, offset);
        sum_up += __shfl_down_sync(0xffffffff, sum_up, offset);
    }
    
    // Lane 0 writes the result
    if (lane_id == 0) {
        float gate_activated = sum_gate / (1.0f + expf(-sum_gate));
        output[row] = gate_activated * sum_up;
    }
}

/**
 * Batched version for multiple tokens (prefill)
 * Uses register tiling to process multiple batch elements with single weight load
 */
template <int BLOCK_SIZE = 256, int BATCH_TILE = 4>
__global__ void batched_fused_gate_up_swiglu_kernel(
    const float* __restrict__ input,    // [batch_size, M]
    const float* __restrict__ w1,       // [K, M]
    const float* __restrict__ w3,       // [K, M]
    float* __restrict__ output,         // [batch_size, K]
    const int M,
    const int K,
    const int batch_size
) {
    const int row = blockIdx.x;
    const int batch_base = blockIdx.y * BATCH_TILE;
    const int tid = threadIdx.x;
    
    if (row >= K) return;
    
    const float* w1_row = w1 + row * M;
    const float* w3_row = w3 + row * M;
    
    // Accumulators for each batch element in tile
    float sum_gate[BATCH_TILE] = {0.0f};
    float sum_up[BATCH_TILE] = {0.0f};
    
    constexpr int vec_size = 4;
    const int num_vecs = M / vec_size;
    
    const float4* w1_vec = reinterpret_cast<const float4*>(w1_row);
    const float4* w3_vec = reinterpret_cast<const float4*>(w3_row);
    
    #pragma unroll 2
    for (int i = tid; i < num_vecs; i += BLOCK_SIZE) {
        float4 g = w1_vec[i];  // Load weight once
        float4 u = w3_vec[i];
        
        // Process each batch element in tile
        #pragma unroll
        for (int b = 0; b < BATCH_TILE; ++b) {
            int batch_idx = batch_base + b;
            if (batch_idx >= batch_size) continue;
            
            const float4* input_vec = reinterpret_cast<const float4*>(input + batch_idx * M);
            float4 x = input_vec[i];
            
            sum_gate[b] += g.x * x.x + g.y * x.y + g.z * x.z + g.w * x.w;
            sum_up[b] += u.x * x.x + u.y * x.y + u.z * x.z + u.w * x.w;
        }
    }
    
    // Handle remainder
    const int base = num_vecs * vec_size;
    for (int i = base + tid; i < M; i += BLOCK_SIZE) {
        float g_val = w1_row[i];
        float u_val = w3_row[i];
        
        #pragma unroll
        for (int b = 0; b < BATCH_TILE; ++b) {
            int batch_idx = batch_base + b;
            if (batch_idx >= batch_size) continue;
            
            float x_val = input[batch_idx * M + i];
            sum_gate[b] += g_val * x_val;
            sum_up[b] += u_val * x_val;
        }
    }
    
    // Block reduction for each batch element
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; ++b) {
        int batch_idx = batch_base + b;
        if (batch_idx >= batch_size) continue;
        
        float reduced_gate = BlockReduce(temp_storage).Sum(sum_gate[b]);
        __syncthreads();
        float reduced_up = BlockReduce(temp_storage).Sum(sum_up[b]);
        __syncthreads();
        
        if (tid == 0) {
            float gate_activated = reduced_gate / (1.0f + expf(-reduced_gate));
            output[batch_idx * K + row] = gate_activated * reduced_up;
        }
    }
}

void fused_gate_up_swiglu_kernel_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& w1,
    const tensor::Tensor& w3,
    const tensor::Tensor& output,
    const CudaConfig* config
) {
    CHECK(!input.is_empty() && input.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(!w1.is_empty() && w1.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(!w3.is_empty() && w3.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(!output.is_empty() && output.device_type() == base::DeviceType::kDeviceCUDA);
    
    const int32_t K = w1.get_dim(0);  // output dimension (intermediate_size)
    const int32_t M = w1.get_dim(1);  // input dimension (hidden_size)
    
    CHECK_EQ(w3.get_dim(0), K);
    CHECK_EQ(w3.get_dim(1), M);
    CHECK_EQ(M, input.get_dim(0));
    CHECK_EQ(K, output.get_dim(0));
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    // Choose kernel based on M dimension
    // For large M (like Qwen2.5-7B's dim=3584), use block-based kernel
    // For smaller M, warp-based kernel may be faster
    if (M > 1024) {
        constexpr int BLOCK_SIZE = 256;
        if (stream) {
            fused_gate_up_swiglu_kernel<BLOCK_SIZE><<<K, BLOCK_SIZE, 0, stream>>>(
                input.ptr<float>(), w1.ptr<float>(), w3.ptr<float>(),
                const_cast<float*>(output.ptr<float>()), M, K);
        } else {
            fused_gate_up_swiglu_kernel<BLOCK_SIZE><<<K, BLOCK_SIZE>>>(
                input.ptr<float>(), w1.ptr<float>(), w3.ptr<float>(),
                const_cast<float*>(output.ptr<float>()), M, K);
        }
    } else {
        // Warp-based kernel for smaller dimensions
        constexpr int WARPS_PER_BLOCK = 4;
        constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
        const int num_blocks = (K + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        
        if (stream) {
            fused_gate_up_swiglu_kernel_warp<32><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                input.ptr<float>(), w1.ptr<float>(), w3.ptr<float>(),
                const_cast<float*>(output.ptr<float>()), M, K);
        } else {
            fused_gate_up_swiglu_kernel_warp<32><<<num_blocks, THREADS_PER_BLOCK>>>(
                input.ptr<float>(), w1.ptr<float>(), w3.ptr<float>(),
                const_cast<float*>(output.ptr<float>()), M, K);
        }
    }
}

/**
 * Mixed precision version: FP16 weights with FP32 input/output
 * Loads FP16 weights and converts to FP32 for computation
 */
template <int BLOCK_SIZE = 256>
__global__ void fused_gate_up_swiglu_kernel_mixed(
    const float* __restrict__ input,    // [M] FP32
    const half* __restrict__ w1,        // [K, M] FP16 gate projection
    const half* __restrict__ w3,        // [K, M] FP16 up projection
    float* __restrict__ output,         // [K] FP32
    const int M,
    const int K
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= K) return;
    
    const half* w1_row = w1 + row * M;
    const half* w3_row = w3 + row * M;
    
    float sum_gate = 0.0f;
    float sum_up = 0.0f;
    
    // Vectorized load: float4 for input, half2x2 (4 halfs) for weights
    constexpr int vec_size = 4;
    const int num_vecs = M / vec_size;
    
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    const half2* w1_vec = reinterpret_cast<const half2*>(w1_row);
    const half2* w3_vec = reinterpret_cast<const half2*>(w3_row);
    
    #pragma unroll 4
    for (int i = tid; i < num_vecs; i += BLOCK_SIZE) {
        float4 x = input_vec[i];
        
        // Load 4 half values as 2 half2
        half2 g0 = w1_vec[i * 2];
        half2 g1 = w1_vec[i * 2 + 1];
        half2 u0 = w3_vec[i * 2];
        half2 u1 = w3_vec[i * 2 + 1];
        
        // Convert FP16 weights to FP32 and accumulate
        sum_gate += __half2float(g0.x) * x.x + __half2float(g0.y) * x.y 
                  + __half2float(g1.x) * x.z + __half2float(g1.y) * x.w;
        sum_up += __half2float(u0.x) * x.x + __half2float(u0.y) * x.y 
                + __half2float(u1.x) * x.z + __half2float(u1.y) * x.w;
    }
    
    // Handle remainder
    const int base = num_vecs * vec_size;
    for (int i = base + tid; i < M; i += BLOCK_SIZE) {
        float x_val = input[i];
        sum_gate += __half2float(w1_row[i]) * x_val;
        sum_up += __half2float(w3_row[i]) * x_val;
    }
    
    // Block-level reduction
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage_gate;
    __shared__ typename BlockReduce::TempStorage temp_storage_up;
    
    sum_gate = BlockReduce(temp_storage_gate).Sum(sum_gate);
    __syncthreads();
    sum_up = BlockReduce(temp_storage_up).Sum(sum_up);
    
    if (tid == 0) {
        float gate_activated = sum_gate / (1.0f + expf(-sum_gate));
        output[row] = gate_activated * sum_up;
    }
}

void fused_gate_up_swiglu_kernel_cu_mixed(
    const tensor::Tensor& input,
    const tensor::Tensor& w1,
    const tensor::Tensor& w3,
    const tensor::Tensor& output,
    const CudaConfig* config
) {
    CHECK(!input.is_empty() && input.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(!w1.is_empty() && w1.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(!w3.is_empty() && w3.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(!output.is_empty() && output.device_type() == base::DeviceType::kDeviceCUDA);
    
    // Mixed precision: FP32 input/output, FP16 weights
    CHECK(input.data_type() == base::DataType::kDataTypeFp32);
    CHECK(w1.data_type() == base::DataType::kDataTypeFp16);
    CHECK(w3.data_type() == base::DataType::kDataTypeFp16);
    CHECK(output.data_type() == base::DataType::kDataTypeFp32);
    
    const int32_t K = w1.get_dim(0);
    const int32_t M = w1.get_dim(1);
    
    CHECK_EQ(w3.get_dim(0), K);
    CHECK_EQ(w3.get_dim(1), M);
    CHECK_EQ(M, input.get_dim(0));
    CHECK_EQ(K, output.get_dim(0));
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    constexpr int BLOCK_SIZE = 256;
    if (stream) {
        fused_gate_up_swiglu_kernel_mixed<BLOCK_SIZE><<<K, BLOCK_SIZE, 0, stream>>>(
            input.ptr<float>(),
            reinterpret_cast<const half*>(w1.ptr<uint16_t>()),
            reinterpret_cast<const half*>(w3.ptr<uint16_t>()),
            const_cast<float*>(output.ptr<float>()),
            M, K);
    } else {
        fused_gate_up_swiglu_kernel_mixed<BLOCK_SIZE><<<K, BLOCK_SIZE>>>(
            input.ptr<float>(),
            reinterpret_cast<const half*>(w1.ptr<uint16_t>()),
            reinterpret_cast<const half*>(w3.ptr<uint16_t>()),
            const_cast<float*>(output.ptr<float>()),
            M, K);
    }
}

/**
 * FP16 version of fused Gate-Up-SwiGLU kernel with float4 vectorization
 * Uses warp-level reduction for higher throughput
 * Each warp processes one output row
 */
template <int WARP_SIZE = 32, int WARPS_PER_BLOCK = 4>
__global__ void fused_gate_up_swiglu_kernel_fp16_v2(
    const half* __restrict__ input,
    const half* __restrict__ w1,
    const half* __restrict__ w3,
    half* __restrict__ output,
    const int M,
    const int K
) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    
    if (row >= K) return;
    
    const half* w1_row = w1 + static_cast<int64_t>(row) * M;
    const half* w3_row = w3 + static_cast<int64_t>(row) * M;
    
    // Multiple accumulators for ILP
    float sum_gate0 = 0.0f, sum_gate1 = 0.0f, sum_gate2 = 0.0f, sum_gate3 = 0.0f;
    float sum_up0 = 0.0f, sum_up1 = 0.0f, sum_up2 = 0.0f, sum_up3 = 0.0f;
    
    // float4 vectorized load (8 halfs per load)
    const int num_float4 = M / 8;
    const float4* input_f4 = reinterpret_cast<const float4*>(input);
    const float4* w1_f4 = reinterpret_cast<const float4*>(w1_row);
    const float4* w3_f4 = reinterpret_cast<const float4*>(w3_row);
    
    #pragma unroll 4
    for (int i = lane_id; i < num_float4; i += WARP_SIZE) {
        float4 x_f4 = input_f4[i];
        float4 g_f4 = w1_f4[i];
        float4 u_f4 = w3_f4[i];
        
        // Extract half2 pairs
        const half2* x_h2 = reinterpret_cast<const half2*>(&x_f4);
        const half2* g_h2 = reinterpret_cast<const half2*>(&g_f4);
        const half2* u_h2 = reinterpret_cast<const half2*>(&u_f4);
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 xf = __half22float2(x_h2[j]);
            float2 gf = __half22float2(g_h2[j]);
            float2 uf = __half22float2(u_h2[j]);
            
            if (j == 0) {
                sum_gate0 += gf.x * xf.x + gf.y * xf.y;
                sum_up0 += uf.x * xf.x + uf.y * xf.y;
            } else if (j == 1) {
                sum_gate1 += gf.x * xf.x + gf.y * xf.y;
                sum_up1 += uf.x * xf.x + uf.y * xf.y;
            } else if (j == 2) {
                sum_gate2 += gf.x * xf.x + gf.y * xf.y;
                sum_up2 += uf.x * xf.x + uf.y * xf.y;
            } else {
                sum_gate3 += gf.x * xf.x + gf.y * xf.y;
                sum_up3 += uf.x * xf.x + uf.y * xf.y;
            }
        }
    }
    
    // Merge accumulators
    float sum_gate = sum_gate0 + sum_gate1 + sum_gate2 + sum_gate3;
    float sum_up = sum_up0 + sum_up1 + sum_up2 + sum_up3;
    
    // Handle remainder
    const int base = num_float4 * 8;
    for (int i = base + lane_id; i < M; i += WARP_SIZE) {
        float x_val = __half2float(input[i]);
        sum_gate += __half2float(w1_row[i]) * x_val;
        sum_up += __half2float(w3_row[i]) * x_val;
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum_gate += __shfl_down_sync(0xffffffff, sum_gate, offset);
        sum_up += __shfl_down_sync(0xffffffff, sum_up, offset);
    }
    
    if (lane_id == 0) {
        float gate_activated = sum_gate / (1.0f + expf(-sum_gate));
        output[row] = __float2half(gate_activated * sum_up);
    }
}

/**
 * FP16 version of fused Gate-Up-SwiGLU kernel (legacy)
 * Uses half precision throughout with FP32 accumulation for numerical stability
 */
template <int BLOCK_SIZE = 256>
__global__ void fused_gate_up_swiglu_kernel_fp16(
    const half* __restrict__ input,
    const half* __restrict__ w1,
    const half* __restrict__ w3,
    half* __restrict__ output,
    const int M,
    const int K
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= K) return;
    
    const half* w1_row = w1 + row * M;
    const half* w3_row = w3 + row * M;
    
    // Use FP32 accumulators for numerical stability
    float sum_gate = 0.0f;
    float sum_up = 0.0f;
    
    // Vectorized load with half2 (4 bytes per load)
    constexpr int vec_size = 2;
    const int num_vecs = M / vec_size;
    
    const half2* input_vec = reinterpret_cast<const half2*>(input);
    const half2* w1_vec = reinterpret_cast<const half2*>(w1_row);
    const half2* w3_vec = reinterpret_cast<const half2*>(w3_row);
    
    #pragma unroll 4
    for (int i = tid; i < num_vecs; i += BLOCK_SIZE) {
        half2 x = input_vec[i];
        half2 g = w1_vec[i];
        half2 u = w3_vec[i];
        
        // Convert to float for accumulation
        float x0 = __half2float(x.x);
        float x1 = __half2float(x.y);
        float g0 = __half2float(g.x);
        float g1 = __half2float(g.y);
        float u0 = __half2float(u.x);
        float u1 = __half2float(u.y);
        
        sum_gate += g0 * x0 + g1 * x1;
        sum_up += u0 * x0 + u1 * x1;
    }
    
    // Handle remainder
    const int base = num_vecs * vec_size;
    for (int i = base + tid; i < M; i += BLOCK_SIZE) {
        float x_val = __half2float(input[i]);
        sum_gate += __half2float(w1_row[i]) * x_val;
        sum_up += __half2float(w3_row[i]) * x_val;
    }
    
    // Block-level reduction
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage_gate;
    __shared__ typename BlockReduce::TempStorage temp_storage_up;
    
    sum_gate = BlockReduce(temp_storage_gate).Sum(sum_gate);
    __syncthreads();
    sum_up = BlockReduce(temp_storage_up).Sum(sum_up);
    
    if (tid == 0) {
        float gate_activated = sum_gate / (1.0f + expf(-sum_gate));
        output[row] = __float2half(gate_activated * sum_up);
    }
}

void fused_gate_up_swiglu_kernel_cu_fp16(
    const tensor::Tensor& input,
    const tensor::Tensor& w1,
    const tensor::Tensor& w3,
    const tensor::Tensor& output,
    const CudaConfig* config
) {
    CHECK(!input.is_empty() && input.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(!w1.is_empty() && w1.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(!w3.is_empty() && w3.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(!output.is_empty() && output.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(input.data_type() == base::DataType::kDataTypeFp16);
    CHECK(w1.data_type() == base::DataType::kDataTypeFp16);
    CHECK(w3.data_type() == base::DataType::kDataTypeFp16);
    CHECK(output.data_type() == base::DataType::kDataTypeFp16);
    
    const int32_t K = w1.get_dim(0);
    const int32_t M = w1.get_dim(1);
    
    CHECK_EQ(w3.get_dim(0), K);
    CHECK_EQ(w3.get_dim(1), M);
    CHECK_EQ(M, input.get_dim(0));
    CHECK_EQ(K, output.get_dim(0));
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    // Use optimized v2 kernel with float4 vectorization and warp-level reduction
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
    const int num_blocks = (K + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    
    if (stream) {
        fused_gate_up_swiglu_kernel_fp16_v2<32, WARPS_PER_BLOCK><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const half*>(input.ptr<uint16_t>()),
            reinterpret_cast<const half*>(w1.ptr<uint16_t>()),
            reinterpret_cast<const half*>(w3.ptr<uint16_t>()),
            reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>())),
            M, K);
    } else {
        fused_gate_up_swiglu_kernel_fp16_v2<32, WARPS_PER_BLOCK><<<num_blocks, THREADS_PER_BLOCK>>>(
            reinterpret_cast<const half*>(input.ptr<uint16_t>()),
            reinterpret_cast<const half*>(w1.ptr<uint16_t>()),
            reinterpret_cast<const half*>(w3.ptr<uint16_t>()),
            reinterpret_cast<half*>(const_cast<uint16_t*>(output.ptr<uint16_t>())),
            M, K);
    }
}

void batched_fused_gate_up_swiglu_kernel_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& w1,
    const tensor::Tensor& w3,
    const tensor::Tensor& output,
    int32_t batch_size,
    const CudaConfig* config
) {
    CHECK(!input.is_empty() && input.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(!w1.is_empty() && w1.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(!w3.is_empty() && w3.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(!output.is_empty() && output.device_type() == base::DeviceType::kDeviceCUDA);
    
    const int32_t K = w1.get_dim(0);
    const int32_t M = w1.get_dim(1);
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    constexpr int BLOCK_SIZE = 256;
    constexpr int BATCH_TILE = 4;
    
    dim3 grid(K, (batch_size + BATCH_TILE - 1) / BATCH_TILE);
    
    if (stream) {
        batched_fused_gate_up_swiglu_kernel<BLOCK_SIZE, BATCH_TILE><<<grid, BLOCK_SIZE, 0, stream>>>(
            input.ptr<float>(), w1.ptr<float>(), w3.ptr<float>(),
            const_cast<float*>(output.ptr<float>()), M, K, batch_size);
    } else {
        batched_fused_gate_up_swiglu_kernel<BLOCK_SIZE, BATCH_TILE><<<grid, BLOCK_SIZE>>>(
            input.ptr<float>(), w1.ptr<float>(), w3.ptr<float>(),
            const_cast<float*>(output.ptr<float>()), M, K, batch_size);
    }
}

}  // namespace kernel
