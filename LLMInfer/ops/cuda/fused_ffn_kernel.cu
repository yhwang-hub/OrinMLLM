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
    
    // Main loop: process 4 elements at a time with __ldg + fmaf
    #pragma unroll 4
    for (int i = tid; i < num_vecs; i += BLOCK_SIZE) {
        float4 x = __ldg(input_vec + i);
        float4 g = __ldg(w1_vec + i);  // gate weight
        float4 u = __ldg(w3_vec + i);  // up weight
        
        // Fused multiply-add for both projections
        sum_gate = fmaf(g.x, x.x, fmaf(g.y, x.y, fmaf(g.z, x.z, fmaf(g.w, x.w, sum_gate))));
        sum_up = fmaf(u.x, x.x, fmaf(u.y, x.y, fmaf(u.z, x.z, fmaf(u.w, x.w, sum_up))));
    }
    
    // Handle remainder (if M is not divisible by 4)
    const int base = num_vecs * vec_size;
    for (int i = base + tid; i < M; i += BLOCK_SIZE) {
        float x_val = __ldg(input + i);
        sum_gate = fmaf(__ldg(w1_row + i), x_val, sum_gate);
        sum_up = fmaf(__ldg(w3_row + i), x_val, sum_up);
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
    
    // float4 for input (128-bit), float4 reinterpret as half2x4 for weights (128-bit)
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    const float4* w1_f4 = reinterpret_cast<const float4*>(w1_row);
    const float4* w3_f4 = reinterpret_cast<const float4*>(w3_row);
    const int num_f4_w = M / 8;  // 8 halfs per float4
    
    #pragma unroll 4
    for (int i = tid; i < num_f4_w; i += BLOCK_SIZE) {
        // Load 2 float4s from input (8 floats) in 2 iterations
        float4 x0 = __ldg(input_vec + i * 2);
        float4 x1 = __ldg(input_vec + i * 2 + 1);
        
        // Load 8 halfs as float4 via __ldg (128-bit)
        float4 g_packed = __ldg(w1_f4 + i);
        float4 u_packed = __ldg(w3_f4 + i);
        const half2* g_h2 = reinterpret_cast<const half2*>(&g_packed);
        const half2* u_h2 = reinterpret_cast<const half2*>(&u_packed);
        
        // Convert and accumulate with fmaf
        sum_gate = fmaf(__half2float(g_h2[0].x), x0.x, fmaf(__half2float(g_h2[0].y), x0.y,
                  fmaf(__half2float(g_h2[1].x), x0.z, fmaf(__half2float(g_h2[1].y), x0.w,
                  fmaf(__half2float(g_h2[2].x), x1.x, fmaf(__half2float(g_h2[2].y), x1.y,
                  fmaf(__half2float(g_h2[3].x), x1.z, fmaf(__half2float(g_h2[3].y), x1.w, sum_gate))))))));
        sum_up = fmaf(__half2float(u_h2[0].x), x0.x, fmaf(__half2float(u_h2[0].y), x0.y,
                 fmaf(__half2float(u_h2[1].x), x0.z, fmaf(__half2float(u_h2[1].y), x0.w,
                 fmaf(__half2float(u_h2[2].x), x1.x, fmaf(__half2float(u_h2[2].y), x1.y,
                 fmaf(__half2float(u_h2[3].x), x1.z, fmaf(__half2float(u_h2[3].y), x1.w, sum_up))))))));
    }
    
    // Handle remainder
    const int base = num_f4_w * 8;
    for (int i = base + tid; i < M; i += BLOCK_SIZE) {
        float x_val = __ldg(input + i);
        sum_gate = fmaf(__half2float(__ldg(w1_row + i)), x_val, sum_gate);
        sum_up = fmaf(__half2float(__ldg(w3_row + i)), x_val, sum_up);
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
        float4 x_f4 = __ldg(input_f4 + i);
        float4 g_f4 = __ldg(w1_f4 + i);
        float4 u_f4 = __ldg(w3_f4 + i);
        
        // Extract half2 pairs and accumulate with fmaf - no branches
        const half2* x_h2 = reinterpret_cast<const half2*>(&x_f4);
        const half2* g_h2 = reinterpret_cast<const half2*>(&g_f4);
        const half2* u_h2 = reinterpret_cast<const half2*>(&u_f4);
        
        float2 xf0 = __half22float2(x_h2[0]);
        float2 gf0 = __half22float2(g_h2[0]);
        float2 uf0 = __half22float2(u_h2[0]);
        sum_gate0 = fmaf(gf0.x, xf0.x, fmaf(gf0.y, xf0.y, sum_gate0));
        sum_up0 = fmaf(uf0.x, xf0.x, fmaf(uf0.y, xf0.y, sum_up0));
        
        float2 xf1 = __half22float2(x_h2[1]);
        float2 gf1 = __half22float2(g_h2[1]);
        float2 uf1 = __half22float2(u_h2[1]);
        sum_gate1 = fmaf(gf1.x, xf1.x, fmaf(gf1.y, xf1.y, sum_gate1));
        sum_up1 = fmaf(uf1.x, xf1.x, fmaf(uf1.y, xf1.y, sum_up1));
        
        float2 xf2 = __half22float2(x_h2[2]);
        float2 gf2 = __half22float2(g_h2[2]);
        float2 uf2 = __half22float2(u_h2[2]);
        sum_gate2 = fmaf(gf2.x, xf2.x, fmaf(gf2.y, xf2.y, sum_gate2));
        sum_up2 = fmaf(uf2.x, xf2.x, fmaf(uf2.y, xf2.y, sum_up2));
        
        float2 xf3 = __half22float2(x_h2[3]);
        float2 gf3 = __half22float2(g_h2[3]);
        float2 uf3 = __half22float2(u_h2[3]);
        sum_gate3 = fmaf(gf3.x, xf3.x, fmaf(gf3.y, xf3.y, sum_gate3));
        sum_up3 = fmaf(uf3.x, xf3.x, fmaf(uf3.y, xf3.y, sum_up3));
    }
    
    // Merge accumulators
    float sum_gate = sum_gate0 + sum_gate1 + sum_gate2 + sum_gate3;
    float sum_up = sum_up0 + sum_up1 + sum_up2 + sum_up3;
    
    // Handle remainder
    const int base = num_float4 * 8;
    for (int i = base + lane_id; i < M; i += WARP_SIZE) {
        float x_val = __half2float(__ldg(input + i));
        sum_gate = fmaf(__half2float(__ldg(w1_row + i)), x_val, sum_gate);
        sum_up = fmaf(__half2float(__ldg(w3_row + i)), x_val, sum_up);
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

}  // namespace kernel
