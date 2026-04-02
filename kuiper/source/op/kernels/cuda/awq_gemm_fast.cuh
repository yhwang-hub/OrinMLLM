/*
 * AWQ Fast W4A16 GEMV Kernel - M=1 Decode Optimized
 *
 * Uses LOP3-based INT4 to FP16 conversion for fast dequantization.
 * Designed exclusively for decode phase where M=1 (single token generation).
 *
 * For M>1 (prefill), use awq_gemm_vllm_cu (Tensor Core MMA path).
 */

#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace kernel {

/**
 * AWQ GEMV with coalesced memory access using transposed qweight layout.
 *
 * qweight_t is [N/8, K] (transposed from [K, N/8]), so 32 warp lanes
 * access consecutive int32 addresses (stride=4 bytes) instead of
 * strided addresses (stride=packed_N*4 bytes), achieving perfect coalescing.
 */
void awq_gemv_coalesced_cu(
    const half* input,
    const int32_t* qweight_t,  // [N/8, K] transposed layout
    const int32_t* qzeros,
    const half* scales,
    half* output,
    int K,
    int N,
    int group_size,
    cudaStream_t stream
);

/**
 * Transpose qweight from [K, N/8] to [N/8, K] for coalesced decode access.
 * Called once during model initialization (to_cuda).
 */
void awq_transpose_qweight_cu(
    const int32_t* src,   // [K, packed_N]
    int32_t* dst,         // [packed_N, K]
    int K,
    int packed_N,
    cudaStream_t stream
);

/**
 * Fused AWQ Gate+Up+SwiGLU GEMV for decode (M=1).
 * Computes: output = SiLU(W1*x) * (W3*x) in a single kernel,
 * avoiding intermediate buffer writes/reads and reducing kernel launches.
 */
void awq_fused_gate_up_swiglu_cu(
    const half* input,              // [K]
    const int32_t* w1_qweight_t,    // [hidden_dim/8, K] transposed
    const int32_t* w1_qzeros,       // [K/G, hidden_dim/8]
    const half* w1_scales,          // [K/G, hidden_dim]
    const int32_t* w3_qweight_t,    // [hidden_dim/8, K] transposed
    const int32_t* w3_qzeros,       // [K/G, hidden_dim/8]
    const half* w3_scales,          // [K/G, hidden_dim]
    half* output,                   // [hidden_dim]
    int K,
    int hidden_dim,
    int group_size,
    cudaStream_t stream
);

/**
 * Fused AWQ Q+K+V projection GEMV for decode (M=1).
 * Merges 3 separate AWQ GEMV calls into a single kernel launch.
 * Blocks are assigned to Q, K, or V based on their block index.
 */
void awq_fused_qkv_cu(
    const half* input,            // [K] shared input
    const int32_t* q_qweight_t,   // [q_N/8, K]
    const int32_t* q_qzeros,      const half* q_scales,
    half* q_output,               int q_N,
    const int32_t* k_qweight_t,   // [k_N/8, K]
    const int32_t* k_qzeros,      const half* k_scales,
    half* k_output,               int k_N,
    const int32_t* v_qweight_t,   // [v_N/8, K]
    const int32_t* v_qzeros,      const half* v_scales,
    half* v_output,               int v_N,
    int K,
    int group_size,
    cudaStream_t stream
);

}  // namespace kernel
