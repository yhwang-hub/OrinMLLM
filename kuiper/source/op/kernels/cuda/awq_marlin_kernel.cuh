/**
 * AWQ Marlin-Style High-Performance W4A16 GEMM Kernel
 * 
 * This kernel implements Marlin-style optimizations for AWQ:
 * 1. Register-level dequantization using LOP3 instructions
 * 2. Tensor Core MMA (m16n8k16) for compute
 * 3. Software pipelining with async copy
 * 4. Bank-conflict-free shared memory layout
 * 
 * Key difference from standard Marlin:
 * - Supports AWQ's original weight format (no repacking required)
 * - Handles AWQ's reverse packing order {0,4,1,5,2,6,3,7}
 * - Supports group quantization with zero-points
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace kernel {

// Forward declarations
void awq_marlin_gemm_cu(
    const half* input,           // [M, K]
    const int32_t* qweight,      // [K, N/8] - AWQ packed INT4
    const int32_t* qzeros,       // [K/G, N/8] - AWQ packed zeros
    const half* scales,          // [K/G, N] - FP16 scales
    half* output,                // [M, N]
    int M,
    int K,
    int N,
    int group_size,
    cudaStream_t stream
);

// Repack AWQ weights to Marlin format (optional, for best performance)
void awq_repack_weights_cu(
    const int32_t* qweight_awq,  // [K, N/8] - AWQ format
    const int32_t* qzeros_awq,   // [K/G, N/8] - AWQ format  
    int32_t* qweight_marlin,     // [K, N/8] - Marlin format
    int32_t* qzeros_marlin,      // [K/G, N/8] - Marlin format
    int K,
    int N,
    int group_size,
    cudaStream_t stream
);

// Initialize resources (call once at startup)
void awq_marlin_init();

// Cleanup resources
void awq_marlin_cleanup();

}  // namespace kernel
