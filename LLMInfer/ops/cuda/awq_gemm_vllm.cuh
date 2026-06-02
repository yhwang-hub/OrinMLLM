#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace kernel {

/**
 * AWQ GEMM using vllm-style implementation
 */
void awq_gemm_vllm_cu(
    const half* input,
    const int32_t* qweight,
    const int32_t* qzeros,
    const half* scales,
    half* output,
    int M,
    int K,
    int N,
    int group_size,
    cudaStream_t stream
);

}  // namespace kernel
