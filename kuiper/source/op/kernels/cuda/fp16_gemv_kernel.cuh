#ifndef FP16_GEMV_KERNEL_CUH
#define FP16_GEMV_KERNEL_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace kernel {

/**
 * Highly optimized FP16 GEMV kernel
 * FP16 weight x FP16 input -> FP16 output
 * Best for decode phase where input is single vector
 */
void fp16_gemv_kernel_cu(
    const half* input, const half* weight, half* output,
    int M, int K, cudaStream_t stream = nullptr);

/**
 * FP16 GEMV with FP32 output
 * FP16 weight x FP16 input -> FP32 output
 * Use when output needs higher precision
 */
void fp16_gemv_fp32_output_kernel_cu(
    const half* input, const half* weight, float* output,
    int M, int K, cudaStream_t stream = nullptr);

/**
 * FP16 GEMV optimized for large input dimensions
 * Uses shared memory to cache input vector
 */
void fp16_gemv_large_m_kernel_cu(
    const half* input, const half* weight, half* output,
    int M, int K, cudaStream_t stream = nullptr);

/**
 * Mixed precision GEMV: FP16 weight x FP32 input -> FP32 output
 * Use when activations are kept in FP32
 */
void fp16_weight_fp32_io_gemv_kernel_cu(
    const float* input, const half* weight, float* output,
    int M, int K, cudaStream_t stream = nullptr);

}  // namespace kernel

#endif  // FP16_GEMV_KERNEL_CUH
