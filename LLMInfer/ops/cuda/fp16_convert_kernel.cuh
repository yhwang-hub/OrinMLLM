#ifndef FP16_CONVERT_KERNEL_CUH
#define FP16_CONVERT_KERNEL_CUH
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "tensor/tensor.h"

namespace kernel {

/**
 * Convert FP16 tensor to FP32 tensor on GPU
 * @param input  FP16 input tensor (already on GPU)
 * @param output FP32 output tensor (already allocated on GPU)
 * @param size   Number of elements
 * @param stream CUDA stream
 */
void fp16_to_fp32_kernel_cu(const half* input, float* output, size_t size, cudaStream_t stream = nullptr);

/**
 * Convert FP16 tensor on CPU to FP32 tensor on GPU
 * This function:
 * 1. Allocates GPU memory for FP16 data
 * 2. Copies FP16 data from CPU to GPU  
 * 3. Converts FP16 to FP32 on GPU
 * @param input_cpu  FP16 input data on CPU
 * @param output     Pre-allocated FP32 output tensor on GPU
 * @param size       Number of elements
 * @param stream     CUDA stream
 */
void fp16_cpu_to_fp32_gpu(const uint16_t* input_cpu, float* output_gpu, size_t size, cudaStream_t stream = nullptr);

}  // namespace kernel

#endif  // FP16_CONVERT_KERNEL_CUH
