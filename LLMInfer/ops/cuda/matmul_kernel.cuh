#ifndef MATMUL_KERNEL_CU_CUH
#define MATMUL_KERNEL_CU_CUH
#include "../kernels_interface.h"
#include "tensor/tensor.h"
namespace kernel {
void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, float scale = 1.f,
                      const CudaConfig* config = nullptr);

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config = nullptr);

// Batched matmul for prefill phase
void batched_matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                              const tensor::Tensor& output, int32_t batch_size, float scale = 1.f,
                              const CudaConfig* config = nullptr);

// FP16 weight matmul (mixed precision: FP16 weight x FP32 input -> FP32 output)
void matmul_kernel_cu_fp16_weight(const tensor::Tensor& input, const tensor::Tensor& weight,
                                   const tensor::Tensor& output, float scale = 1.f,
                                   const CudaConfig* config = nullptr);

// Batched FP16 weight matmul for prefill phase
void batched_matmul_kernel_cu_fp16_weight(const tensor::Tensor& input, const tensor::Tensor& weight,
                                           const tensor::Tensor& output, int32_t batch_size, 
                                           float scale = 1.f, const CudaConfig* config = nullptr);

// ==================== Pure FP16 Kernels (FP16 input x FP16 weight -> FP16 output) ====================

// Pure FP16 matmul for decode phase (single token GEMV)
// Uses cuBLAS HGEMM with Tensor Core for maximum performance
void matmul_kernel_cu_pure_fp16(const tensor::Tensor& input, const tensor::Tensor& weight,
                                 const tensor::Tensor& output, float scale = 1.f,
                                 const CudaConfig* config = nullptr);

// Batched pure FP16 matmul for prefill phase
// Uses cuBLAS HGEMM batch for optimal Tensor Core utilization
void batched_matmul_kernel_cu_pure_fp16(const tensor::Tensor& input, const tensor::Tensor& weight,
                                         const tensor::Tensor& output, int32_t batch_size, 
                                         float scale = 1.f, const CudaConfig* config = nullptr);

// ==================== Mixed FP16 Input Kernels ====================

// FP16 input × FP16 weight → FP32 output (for cls_logits final layer)
// Converts FP16 input to FP32 internally, then uses FP16 weight kernel
void matmul_kernel_cu_fp16_input_fp16_weight(const tensor::Tensor& input, const tensor::Tensor& weight,
                                              const tensor::Tensor& output, float scale = 1.f,
                                              const CudaConfig* config = nullptr);

}  // namespace kernel

#endif  // MATMUL_KERNEL_CU_CUH
