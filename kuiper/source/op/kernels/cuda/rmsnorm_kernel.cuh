#ifndef RMSNORM_KERNEL_CU_CUH
#define RMSNORM_KERNEL_CU_CUH
#include <tensor/tensor.h>
namespace kernel {
// Standard RMSNorm (FP32 input, FP32 or FP16 weight -> FP32 output)
void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, void* stream = nullptr);

// Batched RMSNorm for multi-row input
void rmsnorm_kernel_cu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, int32_t dim, void* stream = nullptr);

// Pure FP16 RMSNorm (FP16 input x FP16 weight -> FP16 output)
void rmsnorm_kernel_cu_pure_fp16(const tensor::Tensor& input, const tensor::Tensor& weight,
                                  const tensor::Tensor& output, void* stream = nullptr);

// Batched Pure FP16 RMSNorm for multi-row input
void rmsnorm_kernel_cu_pure_fp16_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                                      const tensor::Tensor& output, int32_t dim, void* stream = nullptr);
}  // namespace kernel
#endif  // RMSNORM_KERNEL_CU_CUH
