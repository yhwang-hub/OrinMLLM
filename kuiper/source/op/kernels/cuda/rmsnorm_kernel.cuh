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

// FP32 → FP16 conversion with clamping to prevent overflow to INF
void fp32_to_fp16_clamp_cu(const float* in, void* out, int n, void* stream = nullptr);
}  // namespace kernel
#endif  // RMSNORM_KERNEL_CU_CUH
