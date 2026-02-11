#ifndef SWIGLU_KERNEL_CU_CUH
#define SWIGLU_KERNEL_CU_CUH
#include <tensor/tensor.h>
namespace kernel {
// FP32 SwiGLU kernel
void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream);

// Pure FP16 SwiGLU kernel
void swiglu_kernel_cu_pure_fp16(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& output, void* stream);
}
#endif  // SWIGLU_KERNEL_CU_CUH
