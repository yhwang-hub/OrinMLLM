#ifndef EMB_KERNEL_H
#define EMB_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
// Standard embedding: auto-detects weight type, always outputs FP32
void emb_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                   const tensor::Tensor& output, int32_t vocab_size, void* stream = nullptr);

// Pure FP16 embedding: FP16 weight -> FP16 output (no conversion)
void emb_kernel_cu_pure_fp16(const tensor::Tensor& input, const tensor::Tensor& weight,
                              const tensor::Tensor& output, int32_t vocab_size, 
                              void* stream = nullptr);
}
#endif  // EMB_KERNEL_H
