#ifndef ARGMAX_KERNEL_CUH
#define ARGMAX_KERNEL_CUH
#include <cstdint>
namespace kernel {
size_t argmax_kernel_cu(const float* input_ptr, size_t size, void* stream);

// Optimized version that uses pre-allocated buffers to avoid per-call allocation
void argmax_kernel_cu_prealloc(const float* input_ptr, size_t size, 
                                size_t* output_gpu, size_t* output_pinned,
                                void* stream);

// Batched FP16 argmax: runs argmax on each row of [batch, row_size] FP16 input
// output_gpu: pre-allocated GPU buffer [batch] int32
// output_cpu: host buffer [batch] int32 (receives async D2H copy)
void batched_argmax_fp16_cu(const void* input, int32_t* output_gpu, int32_t* output_cpu,
                            int32_t batch, int32_t row_size, void* stream = nullptr);
}
#endif  // ARGMAX_KERNEL_CUH
