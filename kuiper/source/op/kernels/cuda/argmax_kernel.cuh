#ifndef ARGMAX_KERNEL_CUH
#define ARGMAX_KERNEL_CUH
namespace kernel {
size_t argmax_kernel_cu(const float* input_ptr, size_t size, void* stream);

// Optimized version that uses pre-allocated buffers to avoid per-call allocation
// output_gpu: pre-allocated GPU buffer for argmax result
// output_pinned: pre-allocated pinned memory buffer for async D2H transfer
void argmax_kernel_cu_prealloc(const float* input_ptr, size_t size, 
                                size_t* output_gpu, size_t* output_pinned,
                                void* stream);
}
#endif  // ARGMAX_KERNEL_CUH
