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

// Batched argmax: compute argmax for multiple rows in parallel
// input_ptr: [batch_size, vocab_size] logits
// output_gpu: [batch_size] argmax indices on GPU
// batch_size: number of rows
// vocab_size: number of elements per row
void batched_argmax_kernel_cu(const float* input_ptr, int32_t* output_gpu,
                               int32_t batch_size, int32_t vocab_size, void* stream);

// Single row argmax on GPU - output stays on GPU for later batch copy
// input_ptr: [vocab_size] logits on GPU
// output_gpu: single int32_t* on GPU
// vocab_size: number of elements
void single_argmax_kernel_cu(const float* input_ptr, int32_t* output_gpu,
                              int32_t vocab_size, void* stream);

// Argmax + D2T mapping kernel: computes argmax and applies D2T offset mapping
// input_ptr: [vocab_size] logits on GPU
// d2t_gpu: [vocab_size] D2T offset table on GPU (target = draft_idx + d2t[draft_idx])
// output_gpu: single int32_t* on GPU (target token)
// vocab_size: number of elements
void argmax_d2t_kernel_cu(const float* input_ptr, const int32_t* d2t_gpu,
                           int32_t* output_gpu, int32_t vocab_size, void* stream);

// Top-K + D2T mapping kernel: computes top-k indices and applies D2T offset mapping
// input_ptr: [vocab_size] logits on GPU
// d2t_gpu: [vocab_size] D2T offset table on GPU (target = draft_idx + d2t[draft_idx])
// output_gpu: [k] int32_t* on GPU (top-k target tokens)
// vocab_size: number of elements
// k: number of top candidates
void topk_d2t_kernel_cu(const float* input_ptr, const int32_t* d2t_gpu,
                         int32_t* output_gpu, int32_t vocab_size, int32_t k, void* stream);
}
#endif  // ARGMAX_KERNEL_CUH
