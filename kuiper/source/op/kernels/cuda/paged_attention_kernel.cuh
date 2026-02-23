#ifndef PAGED_ATTENTION_KERNEL_CUH
#define PAGED_ATTENTION_KERNEL_CUH

#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cuda_fp16.h>

namespace kernel {

// ============================================================================
// Paged Flash Attention Decode Kernels
// ============================================================================

/**
 * Paged Flash Attention FP16 decode (CPU position, 256 threads)
 */
void paged_flash_attention_decode_fp16_cu(
    int32_t pos, int32_t head_num, int32_t kv_head_num,
    int32_t head_size, int32_t kv_mul, int32_t layer_index,
    int32_t kv_dim, int32_t page_size, int32_t max_blocks_per_seq,
    const tensor::Tensor& query, const tensor::Tensor& output,
    const void* key_pool, const void* value_pool, const int32_t* block_table,
    CudaConfig* config);

/**
 * Paged Flash Attention FP32 decode (CPU position, 256 threads)
 */
void paged_flash_attention_decode_cu(
    int32_t pos, int32_t head_num, int32_t kv_head_num,
    int32_t head_size, int32_t kv_mul, int32_t layer_index,
    int32_t kv_dim, int32_t page_size, int32_t max_blocks_per_seq,
    const tensor::Tensor& query, const tensor::Tensor& output,
    const void* key_pool, const void* value_pool, const int32_t* block_table,
    CudaConfig* config);

// ============================================================================
// Paged Flash Attention Prefill Kernels
// ============================================================================

/**
 * Paged Flash Attention FP16 prefill (128 threads, online softmax)
 */
void paged_flash_attention_prefill_fp16_cu(
    int32_t start_pos, int32_t seq_len,
    int32_t head_num, int32_t kv_head_num,
    int32_t head_size, int32_t kv_mul, int32_t layer_index,
    int32_t kv_dim, int32_t page_size, int32_t max_blocks_per_seq,
    const tensor::Tensor& query, const tensor::Tensor& output,
    const void* key_pool, const void* value_pool, const int32_t* block_table,
    CudaConfig* config);

/**
 * Paged Flash Attention FP32 prefill (256 threads, CUB reduction)
 */
void paged_flash_attention_prefill_cu(
    int32_t start_pos, int32_t seq_len,
    int32_t head_num, int32_t kv_head_num,
    int32_t head_size, int32_t kv_mul, int32_t layer_index,
    int32_t kv_dim, int32_t page_size, int32_t max_blocks_per_seq,
    const tensor::Tensor& query, const tensor::Tensor& output,
    const void* key_pool, const void* value_pool, const int32_t* block_table,
    CudaConfig* config);

// ============================================================================
// Paged Flash Attention with GPU Position (CUDA Graph compatible)
// ============================================================================

/**
 * Paged Flash Attention FP16 decode with GPU position pointer
 * Uses online softmax with fixed shared memory for CUDA Graph
 */
void paged_flash_attention_decode_fp16_gpu_pos_cu(
    const int32_t* pos_ptr, int32_t head_num, int32_t kv_head_num,
    int32_t head_size, int32_t kv_mul, int32_t layer_index,
    int32_t kv_dim, int32_t page_size, int32_t max_blocks_per_seq,
    const tensor::Tensor& query, const tensor::Tensor& output,
    const void* key_pool, const void* value_pool, const int32_t* block_table,
    CudaConfig* config);

// ============================================================================
// Paged KV Cache Write Kernels (CUDA Graph compatible)
// ============================================================================

void paged_copy_to_kv_cache_kernel(float* kv_pool, const float* src, const int32_t* pos,
    const int32_t* block_table, int32_t kv_dim, int32_t layer_idx,
    int32_t max_blocks_per_seq, int32_t page_size, cudaStream_t stream);

void paged_copy_to_kv_cache_kernel_fp16(half* kv_pool, const half* src, const int32_t* pos,
    const int32_t* block_table, int32_t kv_dim, int32_t layer_idx,
    int32_t max_blocks_per_seq, int32_t page_size, cudaStream_t stream);

}  // namespace kernel

#endif  // PAGED_ATTENTION_KERNEL_CUH
