#ifndef FLASH_ATTENTION_KERNEL_H
#define FLASH_ATTENTION_KERNEL_H

#include <base/cuda_config.h>
#include <tensor/tensor.h>

namespace kernel {

/**
 * Flash Attention for batched prefill phase (FP32)
 * 
 * @param start_pos Starting position in KV cache
 * @param seq_len Number of query tokens to process
 * @param head_num Number of attention heads
 * @param kv_head_num Number of KV heads (for GQA)
 * @param head_size Dimension per head
 * @param kv_mul head_num / kv_head_num
 * @param layer_index Current layer index
 * @param max_seq_len Maximum sequence length
 * @param kv_dim KV dimension (kv_head_num * head_size)
 * @param query Query tensor [seq_len, dim]
 * @param output Output tensor [seq_len, dim]
 * @param key_cache KV cache for keys [layer_num, max_seq_len, kv_dim]
 * @param value_cache KV cache for values [layer_num, max_seq_len, kv_dim]
 * @param config CUDA configuration
 */
void flash_attention_prefill_cu(
    int32_t start_pos,
    int32_t seq_len,
    int32_t head_num,
    int32_t kv_head_num,
    int32_t head_size,
    int32_t kv_mul,
    int32_t layer_index,
    int32_t max_seq_len,
    int32_t kv_dim,
    const tensor::Tensor& query,
    const tensor::Tensor& output,
    const tensor::Tensor& key_cache,
    const tensor::Tensor& value_cache,
    CudaConfig* config
);

/**
 * Flash Attention for single token decode phase (FP32)
 */
void flash_attention_decode_cu(
    int32_t pos,
    int32_t head_num,
    int32_t kv_head_num,
    int32_t head_size,
    int32_t kv_mul,
    int32_t layer_index,
    int32_t max_seq_len,
    int32_t kv_dim,
    const tensor::Tensor& query,
    const tensor::Tensor& output,
    const tensor::Tensor& key_cache,
    const tensor::Tensor& value_cache,
    CudaConfig* config
);

// ============================================================================
// FP16 Flash Attention Functions - Optimized with half2 vectorization
// ============================================================================

/**
 * Flash Attention for batched prefill phase (FP16)
 * Uses half2 vectorized dot products for improved throughput
 */
void flash_attention_prefill_fp16_cu(
    int32_t start_pos,
    int32_t seq_len,
    int32_t head_num,
    int32_t kv_head_num,
    int32_t head_size,
    int32_t kv_mul,
    int32_t layer_index,
    int32_t max_seq_len,
    int32_t kv_dim,
    const tensor::Tensor& query,
    const tensor::Tensor& output,
    const tensor::Tensor& key_cache,
    const tensor::Tensor& value_cache,
    CudaConfig* config
);

/**
 * Flash Attention for single token decode phase (FP16)
 * Ultra-optimized with 8 warps, half2 dot products, and register blocking
 */
void flash_attention_decode_fp16_cu(
    int32_t pos,
    int32_t head_num,
    int32_t kv_head_num,
    int32_t head_size,
    int32_t kv_mul,
    int32_t layer_index,
    int32_t max_seq_len,
    int32_t kv_dim,
    const tensor::Tensor& query,
    const tensor::Tensor& output,
    const tensor::Tensor& key_cache,
    const tensor::Tensor& value_cache,
    CudaConfig* config
);

/**
 * Flash Attention for single token decode phase (FP16) with GPU pos pointer
 * Compatible with CUDA Graph - reads position from GPU memory
 */
void flash_attention_decode_fp16_gpu_pos_cu(
    const int32_t* pos_ptr,  // GPU memory pointer to position
    int32_t head_num,
    int32_t kv_head_num,
    int32_t head_size,
    int32_t kv_mul,
    int32_t layer_index,
    int32_t max_seq_len,
    int32_t kv_dim,
    const tensor::Tensor& query,
    const tensor::Tensor& output,
    const tensor::Tensor& key_cache,
    const tensor::Tensor& value_cache,
    CudaConfig* config
);

}  // namespace kernel

#endif  // FLASH_ATTENTION_KERNEL_H
