#ifndef FUSED_ROPE_KV_KERNEL_CUH_
#define FUSED_ROPE_KV_KERNEL_CUH_
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace kernel {

/**
 * @brief Fused M-RoPE + KV Cache Write kernel for GQA decode phase (FP16)
 *
 * Fuses three separate operations into a single kernel launch:
 *   1. Apply M-RoPE to Q tensor
 *   2. Apply M-RoPE to K tensor and write to key_cache
 *   3. Write V tensor to val_cache
 *
 * This eliminates 2 extra kernel launches and reduces global memory traffic:
 * - K data is read once and written directly to cache (instead of read→write→read→write)
 * - V data is read once and written directly to cache (instead of read→write→read→write)
 *
 * For decode phase of Qwen3-VL, all M-RoPE positions are identical (text_pos).
 *
 * @param pos_gpu         GPU pointer to position value (M-RoPE text position, also KV cache position index)
 * @param kv_cache_pos_gpu GPU pointer to KV cache write position
 * @param query           Q tensor [dim] (FP16) — input and output (in-place)
 * @param key             K tensor [kv_dim] (FP16) — input only, RoPE result goes to cache
 * @param value           V tensor [kv_dim] (FP16) — input only, goes to cache directly
 * @param key_cache       Key cache [layer_num, seq_len, kv_dim] (FP16)
 * @param val_cache       Value cache [layer_num, seq_len, kv_dim] (FP16)
 * @param sin_cache       Sin cache [max_pos, head_size] (FP32)
 * @param cos_cache       Cos cache [max_pos, head_size] (FP32)
 * @param dim             Q total dimension (head_num * head_size)
 * @param kv_dim          KV total dimension (kv_head_num * head_size)
 * @param head_size       Per-head dimension (128 for Qwen3-VL)
 * @param section0        M-RoPE section 0 pair count (temporal, 24)
 * @param section1        M-RoPE section 1 pair count (height, 20)
 * @param section2        M-RoPE section 2 pair count (width, 20)
 * @param layer_idx       Current transformer layer index
 * @param seq_len         Maximum sequence length for KV cache
 * @param stream          CUDA stream
 */
void fused_mrope_kv_write_fp16(
    const int32_t* pos_gpu,
    const int32_t* kv_cache_pos_gpu,
    half* query,
    const half* key,
    const half* value,
    half* key_cache,
    half* val_cache,
    const float* sin_cache,
    const float* cos_cache,
    int32_t dim,
    int32_t kv_dim,
    int32_t head_size,
    int32_t section0,
    int32_t section1,
    int32_t section2,
    int32_t layer_idx,
    int32_t seq_len,
    cudaStream_t stream);

/**
 * @brief Fused GQA + M-RoPE + KV Cache Read/Write decode kernel (FP16)
 *
 * Fuses FIVE operations into a single kernel launch:
 *   1. Apply M-RoPE to Q (result stays in shared memory, no global write-back)
 *   2. Apply M-RoPE to K for current token
 *   3. Write RoPE'd K and raw V to KV cache at current position
 *   4. GQA Flash Attention decode (Q·K scoring + online softmax + V accumulation)
 *   5. Write attention output
 *
 * This eliminates global memory round-trips for Q (save 1 read + 1 write per layer),
 * removes 2 kernel launch overheads, and reduces total kernel count from 3 to 1
 * for the MRoPE + KV write + attention portion of each transformer layer.
 *
 * Grid: (num_q_heads), Block: 256 threads (8 warps)
 *
 * @param pos_gpu         GPU pointer to M-RoPE text position
 * @param kv_pos_gpu      GPU pointer to KV cache write position
 * @param query           Q tensor [dim] (FP16, read-only — MRoPE applied in shared memory)
 * @param key             K tensor [kv_dim] (FP16, read-only)
 * @param value           V tensor [kv_dim] (FP16, read-only)
 * @param key_cache       Key cache [layer_num, max_seq_len, kv_dim] (FP16)
 * @param val_cache       Value cache [layer_num, max_seq_len, kv_dim] (FP16)
 * @param output          Attention output [dim] (FP16)
 * @param sin_cache       Sin cache [max_pos, head_size] (FP32)
 * @param cos_cache       Cos cache [max_pos, head_size] (FP32)
 * @param dim             Q total dimension
 * @param kv_dim          KV total dimension
 * @param head_size       Per-head dimension
 * @param section0        M-RoPE temporal section pairs
 * @param section1        M-RoPE height section pairs
 * @param section2        M-RoPE width section pairs
 * @param layer_idx       Current transformer layer index
 * @param max_seq_len     Maximum sequence length for KV cache
 * @param stream          CUDA stream
 */
void fused_gqa_mrope_kv_decode_fp16(
    const int32_t* pos_gpu,
    const int32_t* kv_pos_gpu,
    const half* query,
    const half* key,
    const half* value,
    half* key_cache,
    half* val_cache,
    half* output,
    const float* sin_cache,
    const float* cos_cache,
    int32_t dim,
    int32_t kv_dim,
    int32_t head_size,
    int32_t section0,
    int32_t section1,
    int32_t section2,
    int32_t layer_idx,
    int32_t max_seq_len,
    cudaStream_t stream);

}  // namespace kernel

#endif  // FUSED_ROPE_KV_KERNEL_CUH_
