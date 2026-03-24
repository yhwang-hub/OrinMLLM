#ifndef KUIPER_INCLUDE_OP_MISC_LAYERS_H_
#define KUIPER_INCLUDE_OP_MISC_LAYERS_H_
#include "layer.h"

namespace op {

/**
 * @brief SinCosCacheLayer: Compute sin/cos cache for RoPE embeddings
 */
class SinCosCacheLayer : public Layer {
 public:
  explicit SinCosCacheLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward: compute sin/cos cache
  base::Status forward(int32_t head_size, int32_t seq_len,
                       const tensor::Tensor& sin_cache, 
                       const tensor::Tensor& cos_cache);
};

/**
 * @brief MHAGpuPosLayer: Multi-head attention with GPU position tensor
 * Used for CUDA Graph compatible decode path
 */
class MHAGpuPosLayer : public Layer {
 public:
  explicit MHAGpuPosLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward for MHA with GPU position
  base::Status forward(const int32_t* pos_ptr, int32_t head_num, int32_t layer_idx,
                       int32_t seq_len, int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                       tensor::Tensor& mha_output, const tensor::Tensor& query,
                       tensor::Tensor& score_storage, const tensor::Tensor& key_cache,
                       const tensor::Tensor& val_cache);
};

/**
 * @brief BatchedMHALayer: Batched multi-head attention for prefill
 */
class BatchedMHALayer : public Layer {
 public:
  explicit BatchedMHALayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward for batched MHA
  base::Status forward(int32_t start_pos, int32_t seq_len, int32_t head_num,
                       int32_t layer_idx, int32_t max_seq_len, int32_t dim,
                       int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                       tensor::Tensor& mha_output, const tensor::Tensor& query,
                       tensor::Tensor& score_storage, const tensor::Tensor& key_cache,
                       const tensor::Tensor& val_cache);
};

/**
 * @brief BatchedMatmulHelperLayer: Helper for batched matmul with flexible weight input
 * Unlike BatchedMatmulLayer (LayerParam), this takes weight as forward parameter
 */
class BatchedMatmulHelperLayer : public Layer {
 public:
  explicit BatchedMatmulHelperLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward with explicit weight tensor
  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, int32_t batch_size, float scale);
};

/**
 * @brief MRoPELayer: Multi-dimensional Rotary Position Embedding for VL models
 * Used for Qwen3-VL with separate temporal/height/width positions
 */
class MRoPELayer : public Layer {
 public:
  explicit MRoPELayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward for M-RoPE (CPU position)
  base::Status forward(int32_t pos_t, int32_t pos_h, int32_t pos_w,
                       int32_t dim, int32_t kv_dim, int32_t head_size,
                       int32_t section0, int32_t section1, int32_t section2,
                       const tensor::Tensor& query, const tensor::Tensor& key,
                       const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache);
};

/**
 * @brief MRoPEGpuPosLayer: M-RoPE with GPU position tensor for CUDA Graph
 */
class MRoPEGpuPosLayer : public Layer {
 public:
  explicit MRoPEGpuPosLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward for M-RoPE with GPU position
  base::Status forward(const int32_t* rope_pos_gpu,
                       int32_t dim, int32_t kv_dim, int32_t head_size,
                       int32_t section0, int32_t section1, int32_t section2,
                       const tensor::Tensor& query, const tensor::Tensor& key,
                       const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache);
};

/**
 * @brief BatchedMRoPELayer: Batched M-RoPE for prefill phase
 */
class BatchedMRoPELayer : public Layer {
 public:
  explicit BatchedMRoPELayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward for batched M-RoPE
  base::Status forward(int32_t seq_len, int32_t dim, int32_t kv_dim, int32_t head_size,
                       int32_t section0, int32_t section1, int32_t section2,
                       const int32_t* pos_t, const int32_t* pos_h, const int32_t* pos_w,
                       const tensor::Tensor& query, const tensor::Tensor& key,
                       const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache);
};

/**
 * @brief FusedKVCacheUpdateLayer: Fused update for both K and V caches
 */
class FusedKVCacheUpdateLayer : public Layer {
 public:
  explicit FusedKVCacheUpdateLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward for fused KV cache update
  base::Status forward(const tensor::Tensor& key, const tensor::Tensor& value,
                       const tensor::Tensor& key_cache, const tensor::Tensor& val_cache,
                       int32_t layer_idx, int32_t start_pos, int32_t seq_len,
                       int32_t kv_dim, int32_t max_seq_len);
};

/**
 * @brief RMSNormDimLayer: RMSNorm applied per head dimension
 */
class RMSNormDimLayer : public Layer {
 public:
  explicit RMSNormDimLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward for per-dimension RMSNorm
  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, int32_t dim);
};

/**
 * @brief CopyToKVCacheLayer: Copy key/value to KV cache with FP16 support
 */
class CopyToKVCacheLayer : public Layer {
 public:
  explicit CopyToKVCacheLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward for KV cache copy
  base::Status forward(const tensor::Tensor& kv_cache, const tensor::Tensor& kv_data,
                       const int32_t* pos_gpu, int32_t kv_dim, int32_t layer_idx,
                       int32_t seq_len);
};

/**
 * @brief FlashAttentionDecodeGpuPosLayer: Flash Attention decode with GPU position
 */
class FlashAttentionDecodeGpuPosLayer : public Layer {
 public:
  explicit FlashAttentionDecodeGpuPosLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  void set_attention_type(base::AttentionType type) { attention_type_ = type; }
  base::AttentionType get_attention_type() const { return attention_type_; }

  // Direct forward for Flash Attention decode with GPU position
  base::Status forward(const int32_t* pos_gpu, int32_t head_num, int32_t kv_head_num,
                       int32_t head_size, int32_t kv_mul, int32_t layer_idx,
                       int32_t seq_len, int32_t kv_dim,
                       const tensor::Tensor& query, const tensor::Tensor& mha_output,
                       const tensor::Tensor& key_cache, const tensor::Tensor& val_cache);

 private:
  base::AttentionType attention_type_ = base::AttentionType::kAttentionFlash1;
};

/**
 * @brief FusedMRoPEKVWriteLayer: Fused M-RoPE + KV Cache Write for GQA decode
 *
 * Fuses three separate kernel launches into one:
 *   1. Apply M-RoPE to Q (in-place)
 *   2. Apply M-RoPE to K → write directly to key_cache
 *   3. Copy V → write directly to val_cache
 *
 * This reduces kernel launch overhead and global memory traffic.
 * Only used in decode phase where positions are GPU-resident (CUDA Graph compatible).
 */
class FusedMRoPEKVWriteLayer : public Layer {
 public:
  explicit FusedMRoPEKVWriteLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;

  /**
   * @brief Execute fused M-RoPE + KV cache write
   *
   * @param rope_pos_gpu    GPU pointer to M-RoPE text position
   * @param kv_cache_pos_gpu GPU pointer to KV cache write position
   * @param query           Q tensor [dim] (FP16, in-place RoPE)
   * @param key             K tensor [kv_dim] (FP16, input only)
   * @param value           V tensor [kv_dim] (FP16, input only)
   * @param key_cache       Key cache [layer_num, seq_len, kv_dim] (FP16)
   * @param val_cache       Value cache [layer_num, seq_len, kv_dim] (FP16)
   * @param sin_cache       Sin cache (FP32)
   * @param cos_cache       Cos cache (FP32)
   * @param dim             Q total dimension
   * @param kv_dim          KV total dimension
   * @param head_size       Per-head dimension
   * @param section0        M-RoPE temporal section pairs
   * @param section1        M-RoPE height section pairs
   * @param section2        M-RoPE width section pairs
   * @param layer_idx       Current layer index
   * @param seq_len         Max sequence length
   */
  base::Status forward(const int32_t* rope_pos_gpu, const int32_t* kv_cache_pos_gpu,
                       const tensor::Tensor& query, const tensor::Tensor& key,
                       const tensor::Tensor& value,
                       const tensor::Tensor& key_cache, const tensor::Tensor& val_cache,
                       const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                       int32_t dim, int32_t kv_dim, int32_t head_size,
                       int32_t section0, int32_t section1, int32_t section2,
                       int32_t layer_idx, int32_t seq_len);
};

/**
 * @brief FusedGQAMRoPEKVDecodeLayer: Fused GQA + M-RoPE + KV Cache Read/Write for decode
 *
 * Fuses five operations into a single kernel launch:
 *   1. Apply M-RoPE to Q (stays in shared memory)
 *   2. Apply M-RoPE to K for current token
 *   3. Write K/V to KV cache
 *   4. GQA Flash Attention decode (Q·K + softmax + V accumulation)
 *   5. Write attention output
 *
 * Replaces the separate fused_mrope_kv_write + flash_attention_decode kernels.
 */
class FusedGQAMRoPEKVDecodeLayer : public Layer {
 public:
  explicit FusedGQAMRoPEKVDecodeLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;

  /**
   * @brief Execute fused GQA + M-RoPE + KV cache + attention decode
   *
   * @param rope_pos_gpu    GPU pointer to M-RoPE text position
   * @param kv_cache_pos_gpu GPU pointer to KV cache write position
   * @param query           Q tensor [dim] (FP16, read-only)
   * @param key             K tensor [kv_dim] (FP16, read-only)
   * @param value           V tensor [kv_dim] (FP16, read-only)
   * @param key_cache       Key cache [layer_num, seq_len, kv_dim] (FP16)
   * @param val_cache       Value cache [layer_num, seq_len, kv_dim] (FP16)
   * @param output          Attention output [dim] (FP16)
   * @param sin_cache       Sin cache (FP32)
   * @param cos_cache       Cos cache (FP32)
   * @param dim             Q total dimension
   * @param kv_dim          KV total dimension
   * @param head_size       Per-head dimension
   * @param section0        M-RoPE temporal section pairs
   * @param section1        M-RoPE height section pairs
   * @param section2        M-RoPE width section pairs
   * @param layer_idx       Current layer index
   * @param max_seq_len     Maximum sequence length
   */
  base::Status forward(const int32_t* rope_pos_gpu, const int32_t* kv_cache_pos_gpu,
                       const tensor::Tensor& query, const tensor::Tensor& key,
                       const tensor::Tensor& value,
                       const tensor::Tensor& key_cache, const tensor::Tensor& val_cache,
                       const tensor::Tensor& output,
                       const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                       int32_t dim, int32_t kv_dim, int32_t head_size,
                       int32_t section0, int32_t section1, int32_t section2,
                       int32_t layer_idx, int32_t max_seq_len);
};

}  // namespace op
#endif  // KUIPER_INCLUDE_OP_MISC_LAYERS_H_
