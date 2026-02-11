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
  
  // Direct forward for Flash Attention decode with GPU position
  base::Status forward(const int32_t* pos_gpu, int32_t head_num, int32_t kv_head_num,
                       int32_t head_size, int32_t kv_mul, int32_t layer_idx,
                       int32_t seq_len, int32_t kv_dim,
                       const tensor::Tensor& query, const tensor::Tensor& mha_output,
                       const tensor::Tensor& key_cache, const tensor::Tensor& val_cache);
};

}  // namespace op
#endif  // KUIPER_INCLUDE_OP_MISC_LAYERS_H_
