#ifndef KUIPER_INCLUDE_OP_VISION_LAYERS_H_
#define KUIPER_INCLUDE_OP_VISION_LAYERS_H_
#include "layer.h"

namespace op {

/**
 * @brief ExtractPatchesLayer: Extract image patches for ViT
 */
class ExtractPatchesLayer : public Layer {
 public:
  explicit ExtractPatchesLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward for patch extraction
  base::Status forward(const tensor::Tensor& image, const tensor::Tensor& patches,
                       int32_t channels, int32_t height, int32_t width,
                       int32_t patch_size, int32_t temporal_patch_size);
};

/**
 * @brief BiasAddResidualLayer: Add bias and residual connection
 */
class BiasAddResidualLayer : public Layer {
 public:
  explicit BiasAddResidualLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward: output = input + bias + residual
  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& bias,
                       const tensor::Tensor& residual, const tensor::Tensor& output,
                       cudaStream_t stream = nullptr);
};

/**
 * @brief PosEmbedInterpolateLayer: Interpolate position embeddings
 */
class PosEmbedInterpolateLayer : public Layer {
 public:
  explicit PosEmbedInterpolateLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward for position embedding interpolation
  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_embed,
                       const tensor::Tensor& output,
                       int32_t grid_h, int32_t grid_w, int32_t grid_t,
                       int32_t num_grid_per_side, int32_t spatial_merge_size,
                       cudaStream_t stream = nullptr);
};

/**
 * @brief LayerNormWithBiasLayer: LayerNorm with bias
 */
class LayerNormWithBiasLayer : public Layer {
 public:
  explicit LayerNormWithBiasLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward for LayerNorm with bias
  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& bias, const tensor::Tensor& output,
                       float eps = 1e-6f, cudaStream_t stream = nullptr);
};

/**
 * @brief FusedSplitRopeTransposeLayer: Fused split + RoPE + transpose
 */
class FusedSplitRopeTransposeLayer : public Layer {
 public:
  explicit FusedSplitRopeTransposeLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward for fused operation
  base::Status forward(const tensor::Tensor& qkv, const tensor::Tensor& cos_cache,
                       const tensor::Tensor& sin_cache,
                       const tensor::Tensor& q_out, const tensor::Tensor& k_out,
                       const tensor::Tensor& v_out,
                       int32_t num_tokens, int32_t num_heads, int32_t head_dim,
                       cudaStream_t stream = nullptr);
};

/**
 * @brief VisionAttentionLayer: Vision self-attention with pre-transposed Q/K/V
 */
class VisionAttentionLayer : public Layer {
 public:
  explicit VisionAttentionLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward for vision attention
  base::Status forward(const tensor::Tensor& q_transposed, const tensor::Tensor& k_transposed,
                       const tensor::Tensor& v_transposed, const tensor::Tensor& attn_out,
                       const tensor::Tensor& out_transposed, const tensor::Tensor& attn_scores,
                       int32_t num_tokens, int32_t num_heads, int32_t head_dim, float scale,
                       kernel::CudaConfig* cuda_config = nullptr);
};

/**
 * @brief VisionMLPLayer: Vision MLP (fc1 + GELU + fc2 + residual)
 */
class VisionMLPLayer : public Layer {
 public:
  explicit VisionMLPLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward for vision MLP
  base::Status forward(const tensor::Tensor& input,
                       const tensor::Tensor& fc1_weight, const tensor::Tensor& fc1_bias,
                       const tensor::Tensor& fc2_weight, const tensor::Tensor& fc2_bias,
                       const tensor::Tensor& residual, const tensor::Tensor& output,
                       const tensor::Tensor& intermediate,
                       kernel::CudaConfig* cuda_config = nullptr);
};

/**
 * @brief SpatialMergeLayer: Merge spatial patches
 */
class SpatialMergeLayer : public Layer {
 public:
  explicit SpatialMergeLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward for spatial merge
  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& output,
                       int32_t grid_t, int32_t grid_h, int32_t grid_w,
                       int32_t hidden_size, int32_t merge_size,
                       cudaStream_t stream = nullptr);
};

/**
 * @brief VisionMergerMLPLayer: Vision merger MLP
 */
class VisionMergerMLPLayer : public Layer {
 public:
  explicit VisionMergerMLPLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward for merger MLP
  base::Status forward(const tensor::Tensor& input,
                       const tensor::Tensor& fc1_weight, const tensor::Tensor& fc1_bias,
                       const tensor::Tensor& fc2_weight, const tensor::Tensor& fc2_bias,
                       const tensor::Tensor& output, const tensor::Tensor& intermediate,
                       kernel::CudaConfig* cuda_config = nullptr);
};

/**
 * @brief FusedMultimodalEmbedLayer: Fused multimodal embedding assembly
 */
class FusedMultimodalEmbedLayer : public Layer {
 public:
  explicit FusedMultimodalEmbedLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward for multimodal embedding
  base::Status forward(const tensor::Tensor& text_embeds, const tensor::Tensor& visual_embeds,
                       const tensor::Tensor& output,
                       int32_t image_token_pos, int32_t num_vision_tokens,
                       int32_t num_text_tokens, int32_t dim,
                       cudaStream_t stream = nullptr);
};

}  // namespace op
#endif  // KUIPER_INCLUDE_OP_VISION_LAYERS_H_
