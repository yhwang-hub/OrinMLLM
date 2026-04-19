#ifndef KUIPER_INCLUDE_OP_VISION_LAYERS_H_
#define KUIPER_INCLUDE_OP_VISION_LAYERS_H_
#include "layer.h"
#include <cuda_fp16.h>
#include <string>
#include <vector>
#include <tuple>

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

/**
 * @brief FusedNormalizePatchesLayer: Fused normalize + patch extraction from uint8 pixels
 * Converts uint8 HWC pixels to fp16 normalized patches in 2x2 block interleaved order.
 */
class FusedNormalizePatchesLayer : public Layer {
 public:
  explicit FusedNormalizePatchesLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;

  // Direct forward: fused normalize + patch extraction
  base::Status forward(const unsigned char* pixels_gpu, half* patches_gpu,
                       int32_t height, int32_t width,
                       int32_t patch_size, int32_t temporal_patch_size,
                       float mean_r, float mean_g, float mean_b,
                       float std_r, float std_g, float std_b,
                       cudaStream_t stream = nullptr);

  // GPU fused-resize + normalize + patches (9.4): takes ORIGINAL uint8 pixels
  // and produces patches directly, replacing CPU stbir resize.
  base::Status forward_resize(const unsigned char* src_pixels_gpu, half* patches_gpu,
                              int32_t src_h, int32_t src_w,
                              int32_t dst_h, int32_t dst_w,
                              int32_t patch_size, int32_t temporal_patch_size,
                              float mean_r, float mean_g, float mean_b,
                              float std_r, float std_g, float std_b,
                              cudaStream_t stream = nullptr);
};

/**
 * @brief CausalSoftmaxLayer: Causal softmax for cuBLAS-based prefill attention (FP16)
 * Applied to score matrix: [head_num × kv_len × seq_len] column-major per head
 */
class CausalSoftmaxLayer : public Layer {
 public:
  explicit CausalSoftmaxLayer(base::DeviceType device_type);

  base::Status check() const override;
  base::Status forward() override;

  // Direct forward: causal softmax on attention scores
  base::Status forward(half* scores, int32_t head_num, int32_t seq_len,
                       int32_t kv_len, int32_t start_pos,
                       cudaStream_t stream = nullptr);
};

}  // namespace op

// ============================================================================
// CPU Preprocessing Layers (image load, resize, rotary emb, mrope positions)
// ============================================================================
namespace op {

/**
 * @brief LoadImageLayer: Load image from file (CPU)
 */
class LoadImageLayer : public Layer {
 public:
  explicit LoadImageLayer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;

  // Direct forward: load image from path
  // Returns pixel data in result_pixels, dimensions in width/height/channels
  base::Status forward(const std::string& path,
                       std::vector<uint8_t>& result_pixels,
                       int& width, int& height, int& channels);
};

/**
 * @brief SmartResizeLayer: Smart resize for VL model (CPU)
 */
class SmartResizeLayer : public Layer {
 public:
  explicit SmartResizeLayer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;

  // Direct forward: smart resize
  base::Status forward(const std::vector<uint8_t>& pixels,
                       int src_width, int src_height,
                       int min_pixels, int max_pixels, int factor,
                       std::vector<uint8_t>& result_pixels,
                       int& new_width, int& new_height);
};

/**
 * @brief VisionRotaryEmbLayer: Compute vision rotary embeddings (CPU)
 */
class VisionRotaryEmbLayer : public Layer {
 public:
  explicit VisionRotaryEmbLayer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;

  // Direct forward: compute vision rotary embeddings
  // cos_data/sin_data are uint16_t vectors (FP16 data)
  base::Status forward(std::vector<uint16_t>& cos_data,
                       std::vector<uint16_t>& sin_data,
                       int grid_h, int grid_w, int grid_t,
                       int num_heads, int hidden_size, int spatial_merge_size);
};

/**
 * @brief GenerateMRoPEPositionsLayer: Generate M-RoPE 3D positions (CPU)
 * Also handles GPU upload of positions (fused from upload_mrope_positions_to_gpu)
 */
class GenerateMRoPEPositionsLayer : public Layer {
 public:
  explicit GenerateMRoPEPositionsLayer(base::DeviceType device_type);
  ~GenerateMRoPEPositionsLayer();
  base::Status check() const override;
  base::Status forward() override;

  // Direct forward: generate positions + upload to GPU
  // Returns max_text_pos, GPU pointers are stored internally
  base::Status forward(const std::vector<int>& tokens,
                       int image_token_pos, int num_vision_tokens,
                       int grid_h, int grid_w,
                       std::vector<int32_t>& mrope_pos_t,
                       std::vector<int32_t>& mrope_pos_h,
                       std::vector<int32_t>& mrope_pos_w,
                       int& max_text_pos,
                       int32_t*& pos_t_gpu, int32_t*& pos_h_gpu, int32_t*& pos_w_gpu,
                       cudaStream_t stream = nullptr);

  // Upload-only: upload pre-computed positions to GPU (for Qwen3.5 text-only path)
  base::Status upload(const std::vector<int32_t>& mrope_pos_t,
                      const std::vector<int32_t>& mrope_pos_h,
                      const std::vector<int32_t>& mrope_pos_w,
                      int32_t*& pos_t_gpu, int32_t*& pos_h_gpu, int32_t*& pos_w_gpu,
                      cudaStream_t stream = nullptr);

 private:
  // GPU memory for uploaded positions
  int32_t* mrope_pos_gpu_ = nullptr;
  size_t mrope_pos_gpu_capacity_ = 0;
  int32_t* mrope_pos_pinned_ = nullptr;
  size_t mrope_pos_pinned_capacity_ = 0;
};

/**
 * @brief VisionPatchEmbedLayer: Fused Conv3D patch embedding (GEMM + bias)
 * Replaces cublasHgemm + BiasAddResidual with a single fused layer call.
 */
class VisionPatchEmbedLayer : public Layer {
 public:
  explicit VisionPatchEmbedLayer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;

  // Direct forward: patch embedding via GEMM + bias
  base::Status forward(const tensor::Tensor& pixel_values,
                       const tensor::Tensor& weight,
                       const tensor::Tensor& bias,
                       tensor::Tensor& output,
                       int num_patches, int hidden_size, int patch_dim,
                       kernel::CudaConfig* cuda_config = nullptr);
};

/**
 * @brief BatchedGemmLayer: Wrapper for cublasHgemm batched operations
 * Used to encapsulate cuBLAS calls as layers for cleaner model code.
 */
class BatchedGemmLayer : public Layer {
 public:
  explicit BatchedGemmLayer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;

  // Forward: C = alpha * op(A) @ op(B) + beta * C
  // A: [M, K] or [K, M], B: [K, N] or [N, K], C: [M, N]
  // trans_a/trans_b: whether to transpose A/B
  base::Status forward(const half* A, const half* B, half* C,
                       int M, int N, int K,
                       bool trans_a, bool trans_b,
                       float alpha, float beta,
                       int lda, int ldb, int ldc,
                       kernel::CudaConfig* cuda_config = nullptr);

  // Batched variant: C[i] = alpha * op(A[i]) @ op(B[i]) + beta * C[i]
  base::Status forward_batched(const half** A_array, const half** B_array, half** C_array,
                               int M, int N, int K,
                               bool trans_a, bool trans_b,
                               float alpha, float beta,
                               int lda, int ldb, int ldc,
                               int batch_count,
                               kernel::CudaConfig* cuda_config = nullptr);
};

}  // namespace op

#endif  // KUIPER_INCLUDE_OP_VISION_LAYERS_H_
