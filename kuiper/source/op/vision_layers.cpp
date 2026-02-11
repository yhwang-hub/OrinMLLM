#include "op/vision_layers.h"
#include "kernels/cuda/vision_encoder_kernel.cuh"
#include "kernels/cuda/fused_kernels.cuh"

namespace op {

// ==================== ExtractPatchesLayer ====================

ExtractPatchesLayer::ExtractPatchesLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "ExtractPatches") {
  reset_input_size(1);
  reset_output_size(1);
}

base::Status ExtractPatchesLayer::check() const {
  return base::error::Success();
}

base::Status ExtractPatchesLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status ExtractPatchesLayer::forward(const tensor::Tensor& image, const tensor::Tensor& patches,
                                           int32_t channels, int32_t height, int32_t width,
                                           int32_t patch_size, int32_t temporal_patch_size) {
  kernel::extract_patches_cu(image, const_cast<tensor::Tensor&>(patches), 
                             channels, height, width,
                             patch_size, temporal_patch_size,
                             cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== BiasAddResidualLayer ====================

BiasAddResidualLayer::BiasAddResidualLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "BiasAddResidual") {
  reset_input_size(3);  // input, bias, residual
  reset_output_size(1);
}

base::Status BiasAddResidualLayer::check() const {
  return base::error::Success();
}

base::Status BiasAddResidualLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status BiasAddResidualLayer::forward(const tensor::Tensor& input, const tensor::Tensor& bias,
                                            const tensor::Tensor& residual, const tensor::Tensor& output,
                                            cudaStream_t stream) {
  kernel::bias_add_residual_cu(input, bias, residual, const_cast<tensor::Tensor&>(output),
                               stream ? stream : (cuda_config_ ? cuda_config_->stream : nullptr));
  return base::error::Success();
}

// ==================== PosEmbedInterpolateLayer ====================

PosEmbedInterpolateLayer::PosEmbedInterpolateLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "PosEmbedInterpolate") {
  reset_input_size(2);
  reset_output_size(1);
}

base::Status PosEmbedInterpolateLayer::check() const {
  return base::error::Success();
}

base::Status PosEmbedInterpolateLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status PosEmbedInterpolateLayer::forward(const tensor::Tensor& input, const tensor::Tensor& pos_embed,
                                                const tensor::Tensor& output,
                                                int32_t grid_h, int32_t grid_w, int32_t grid_t,
                                                int32_t num_grid_per_side, int32_t spatial_merge_size,
                                                cudaStream_t stream) {
  kernel::pos_embed_interpolate_cu(input, pos_embed, const_cast<tensor::Tensor&>(output),
                                   grid_h, grid_w, grid_t,
                                   num_grid_per_side, spatial_merge_size,
                                   stream ? stream : (cuda_config_ ? cuda_config_->stream : nullptr));
  return base::error::Success();
}

// ==================== LayerNormWithBiasLayer ====================

LayerNormWithBiasLayer::LayerNormWithBiasLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "LayerNormWithBias") {
  reset_input_size(3);  // input, weight, bias
  reset_output_size(1);
}

base::Status LayerNormWithBiasLayer::check() const {
  return base::error::Success();
}

base::Status LayerNormWithBiasLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status LayerNormWithBiasLayer::forward(const tensor::Tensor& input, const tensor::Tensor& weight,
                                              const tensor::Tensor& bias, const tensor::Tensor& output,
                                              float eps, cudaStream_t stream) {
  kernel::layernorm_with_bias_cu(input, weight, bias, const_cast<tensor::Tensor&>(output), eps,
                                 stream ? stream : (cuda_config_ ? cuda_config_->stream : nullptr));
  return base::error::Success();
}

// ==================== FusedSplitRopeTransposeLayer ====================

FusedSplitRopeTransposeLayer::FusedSplitRopeTransposeLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "FusedSplitRopeTranspose") {
  reset_input_size(3);  // qkv, cos_cache, sin_cache
  reset_output_size(3); // q, k, v
}

base::Status FusedSplitRopeTransposeLayer::check() const {
  return base::error::Success();
}

base::Status FusedSplitRopeTransposeLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status FusedSplitRopeTransposeLayer::forward(const tensor::Tensor& qkv, const tensor::Tensor& cos_cache,
                                                    const tensor::Tensor& sin_cache,
                                                    const tensor::Tensor& q_out, const tensor::Tensor& k_out,
                                                    const tensor::Tensor& v_out,
                                                    int32_t num_tokens, int32_t num_heads, int32_t head_dim,
                                                    cudaStream_t stream) {
  kernel::fused_split_rope_transpose_cu(qkv, cos_cache, sin_cache,
                                        const_cast<tensor::Tensor&>(q_out), 
                                        const_cast<tensor::Tensor&>(k_out), 
                                        const_cast<tensor::Tensor&>(v_out),
                                        num_tokens, num_heads, head_dim,
                                        stream);
  return base::error::Success();
}

// ==================== VisionAttentionLayer ====================

VisionAttentionLayer::VisionAttentionLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerMHA, "VisionAttention") {
  reset_input_size(3);  // q, k, v
  reset_output_size(1);
}

base::Status VisionAttentionLayer::check() const {
  return base::error::Success();
}

base::Status VisionAttentionLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status VisionAttentionLayer::forward(const tensor::Tensor& q_transposed, const tensor::Tensor& k_transposed,
                                            const tensor::Tensor& v_transposed, const tensor::Tensor& attn_out,
                                            const tensor::Tensor& out_transposed, const tensor::Tensor& attn_scores,
                                            int32_t num_tokens, int32_t num_heads, int32_t head_dim, float scale,
                                            kernel::CudaConfig* cuda_config) {
  kernel::vision_attention_pretransposed_cu(q_transposed, k_transposed, v_transposed,
                                            const_cast<tensor::Tensor&>(attn_out), 
                                            const_cast<tensor::Tensor&>(out_transposed), 
                                            const_cast<tensor::Tensor&>(attn_scores),
                                            num_tokens, num_heads, head_dim, scale,
                                            cuda_config);
  return base::error::Success();
}

// ==================== VisionMLPLayer ====================

VisionMLPLayer::VisionMLPLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "VisionMLP") {
  reset_input_size(5);  // input, fc1_w, fc1_b, fc2_w, fc2_b
  reset_output_size(1);
}

base::Status VisionMLPLayer::check() const {
  return base::error::Success();
}

base::Status VisionMLPLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status VisionMLPLayer::forward(const tensor::Tensor& input,
                                      const tensor::Tensor& fc1_weight, const tensor::Tensor& fc1_bias,
                                      const tensor::Tensor& fc2_weight, const tensor::Tensor& fc2_bias,
                                      const tensor::Tensor& residual, const tensor::Tensor& output,
                                      const tensor::Tensor& intermediate,
                                      kernel::CudaConfig* cuda_config) {
  kernel::vision_mlp_cu(input, fc1_weight, fc1_bias, fc2_weight, fc2_bias,
                        residual, const_cast<tensor::Tensor&>(output), 
                        const_cast<tensor::Tensor&>(intermediate), cuda_config);
  return base::error::Success();
}

// ==================== SpatialMergeLayer ====================

SpatialMergeLayer::SpatialMergeLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "SpatialMerge") {
  reset_input_size(1);
  reset_output_size(1);
}

base::Status SpatialMergeLayer::check() const {
  return base::error::Success();
}

base::Status SpatialMergeLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status SpatialMergeLayer::forward(const tensor::Tensor& input, const tensor::Tensor& output,
                                         int32_t grid_t, int32_t grid_h, int32_t grid_w,
                                         int32_t hidden_size, int32_t merge_size,
                                         cudaStream_t stream) {
  kernel::spatial_merge_cu(input, const_cast<tensor::Tensor&>(output), grid_t, grid_h, grid_w,
                           hidden_size, merge_size, stream);
  return base::error::Success();
}

// ==================== VisionMergerMLPLayer ====================

VisionMergerMLPLayer::VisionMergerMLPLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "VisionMergerMLP") {
  reset_input_size(5);
  reset_output_size(1);
}

base::Status VisionMergerMLPLayer::check() const {
  return base::error::Success();
}

base::Status VisionMergerMLPLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status VisionMergerMLPLayer::forward(const tensor::Tensor& input,
                                            const tensor::Tensor& fc1_weight, const tensor::Tensor& fc1_bias,
                                            const tensor::Tensor& fc2_weight, const tensor::Tensor& fc2_bias,
                                            const tensor::Tensor& output, const tensor::Tensor& intermediate,
                                            kernel::CudaConfig* cuda_config) {
  kernel::vision_merger_mlp_cu(input, fc1_weight, fc1_bias, fc2_weight, fc2_bias,
                               const_cast<tensor::Tensor&>(output), 
                               const_cast<tensor::Tensor&>(intermediate), cuda_config);
  return base::error::Success();
}

// ==================== FusedMultimodalEmbedLayer ====================

FusedMultimodalEmbedLayer::FusedMultimodalEmbedLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "FusedMultimodalEmbed") {
  reset_input_size(2);  // text_embeds, visual_embeds
  reset_output_size(1);
}

base::Status FusedMultimodalEmbedLayer::check() const {
  return base::error::Success();
}

base::Status FusedMultimodalEmbedLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status FusedMultimodalEmbedLayer::forward(const tensor::Tensor& text_embeds, const tensor::Tensor& visual_embeds,
                                                 const tensor::Tensor& output,
                                                 int32_t image_token_pos, int32_t num_vision_tokens,
                                                 int32_t num_text_tokens, int32_t dim,
                                                 cudaStream_t stream) {
  kernel::fused_multimodal_embed_cu(text_embeds, visual_embeds, const_cast<tensor::Tensor&>(output),
                                    image_token_pos, num_vision_tokens,
                                    num_text_tokens, dim, stream);
  return base::error::Success();
}

}  // namespace op
