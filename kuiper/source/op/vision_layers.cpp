#include "op/vision_layers.h"
#include "kernels/cuda/vision_encoder_kernel.cuh"
#include "kernels/cuda/fused_kernels.cuh"
#include "kernels/cuda/flash_attention_kernel.cuh"
#include "kernels/cpu/image_preprocess_kernel.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

// ==================== FusedNormalizePatchesLayer ====================

FusedNormalizePatchesLayer::FusedNormalizePatchesLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "FusedNormalizePatches") {
  reset_input_size(0);
  reset_output_size(0);
}

base::Status FusedNormalizePatchesLayer::check() const {
  return base::error::Success();
}

base::Status FusedNormalizePatchesLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status FusedNormalizePatchesLayer::forward(const unsigned char* pixels_gpu, half* patches_gpu,
                                                  int32_t height, int32_t width,
                                                  int32_t patch_size, int32_t temporal_patch_size,
                                                  float mean_r, float mean_g, float mean_b,
                                                  float std_r, float std_g, float std_b,
                                                  cudaStream_t stream) {
  kernel::fused_normalize_patches_cu(pixels_gpu, patches_gpu, height, width,
                                      patch_size, temporal_patch_size,
                                      mean_r, mean_g, mean_b,
                                      std_r, std_g, std_b,
                                      stream ? stream : (cuda_config_ ? cuda_config_->stream : nullptr));
  return base::error::Success();
}

base::Status FusedNormalizePatchesLayer::forward_resize(
    const unsigned char* src_pixels_gpu, half* patches_gpu,
    int32_t src_h, int32_t src_w, int32_t dst_h, int32_t dst_w,
    int32_t patch_size, int32_t temporal_patch_size,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b,
    cudaStream_t stream) {
  kernel::fused_resize_normalize_patches_cu(src_pixels_gpu, patches_gpu,
                                             src_h, src_w, dst_h, dst_w,
                                             patch_size, temporal_patch_size,
                                             mean_r, mean_g, mean_b,
                                             std_r, std_g, std_b,
                                             stream ? stream : (cuda_config_ ? cuda_config_->stream : nullptr));
  return base::error::Success();
}

// ==================== CausalSoftmaxLayer ====================

CausalSoftmaxLayer::CausalSoftmaxLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "CausalSoftmax") {
  reset_input_size(0);
  reset_output_size(0);
}

base::Status CausalSoftmaxLayer::check() const {
  return base::error::Success();
}

base::Status CausalSoftmaxLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status CausalSoftmaxLayer::forward(half* scores, int32_t head_num, int32_t seq_len,
                                          int32_t kv_len, int32_t start_pos,
                                          cudaStream_t stream) {
  kernel::causal_softmax_fp16_cu(scores, head_num, seq_len, kv_len, start_pos,
                                  stream ? stream : (cuda_config_ ? cuda_config_->stream : nullptr));
  return base::error::Success();
}

// ==================== LoadImageLayer ====================

LoadImageLayer::LoadImageLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "LoadImage") {}

base::Status LoadImageLayer::check() const { return base::error::Success(); }
base::Status LoadImageLayer::forward() { return base::error::InvalidArgument("Use forward(...) with parameters"); }

base::Status LoadImageLayer::forward(const std::string& path,
                                      std::vector<uint8_t>& result_pixels,
                                      int& width, int& height, int& channels) {
  result_pixels = kernel::load_image_cpu(path, width, height, channels);
  if (result_pixels.empty()) {
    return base::error::InternalError("Failed to load image: " + path);
  }
  return base::error::Success();
}

// ==================== SmartResizeLayer ====================

SmartResizeLayer::SmartResizeLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "SmartResize") {}

base::Status SmartResizeLayer::check() const { return base::error::Success(); }
base::Status SmartResizeLayer::forward() { return base::error::InvalidArgument("Use forward(...) with parameters"); }

base::Status SmartResizeLayer::forward(const std::vector<uint8_t>& pixels,
                                        int src_width, int src_height,
                                        int min_pixels, int max_pixels, int factor,
                                        std::vector<uint8_t>& result_pixels,
                                        int& new_width, int& new_height) {
  auto [resized, w, h] = kernel::smart_resize_cpu(pixels, src_width, src_height,
                                                    min_pixels, max_pixels, factor);
  result_pixels = std::move(resized);
  new_width = w;
  new_height = h;
  return base::error::Success();
}

// ==================== VisionRotaryEmbLayer ====================

VisionRotaryEmbLayer::VisionRotaryEmbLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "VisionRotaryEmb") {}

base::Status VisionRotaryEmbLayer::check() const { return base::error::Success(); }
base::Status VisionRotaryEmbLayer::forward() { return base::error::InvalidArgument("Use forward(...) with parameters"); }

base::Status VisionRotaryEmbLayer::forward(std::vector<uint16_t>& cos_data,
                                            std::vector<uint16_t>& sin_data,
                                            int grid_h, int grid_w, int grid_t,
                                            int num_heads, int hidden_size, int spatial_merge_size) {
  kernel::compute_vision_rotary_emb_cpu(cos_data, sin_data, grid_h, grid_w, grid_t,
                                         num_heads, hidden_size, spatial_merge_size);
  return base::error::Success();
}

// ==================== GenerateMRoPEPositionsLayer ====================

GenerateMRoPEPositionsLayer::GenerateMRoPEPositionsLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "GenerateMRoPEPositions") {}

GenerateMRoPEPositionsLayer::~GenerateMRoPEPositionsLayer() {
  if (mrope_pos_gpu_) {
    cudaFree(mrope_pos_gpu_);
    mrope_pos_gpu_ = nullptr;
  }
  if (mrope_pos_pinned_) {
    cudaFreeHost(mrope_pos_pinned_);
    mrope_pos_pinned_ = nullptr;
  }
}

base::Status GenerateMRoPEPositionsLayer::check() const { return base::error::Success(); }
base::Status GenerateMRoPEPositionsLayer::forward() { return base::error::InvalidArgument("Use forward(...) with parameters"); }

base::Status GenerateMRoPEPositionsLayer::forward(
    const std::vector<int>& tokens,
    int image_token_pos, int num_vision_tokens,
    int grid_h, int grid_w,
    std::vector<int32_t>& mrope_pos_t,
    std::vector<int32_t>& mrope_pos_h,
    std::vector<int32_t>& mrope_pos_w,
    int& max_text_pos,
    int32_t*& pos_t_gpu, int32_t*& pos_h_gpu, int32_t*& pos_w_gpu,
    cudaStream_t stream) {
  
  // Step 1: Generate positions on CPU
  max_text_pos = kernel::generate_mrope_positions_cpu(
      mrope_pos_t, mrope_pos_h, mrope_pos_w,
      tokens, image_token_pos, num_vision_tokens, grid_h, grid_w);
  
  // Step 2: Upload to GPU (fused from upload_mrope_positions_to_gpu)
  size_t total_positions = mrope_pos_t.size();
  if (total_positions == 0) {
    pos_t_gpu = pos_h_gpu = pos_w_gpu = nullptr;
    return base::error::Success();
  }
  
  // Grow GPU allocation if needed
  if (total_positions > mrope_pos_gpu_capacity_) {
    if (mrope_pos_gpu_) cudaFree(mrope_pos_gpu_);
    cudaMalloc(&mrope_pos_gpu_, 3 * total_positions * sizeof(int32_t));
    mrope_pos_gpu_capacity_ = total_positions;
    
    if (total_positions > mrope_pos_pinned_capacity_) {
      if (mrope_pos_pinned_) cudaFreeHost(mrope_pos_pinned_);
      cudaMallocHost(&mrope_pos_pinned_, 3 * total_positions * sizeof(int32_t));
      mrope_pos_pinned_capacity_ = total_positions;
    }
  }
  
  // Pack into contiguous pinned memory
  int32_t* pinned_t = mrope_pos_pinned_;
  int32_t* pinned_h = mrope_pos_pinned_ + total_positions;
  int32_t* pinned_w = mrope_pos_pinned_ + 2 * total_positions;
  memcpy(pinned_t, mrope_pos_t.data(), total_positions * sizeof(int32_t));
  memcpy(pinned_h, mrope_pos_h.data(), total_positions * sizeof(int32_t));
  memcpy(pinned_w, mrope_pos_w.data(), total_positions * sizeof(int32_t));
  
  // Single async H2D transfer
  cudaMemcpyAsync(mrope_pos_gpu_, mrope_pos_pinned_,
                  3 * total_positions * sizeof(int32_t), cudaMemcpyHostToDevice,
                  stream);
  
  // Set output pointers
  pos_t_gpu = mrope_pos_gpu_;
  pos_h_gpu = mrope_pos_gpu_ + total_positions;
  pos_w_gpu = mrope_pos_gpu_ + 2 * total_positions;
  
  return base::error::Success();
}

base::Status GenerateMRoPEPositionsLayer::upload(
    const std::vector<int32_t>& mrope_pos_t,
    const std::vector<int32_t>& mrope_pos_h,
    const std::vector<int32_t>& mrope_pos_w,
    int32_t*& pos_t_gpu, int32_t*& pos_h_gpu, int32_t*& pos_w_gpu,
    cudaStream_t stream) {
  size_t total_positions = mrope_pos_t.size();
  if (total_positions == 0) {
    pos_t_gpu = pos_h_gpu = pos_w_gpu = nullptr;
    return base::error::Success();
  }
  if (total_positions > mrope_pos_gpu_capacity_) {
    if (mrope_pos_gpu_) cudaFree(mrope_pos_gpu_);
    cudaMalloc(&mrope_pos_gpu_, 3 * total_positions * sizeof(int32_t));
    mrope_pos_gpu_capacity_ = total_positions;
    if (total_positions > mrope_pos_pinned_capacity_) {
      if (mrope_pos_pinned_) cudaFreeHost(mrope_pos_pinned_);
      cudaMallocHost(&mrope_pos_pinned_, 3 * total_positions * sizeof(int32_t));
      mrope_pos_pinned_capacity_ = total_positions;
    }
  }
  int32_t* pinned_t = mrope_pos_pinned_;
  int32_t* pinned_h = mrope_pos_pinned_ + total_positions;
  int32_t* pinned_w = mrope_pos_pinned_ + 2 * total_positions;
  memcpy(pinned_t, mrope_pos_t.data(), total_positions * sizeof(int32_t));
  memcpy(pinned_h, mrope_pos_h.data(), total_positions * sizeof(int32_t));
  memcpy(pinned_w, mrope_pos_w.data(), total_positions * sizeof(int32_t));
  cudaMemcpyAsync(mrope_pos_gpu_, mrope_pos_pinned_,
                  3 * total_positions * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
  pos_t_gpu = mrope_pos_gpu_;
  pos_h_gpu = mrope_pos_gpu_ + total_positions;
  pos_w_gpu = mrope_pos_gpu_ + 2 * total_positions;
  return base::error::Success();
}

// ==================== VisionPatchEmbedLayer ====================

VisionPatchEmbedLayer::VisionPatchEmbedLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "VisionPatchEmbed") {}

base::Status VisionPatchEmbedLayer::check() const { return base::error::Success(); }
base::Status VisionPatchEmbedLayer::forward() { return base::error::InvalidArgument("Use forward(...) with parameters"); }

base::Status VisionPatchEmbedLayer::forward(const tensor::Tensor& pixel_values,
                                             const tensor::Tensor& weight,
                                             const tensor::Tensor& bias,
                                             tensor::Tensor& output,
                                             int num_patches, int hidden_size, int patch_dim,
                                             kernel::CudaConfig* cuda_config) {
  if (!cuda_config) cuda_config = cuda_config_.get();
  
  // GEMM: output = pixel_values @ weight^T + bias
  const half alpha = __float2half(1.0f);
  const half beta = __float2half(0.0f);
  
  cublasHgemm(cuda_config->cublas_handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              hidden_size, num_patches, patch_dim,
              &alpha,
              weight.ptr<half>(), patch_dim,
              pixel_values.ptr<half>(), patch_dim,
              &beta,
              output.ptr<half>(), hidden_size);
  
  // Add bias
  kernel::bias_add_residual_cu(output, bias, tensor::Tensor(), output,
                               cuda_config->stream);
  
  return base::error::Success();
}

// ==================== BatchedGemmLayer ====================

BatchedGemmLayer::BatchedGemmLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "BatchedGemm") {}

base::Status BatchedGemmLayer::check() const { return base::error::Success(); }
base::Status BatchedGemmLayer::forward() { return base::error::InvalidArgument("Use forward(...) with parameters"); }

base::Status BatchedGemmLayer::forward(const half* A, const half* B, half* C,
                                        int M, int N, int K,
                                        bool trans_a, bool trans_b,
                                        float alpha_f, float beta_f,
                                        int lda, int ldb, int ldc,
                                        kernel::CudaConfig* cuda_config) {
  if (!cuda_config) cuda_config = cuda_config_.get();
  
  const half alpha = __float2half(alpha_f);
  const half beta = __float2half(beta_f);
  
  cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  
  cublasHgemm(cuda_config->cublas_handle,
              op_a, op_b,
              M, N, K,
              &alpha,
              A, lda,
              B, ldb,
              &beta,
              C, ldc);
  
  return base::error::Success();
}

base::Status BatchedGemmLayer::forward_batched(const half** A_array, const half** B_array, half** C_array,
                                                int M, int N, int K,
                                                bool trans_a, bool trans_b,
                                                float alpha_f, float beta_f,
                                                int lda, int ldb, int ldc,
                                                int batch_count,
                                                kernel::CudaConfig* cuda_config) {
  if (!cuda_config) cuda_config = cuda_config_.get();
  
  const half alpha = __float2half(alpha_f);
  const half beta = __float2half(beta_f);
  
  cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  
  cublasHgemmBatched(cuda_config->cublas_handle,
                     op_a, op_b,
                     M, N, K,
                     &alpha,
                     A_array, lda,
                     B_array, ldb,
                     &beta,
                     C_array, ldc,
                     batch_count);
  
  return base::error::Success();
}

}  // namespace op
