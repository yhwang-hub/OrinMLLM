/**
 * @file qwen3_vl.cpp
 * @brief Qwen3-VL Vision-Language Model Forward Pass Implementation
 * 
 * This file implements the forward pass (encode, prefill, decode) for Qwen3-VL.
 * Initialization, weight loading, and memory management are in qwen3_vl_base.cpp.
 */

#ifdef QWEN3_VL_SUPPORT
#include "model/qwen3_vl.h"
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>

#include "op/matmul.h"
#include "op/mha.h"
#include "op/rmsnorm.h"
#include "op/batched_add.h"
#include "op/vision_layers.h"
#include "../op/kernels/cuda/kv_cache_kernel.cuh"
#include "../op/kernels/cuda/argmax_kernel.cuh"
#include "../op/kernels/cpu/image_preprocess_kernel.h"
#include "sampler/argmax_sampler.h"
#include "base/tick.h"

namespace model {

// ============================================================================
// LLM Forward Helper Functions
// ============================================================================

void Qwen3VLModel::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  std::shared_ptr<op::Layer> rmsnorm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx);
  CHECK_NE(rmsnorm_layer, nullptr);
  STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

void Qwen3VLModel::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  
  int32_t pos = pos_tensor.index<int32_t>(0);
  auto [key, val] = slice_kv_cache(layer_idx, pos);

  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));
  const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
  STATUS_CHECK(key_layer->forward(rmsnorm_output, key));
  const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);
  STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

  auto query_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 2 * config_->layer_num_ + 1);
  query.reshape({(int32_t)query.size() / config_->head_size_, config_->head_size_});
  query_norm->forward(query, query);
  query.reshape({(int32_t)query.size()});

  auto key_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 3 * config_->layer_num_ + 1);
  key.reshape({(int32_t)key.size() / config_->head_size_, config_->head_size_});
  key_norm->forward(key, key);
  key.reshape({(int32_t)key.size()});

  const auto& section = vl_config_.text.mrope_section;
  if (!mrope_pos_t_.empty() && pos < static_cast<int32_t>(mrope_pos_t_.size())) {
    qwen_layers_->mrope_layer_->forward(
        mrope_pos_t_[pos], mrope_pos_h_[pos], mrope_pos_w_[pos],
        config_->dim_, config_->kv_dim_, config_->head_size_,
        section[0], section[1], section[2],
        query, key,
        get_buffer(ModelBufferType::kSinCache),
        get_buffer(ModelBufferType::kCosCache));
  } else {
    int32_t text_pos = mrope_max_text_pos_ + (pos - prefill_seq_len_) + 1;
    qwen_layers_->mrope_layer_->forward(
        text_pos, text_pos, text_pos,
        config_->dim_, config_->kv_dim_, config_->head_size_,
        section[0], section[1], section[2],
        query, key,
        get_buffer(ModelBufferType::kSinCache),
        get_buffer(ModelBufferType::kCosCache));
  }
}

void Qwen3VLModel::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor query = get_buffer(ModelBufferType::kQuery);
  int pos = pos_tensor.index<int32_t>(0);

  if (query.data_type() == base::DataType::kDataTypeFp16 &&
      key_cache.data_type() == base::DataType::kDataTypeFp16) {
    qwen_layers_->flash_attention_decode_layer_->forward(
        pos, config_->head_num_, config_->kv_head_num_,
        config_->head_size_, config_->kv_mul_, layer_idx,
        config_->seq_len_, config_->kv_dim_,
        query, mha_output, key_cache, val_cache);
  } else if (attention_type_ == base::AttentionType::kAttentionMHA) {
    tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
    const auto& mha_layer = qwen_layers_->mha_layer_;
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
    STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));
  } else {
    qwen_layers_->flash_attention_decode_layer_->forward(
        pos, config_->head_num_, config_->kv_head_num_,
        config_->head_size_, config_->kv_mul_, layer_idx,
        config_->seq_len_, config_->kv_dim_,
        query, mha_output, key_cache, val_cache);
  }

  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = qwen_layers_->wo_layers_.at(layer_idx);
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void Qwen3VLModel::attention_qkv_with_graph(int32_t layer_idx,
                                             const tensor::Tensor& rope_pos_gpu,
                                             const tensor::Tensor& kv_cache_pos_gpu) const {
  CHECK(qwen_layers_ != nullptr && cuda_config_ != nullptr);
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  tensor::Tensor temp_key = this->get_buffer(ModelBufferType::kTempKey);
  tensor::Tensor temp_value = this->get_buffer(ModelBufferType::kTempValue);
  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);

  STATUS_CHECK(qwen_layers_->wq_layers_.at(layer_idx)->forward(rmsnorm_output, query));
  STATUS_CHECK(qwen_layers_->wk_layers_.at(layer_idx)->forward(rmsnorm_output, temp_key));
  STATUS_CHECK(qwen_layers_->wv_layers_.at(layer_idx)->forward(rmsnorm_output, temp_value));

  auto query_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 2 * config_->layer_num_ + 1);
  query.reshape({(int32_t)query.size() / config_->head_size_, config_->head_size_});
  query_norm->forward(query, query);
  query.reshape({(int32_t)query.size()});

  auto key_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 3 * config_->layer_num_ + 1);
  temp_key.reshape({(int32_t)temp_key.size() / config_->head_size_, config_->head_size_});
  key_norm->forward(temp_key, temp_key);
  temp_key.reshape({(int32_t)temp_key.size()});

  const auto& section = vl_config_.text.mrope_section;
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

  qwen_layers_->mrope_gpu_pos_layer_->forward(
      rope_pos_gpu.ptr<int32_t>(),
      config_->dim_, config_->kv_dim_, config_->head_size_,
      section[0], section[1], section[2],
      query, temp_key,
      get_buffer(ModelBufferType::kSinCache), get_buffer(ModelBufferType::kCosCache));

  qwen_layers_->copy_to_kv_cache_layer_->forward(
      key_cache, temp_key, kv_cache_pos_gpu.ptr<int32_t>(),
      config_->kv_dim_, layer_idx, config_->seq_len_);
  qwen_layers_->copy_to_kv_cache_layer_->forward(
      val_cache, temp_value, kv_cache_pos_gpu.ptr<int32_t>(),
      config_->kv_dim_, layer_idx, config_->seq_len_);
}

void Qwen3VLModel::attention_mha_with_graph(int32_t layer_idx,
                                             const tensor::Tensor& rope_pos_gpu,
                                             const tensor::Tensor& kv_cache_pos_gpu) const {
  CHECK(qwen_layers_ != nullptr && cuda_config_ != nullptr);
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);

  if (query.data_type() == base::DataType::kDataTypeFp16 &&
      key_cache.data_type() == base::DataType::kDataTypeFp16) {
    qwen_layers_->flash_attention_decode_gpu_pos_layer_->forward(
        kv_cache_pos_gpu.ptr<int32_t>(), config_->head_num_, config_->kv_head_num_,
        config_->head_size_, config_->kv_mul_, layer_idx,
        config_->seq_len_, config_->kv_dim_,
        query, mha_output, key_cache, val_cache);
  } else if (attention_type_ == base::AttentionType::kAttentionMHA) {
    tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
    qwen_layers_->mha_gpu_pos_layer_->forward(
        kv_cache_pos_gpu.ptr<int32_t>(), config_->head_num_, layer_idx,
        config_->seq_len_, config_->kv_dim_, config_->kv_mul_, config_->head_size_,
        mha_output, query, score_storage, key_cache, val_cache);
  } else {
    qwen_layers_->flash_attention_decode_gpu_pos_layer_->forward(
        kv_cache_pos_gpu.ptr<int32_t>(), config_->head_num_, config_->kv_head_num_,
        config_->head_size_, config_->kv_mul_, layer_idx,
        config_->seq_len_, config_->kv_dim_,
        query, mha_output, key_cache, val_cache);
  }

  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  STATUS_CHECK(qwen_layers_->wo_layers_.at(layer_idx)->forward(mha_output, attn_output));
}

void Qwen3VLModel::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  STATUS_CHECK(qwen_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));

  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  STATUS_CHECK(qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_)->forward(input, ffn_norm_output));

  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = qwen_layers_->w1_layers_.at(layer_idx);
  const auto& w3_layer = qwen_layers_->w3_layers_.at(layer_idx);
  auto w1_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w1_layer);
  auto w3_matmul = std::dynamic_pointer_cast<op::MatmulLayer>(w3_layer);

  if (w1_matmul && w3_matmul && ffn_norm_output.data_type() == base::DataType::kDataTypeFp16) {
    auto fused_ffn = qwen_layers_->fused_ffn_layer_;
    fused_ffn->set_use_fp16(true);
    fused_ffn->set_input(0, ffn_norm_output);
    fused_ffn->set_input(1, w1_matmul->get_weight(0));
    fused_ffn->set_input(2, w3_matmul->get_weight(0));
    fused_ffn->set_output(0, w1_output);
    fused_ffn->set_cuda_config(cuda_config_);
    STATUS_CHECK(fused_ffn->forward());
  } else {
    STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));
    tensor::Tensor w3_output = get_buffer(ModelBufferType::kW3Output);
    STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_output));
    STATUS_CHECK(qwen_layers_->swiglu_layer_->forward(w1_output, w3_output, w1_output));
  }

  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  STATUS_CHECK(qwen_layers_->w2_layers_.at(layer_idx)->forward(w1_output, w2_output));
  STATUS_CHECK(qwen_layers_->add_layer_->forward(input, w2_output, input));
}

void Qwen3VLModel::cls_logits(const tensor::Tensor& input) const {
  CHECK(qwen_layers_ != nullptr);
  tensor::Tensor final_norm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  STATUS_CHECK(qwen_layers_->rmsnorm_layers_.at(2 * config_->layer_num_)->forward(input, final_norm_output));
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  STATUS_CHECK(qwen_layers_->cls_layer_->forward(final_norm_output, forward_output));
}

int32_t Qwen3VLModel::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  size_t next = sampler_->sample(forward_output.ptr<float>(), forward_output.size(),
                                  cuda_config_ ? cuda_config_->stream : nullptr);
  return static_cast<int32_t>(next);
}

// ============================================================================
// Image Preprocessing (using layer->forward())
// ============================================================================

ImageData Qwen3VLModel::preprocess_image(const std::string& image_path, int max_pixels) const {
  ImageData result;

  // 1. Load image via layer
  int width, height, channels;
  std::vector<uint8_t> pixels;
  auto status = vision_vl_layers_.load_image_layer_->forward(image_path, pixels, width, height, channels);
  if (!status) {
    LOG(ERROR) << "Failed to load image: " << image_path;
    return result;
  }

  // 2. Smart resize via layer (CPU stbir with STBIR_FILTER_DEFAULT).
  //    Kept on CPU to guarantee bit-for-bit identical preprocessing output
  //    (GPU bicubic re-implementation diverges from stbir's fixed-point +
  //    separable-pass + edge-replicate pipeline, which cascades through 27
  //    ViT layers and flips sampled tokens).
  int factor = vl_config_.vision.patch_size * vl_config_.vision.spatial_merge_size;
  constexpr int min_pixels = 56 * 56;
  std::vector<uint8_t> resized_pixels;
  int new_width = 0, new_height = 0;
  status = vision_vl_layers_.smart_resize_layer_->forward(
      pixels, width, height, min_pixels, max_pixels, factor,
      resized_pixels, new_width, new_height);
  if (!status) {
    LOG(ERROR) << "Smart resize failed";
    return result;
  }

  // 3. Calculate grid dimensions
  result.grid_h = new_height / vl_config_.vision.patch_size;
  result.grid_w = new_width / vl_config_.vision.patch_size;
  result.grid_t = 1;
  result.num_patches = result.grid_h * result.grid_w * result.grid_t;
  int merge_size = vl_config_.vision.spatial_merge_size;
  result.num_vision_tokens = result.num_patches / (merge_size * merge_size);

  // 4. Upload resized pixels + GPU fused normalize/patch extraction
  int patch_dim = 3 * vl_config_.vision.temporal_patch_size *
                  vl_config_.vision.patch_size * vl_config_.vision.patch_size;
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  result.pixel_values = tensor::Tensor(base::DataType::kDataTypeFp16,
                                        result.num_patches, patch_dim, true, alloc);

  cudaStream_t stream = cuda_config_ ? cuda_config_->stream : nullptr;
  size_t resized_bytes = resized_pixels.size();
  if (resized_bytes > pixel_buf_gpu_capacity_) {
    if (pixel_buf_gpu_) cudaFree(pixel_buf_gpu_);
    cudaMalloc(&pixel_buf_gpu_, resized_bytes);
    pixel_buf_gpu_capacity_ = resized_bytes;
  }
  cudaMemcpyAsync(pixel_buf_gpu_, resized_pixels.data(), resized_bytes,
                  cudaMemcpyHostToDevice, stream);

  vision_vl_layers_.fused_normalize_patches_layer_->forward(
      pixel_buf_gpu_, result.pixel_values.ptr<half>(),
      new_height, new_width,
      vl_config_.vision.patch_size, vl_config_.vision.temporal_patch_size,
      0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, stream);

  LOG(INFO) << "Preprocessed image: " << image_path
            << " Grid: " << result.grid_t << "x" << result.grid_h << "x" << result.grid_w
            << " Patches: " << result.num_patches << " -> Vision tokens: " << result.num_vision_tokens;
  return result;
}

// ============================================================================
// Vision Encoder Forward
// ============================================================================

tensor::Tensor Qwen3VLModel::encode_image(const ImageData& image_data) const {
  LOG(INFO) << "Running vision encoder...";
  auto vit_start = std::chrono::high_resolution_clock::now();

  int num_patches = image_data.num_patches;
  int hidden_size = vl_config_.vision.hidden_size;
  int intermediate_size = vl_config_.vision.intermediate_size;
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  int num_heads = vl_config_.vision.num_heads;
  int head_dim = hidden_size / num_heads;

  if (!vision_workspace_ || !vision_workspace_->is_valid_for(num_patches)) {
    vision_workspace_ = std::make_unique<VisionWorkspace>();
    vision_workspace_->max_patches = num_patches;
    vision_workspace_->normed1 = tensor::Tensor(base::DataType::kDataTypeFp16, num_patches, hidden_size, true, alloc);
    vision_workspace_->qkv = tensor::Tensor(base::DataType::kDataTypeFp16, num_patches, 3 * hidden_size, true, alloc);
    vision_workspace_->query = tensor::Tensor(base::DataType::kDataTypeFp16, num_patches, hidden_size, true, alloc);
    vision_workspace_->key = tensor::Tensor(base::DataType::kDataTypeFp16, num_patches, hidden_size, true, alloc);
    vision_workspace_->value = tensor::Tensor(base::DataType::kDataTypeFp16, num_patches, hidden_size, true, alloc);
    vision_workspace_->attn_out = tensor::Tensor(base::DataType::kDataTypeFp16, num_patches, hidden_size, true, alloc);
    vision_workspace_->normed2 = tensor::Tensor(base::DataType::kDataTypeFp16, num_patches, hidden_size, true, alloc);
    vision_workspace_->mlp_intermediate = tensor::Tensor(base::DataType::kDataTypeFp16, num_patches, intermediate_size, true, alloc);
    vision_workspace_->proj_out = tensor::Tensor(base::DataType::kDataTypeFp16, num_patches, hidden_size, true, alloc);
    vision_workspace_->output = tensor::Tensor(base::DataType::kDataTypeFp16, num_patches, hidden_size, true, alloc);
    vision_workspace_->output2 = tensor::Tensor(base::DataType::kDataTypeFp16, num_patches, hidden_size, true, alloc);
    vision_workspace_->q_transposed = tensor::Tensor(base::DataType::kDataTypeFp16, num_heads, num_patches, head_dim, true, alloc);
    vision_workspace_->k_transposed = tensor::Tensor(base::DataType::kDataTypeFp16, num_heads, num_patches, head_dim, true, alloc);
    vision_workspace_->v_transposed = tensor::Tensor(base::DataType::kDataTypeFp16, num_heads, num_patches, head_dim, true, alloc);
    vision_workspace_->out_transposed = tensor::Tensor(base::DataType::kDataTypeFp16, num_heads, num_patches, head_dim, true, alloc);
    vision_workspace_->attn_scores = tensor::Tensor(base::DataType::kDataTypeFp16, num_heads, num_patches, num_patches, true, alloc);
  }

  auto hidden_states = vision_patch_embed(image_data);
  hidden_states = vision_add_pos_embed(hidden_states, image_data.grid_h, image_data.grid_w);
  auto [cos_cache, sin_cache] = compute_vision_rotary_emb(
      image_data.grid_h, image_data.grid_w, image_data.grid_t);

  const auto& deepstack_indexes = vl_config_.vision.deepstack_visual_indexes;
  std::vector<tensor::Tensor> deepstack_features;

  tensor::Tensor cu_seqlens(base::DataType::kDataTypeInt32, 2, true, alloc);
  cudaMemsetAsync(cu_seqlens.ptr<void>(), 0, sizeof(int32_t), cuda_config_->stream);
  int32_t num_patches_val = image_data.num_patches;
  cudaMemcpyAsync(cu_seqlens.ptr<int32_t>() + 1, &num_patches_val,
                  sizeof(int32_t), cudaMemcpyHostToDevice, cuda_config_->stream);

  tensor::Tensor* current_input = &hidden_states;
  tensor::Tensor* buffers[2] = {&vision_workspace_->output, &vision_workspace_->output2};

  for (int layer_idx = 0; layer_idx < vl_config_.vision.depth; ++layer_idx) {
    tensor::Tensor* current_output = buffers[layer_idx % 2];
    vision_transformer_block(*current_input, *current_output, layer_idx,
                              cu_seqlens, image_data.num_patches,
                              cos_cache, sin_cache, *vision_workspace_);
    auto it = std::find(deepstack_indexes.begin(), deepstack_indexes.end(), layer_idx);
    if (it != deepstack_indexes.end()) {
      int merger_idx = std::distance(deepstack_indexes.begin(), it);
      deepstack_features.push_back(vision_merger(*current_output,
          image_data.grid_h, image_data.grid_w, image_data.grid_t, true, merger_idx));
    }
    current_input = current_output;
  }

  tensor::Tensor& final_hidden = *buffers[(vl_config_.vision.depth - 1) % 2];
  auto main_output = vision_merger(final_hidden, image_data.grid_h, image_data.grid_w,
                                    image_data.grid_t, false);

  deepstack_features_.clear();
  deepstack_features_ = std::move(deepstack_features);

  cudaStreamSynchronize(cuda_config_->stream);
  auto vit_end = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Vision encoder: " << std::fixed << std::setprecision(2)
            << std::chrono::duration<double, std::milli>(vit_end - vit_start).count() << " ms";
  return main_output;
}

tensor::Tensor Qwen3VLModel::vision_patch_embed(const ImageData& image_data) const {
  int num_patches = image_data.num_patches;
  int hidden_size = vl_config_.vision.hidden_size;
  int patch_dim = 3 * vl_config_.vision.temporal_patch_size *
                  vl_config_.vision.patch_size * vl_config_.vision.patch_size;

  tensor::Tensor output(base::DataType::kDataTypeFp16, num_patches, hidden_size, true,
                        base::CUDADeviceAllocatorFactory::get_instance());

  // Use VisionPatchEmbedLayer (fused GEMM + bias) instead of direct cublas call
  vision_vl_layers_.vision_patch_embed_layer_->forward(
      image_data.pixel_values, vision_layers_->patch_embed_weight,
      vision_layers_->patch_embed_bias, output,
      num_patches, hidden_size, patch_dim, cuda_config_.get());
  return output;
}

tensor::Tensor Qwen3VLModel::vision_add_pos_embed(const tensor::Tensor& patch_embeds,
                                                   int grid_h, int grid_w) const {
  int num_patches = patch_embeds.get_dim(0);
  int hidden_size = patch_embeds.get_dim(1);
  int num_grid_per_side = static_cast<int>(std::sqrt(vl_config_.vision.num_position_embeddings));

  tensor::Tensor output(base::DataType::kDataTypeFp16, num_patches, hidden_size, true,
                        base::CUDADeviceAllocatorFactory::get_instance());
  vision_vl_layers_.pos_embed_interpolate_layer_->forward(
      patch_embeds, vision_layers_->pos_embed_weight, output,
      grid_h, grid_w, 1, num_grid_per_side,
      vl_config_.vision.spatial_merge_size, cuda_config_->stream);
  return output;
}

std::pair<tensor::Tensor, tensor::Tensor> Qwen3VLModel::compute_vision_rotary_emb(
    int grid_h, int grid_w, int grid_t) const {
  // Use VisionRotaryEmbLayer instead of inline computation
  int num_heads = vl_config_.vision.num_heads;
  int hidden_size = vl_config_.vision.hidden_size;
  int head_dim = hidden_size / num_heads;
  int num_tokens = grid_t * grid_h * grid_w;

  std::vector<uint16_t> cos_data, sin_data;
  vision_vl_layers_.vision_rotary_emb_layer_->forward(
      cos_data, sin_data, grid_h, grid_w, grid_t,
      num_heads, hidden_size, vl_config_.vision.spatial_merge_size);

  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp16, num_tokens, head_dim, true, alloc);
  tensor::Tensor sin_cache(base::DataType::kDataTypeFp16, num_tokens, head_dim, true, alloc);
  cudaMemcpyAsync(cos_cache.ptr<void>(), cos_data.data(),
                  num_tokens * head_dim * sizeof(half), cudaMemcpyHostToDevice, cuda_config_->stream);
  cudaMemcpyAsync(sin_cache.ptr<void>(), sin_data.data(),
                  num_tokens * head_dim * sizeof(half), cudaMemcpyHostToDevice, cuda_config_->stream);
  return {cos_cache, sin_cache};
}

void Qwen3VLModel::vision_transformer_block(const tensor::Tensor& hidden_states,
                                             tensor::Tensor& output_buffer,
                                             int block_idx,
                                             const tensor::Tensor& cu_seqlens,
                                             int max_seqlen,
                                             const tensor::Tensor& cos_cache,
                                             const tensor::Tensor& sin_cache,
                                             VisionWorkspace& ws) const {
  const auto& block = vision_layers_->blocks[block_idx];
  int num_tokens = hidden_states.get_dim(0);
  int hidden_size = hidden_states.get_dim(1);
  int num_heads = vl_config_.vision.num_heads;
  int head_dim = hidden_size / num_heads;

  // 1. LayerNorm
  vision_vl_layers_.layernorm_with_bias_layer_->forward(
      hidden_states, block.norm1_weight, block.norm1_bias, ws.normed1, 1e-6f, cuda_config_->stream);

  // 2. QKV projection via BatchedGemmLayer (not direct cublas)
  vision_vl_layers_.batched_gemm_layer_->forward(
      block.qkv_weight.ptr<half>(), ws.normed1.ptr<half>(), ws.qkv.ptr<half>(),
      3 * hidden_size, num_tokens, hidden_size,
      true, false, 1.0f, 0.0f,
      hidden_size, hidden_size, 3 * hidden_size, cuda_config_.get());

  vision_vl_layers_.bias_add_residual_layer_->forward(
      ws.qkv, block.qkv_bias, tensor::Tensor(), ws.qkv, cuda_config_->stream);

  // 3. Fused split + RoPE + transpose
  vision_vl_layers_.fused_split_rope_transpose_layer_->forward(
      ws.qkv, cos_cache, sin_cache,
      vision_workspace_->q_transposed, vision_workspace_->k_transposed,
      vision_workspace_->v_transposed,
      num_tokens, num_heads, head_dim, cuda_config_->stream);

  // 4. Attention
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  vision_vl_layers_.vision_attention_layer_->forward(
      vision_workspace_->q_transposed, vision_workspace_->k_transposed,
      vision_workspace_->v_transposed, ws.attn_out,
      vision_workspace_->out_transposed, vision_workspace_->attn_scores,
      num_tokens, num_heads, head_dim, scale, cuda_config_.get());

  // 5. Output projection via BatchedGemmLayer
  vision_vl_layers_.batched_gemm_layer_->forward(
      block.proj_weight.ptr<half>(), ws.attn_out.ptr<half>(), ws.proj_out.ptr<half>(),
      hidden_size, num_tokens, hidden_size,
      true, false, 1.0f, 0.0f,
      hidden_size, hidden_size, hidden_size, cuda_config_.get());

  vision_vl_layers_.bias_add_residual_layer_->forward(
      ws.proj_out, block.proj_bias, hidden_states, output_buffer, cuda_config_->stream);

  // 6. LayerNorm + MLP
  vision_vl_layers_.layernorm_with_bias_layer_->forward(
      output_buffer, block.norm2_weight, block.norm2_bias, ws.normed2, 1e-6f, cuda_config_->stream);

  vision_vl_layers_.vision_mlp_layer_->forward(
      ws.normed2, block.mlp_fc1_weight, block.mlp_fc1_bias,
      block.mlp_fc2_weight, block.mlp_fc2_bias,
      output_buffer, output_buffer, ws.mlp_intermediate, cuda_config_.get());
}

tensor::Tensor Qwen3VLModel::vision_merger(const tensor::Tensor& hidden_states,
                                            int grid_h, int grid_w, int grid_t,
                                            bool is_deepstack, int merger_idx) const {
  int merge_size = vl_config_.vision.spatial_merge_size;
  int num_patches = hidden_states.get_dim(0);
  int hidden_size = hidden_states.get_dim(1);
  int num_vision_tokens = (grid_h * grid_w * grid_t) / (merge_size * merge_size);
  int merged_hidden = hidden_size * merge_size * merge_size;
  int out_hidden = vl_config_.vision.out_hidden_size;
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();

  const Qwen3VLVisionLayers::Merger* merger = is_deepstack
      ? &vision_layers_->deepstack_mergers[merger_idx]
      : &vision_layers_->merger;

  tensor::Tensor normed(base::DataType::kDataTypeFp16, num_patches, hidden_size, true, alloc);
  vision_vl_layers_.layernorm_with_bias_layer_->forward(
      hidden_states, merger->norm_weight, merger->norm_bias, normed, 1e-6f, cuda_config_->stream);

  tensor::Tensor merged(base::DataType::kDataTypeFp16, num_vision_tokens, merged_hidden, true, alloc);
  vision_vl_layers_.spatial_merge_layer_->forward(
      normed, merged, grid_t, grid_h, grid_w, hidden_size, merge_size, cuda_config_->stream);

  tensor::Tensor output(base::DataType::kDataTypeFp16, num_vision_tokens, out_hidden, true, alloc);
  tensor::Tensor intermediate(base::DataType::kDataTypeFp16, num_vision_tokens, merged_hidden, true, alloc);
  vision_vl_layers_.vision_merger_mlp_layer_->forward(
      merged, merger->fc1_weight, merger->fc1_bias,
      merger->fc2_weight, merger->fc2_bias, output, intermediate, cuda_config_.get());
  return output;
}

// ============================================================================
// Multimodal Embedding
// ============================================================================

tensor::Tensor Qwen3VLModel::prepare_multimodal_embeddings(
    const std::vector<int>& tokens, const ImageData* image_data) const {
  auto embed_out = embedding(tokens);
  if (!image_data || image_data->pixel_values.is_empty()) return embed_out.input_embeddings;

  auto visual_embeds = encode_image(*image_data);
  int num_vision_tokens = image_data->num_vision_tokens;
  int dim = config_->dim_;
  int image_token_id = vl_config_.special_tokens.image_token_id;

  int image_token_pos = -1;
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (tokens[i] == image_token_id) { image_token_pos = static_cast<int>(i); break; }
  }
  if (image_token_pos < 0) {
    LOG(WARNING) << "No image token found, using text-only embeddings";
    return embed_out.input_embeddings;
  }

  visual_pos_start_ = image_token_pos;
  visual_pos_end_ = image_token_pos + num_vision_tokens;
  int new_seq_len = static_cast<int>(tokens.size()) - 1 + num_vision_tokens;

  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor multimodal_embeds(base::DataType::kDataTypeFp16, new_seq_len, dim, true, alloc);

  vision_vl_layers_.fused_multimodal_embed_layer_->forward(
      embed_out.input_embeddings, visual_embeds, multimodal_embeds,
      image_token_pos, num_vision_tokens, static_cast<int>(tokens.size()), dim,
      cuda_config_->stream);

  // Generate M-RoPE positions + upload to GPU (fused into single layer call)
  int merged_grid_h = image_data->grid_h / vl_config_.vision.spatial_merge_size;
  int merged_grid_w = image_data->grid_w / vl_config_.vision.spatial_merge_size;
  int max_text_pos = 0;
  vision_vl_layers_.generate_mrope_positions_layer_->forward(
      tokens, image_token_pos, num_vision_tokens,
      merged_grid_h, merged_grid_w,
      mrope_pos_t_, mrope_pos_h_, mrope_pos_w_,
      max_text_pos, mrope_pos_t_gpu_, mrope_pos_h_gpu_, mrope_pos_w_gpu_,
      cuda_config_->stream);
  mrope_max_text_pos_ = max_text_pos;

  LOG(INFO) << "Multimodal embeddings: text=" << tokens.size()
            << ", vision=" << num_vision_tokens << ", total=" << new_seq_len;
  return multimodal_embeds;
}

// ============================================================================
// Prefill and Decode
// ============================================================================

base::Status Qwen3VLModel::prefill(const tensor::Tensor& input_embeddings,
                                    int32_t seq_len, int32_t start_pos) const {
  LOG(INFO) << "Batched Prefill: seq_len=" << seq_len << ", start_pos=" << start_pos;

  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  size_t elem_size = sizeof(uint16_t);
  int dim = config_->dim_;
  int kv_dim = config_->kv_dim_;
  int hidden_dim = config_->hidden_dim_;

  int num_deepstack_layers = std::min(static_cast<int>(deepstack_features_.size()), config_->layer_num_);

  tensor::Tensor hidden_buf0(base::DataType::kDataTypeFp16, seq_len, dim, true, alloc);
  tensor::Tensor hidden_buf1(base::DataType::kDataTypeFp16, seq_len, dim, true, alloc);
  tensor::Tensor rms_out(base::DataType::kDataTypeFp16, seq_len, dim, true, alloc);
  tensor::Tensor query_out(base::DataType::kDataTypeFp16, seq_len, dim, true, alloc);
  tensor::Tensor key_out(base::DataType::kDataTypeFp16, seq_len, kv_dim, true, alloc);
  tensor::Tensor value_out(base::DataType::kDataTypeFp16, seq_len, kv_dim, true, alloc);
  tensor::Tensor mha_out(base::DataType::kDataTypeFp16, seq_len, dim, true, alloc);
  tensor::Tensor ffn_norm_out(base::DataType::kDataTypeFp16, seq_len, dim, true, alloc);
  tensor::Tensor w1_out(base::DataType::kDataTypeFp16, seq_len, hidden_dim, true, alloc);
  tensor::Tensor w3_out(base::DataType::kDataTypeFp16, seq_len, hidden_dim, true, alloc);
  tensor::Tensor w2_out(base::DataType::kDataTypeFp16, seq_len, dim, true, alloc);

  int kv_len = start_pos + seq_len;
  int64_t score_buf_size = (int64_t)config_->head_num_ * kv_len * seq_len;
  half* score_buf = nullptr;
  cudaMalloc(&score_buf, score_buf_size * sizeof(half));

  const int head_num = config_->head_num_;
  const int kv_mul = config_->kv_mul_;
  const int head_size_val = config_->head_size_;
  const size_t ptrs_per_step = 3 * head_num;
  const size_t ptrs_per_layer = 2 * ptrs_per_step;
  const size_t total_ptrs = config_->layer_num_ * ptrs_per_layer;

  half** d_ptr_buf = nullptr;
  cudaMalloc(&d_ptr_buf, total_ptrs * sizeof(half*));

  {
    half** h_ptr_buf = nullptr;
    cudaMallocHost(&h_ptr_buf, total_ptrs * sizeof(half*));
    half* Q = const_cast<half*>(query_out.ptr<half>());
    half* K_base = const_cast<half*>(get_buffer(ModelBufferType::kKeyCache).ptr<half>());
    half* V_base = const_cast<half*>(get_buffer(ModelBufferType::kValueCache).ptr<half>());

    for (int l = 0; l < config_->layer_num_; l++) {
      half** step1 = h_ptr_buf + l * ptrs_per_layer;
      half** step3 = step1 + ptrs_per_step;
      int64_t layer_offset = (int64_t)l * config_->seq_len_ * kv_dim;
      half* K = K_base + layer_offset;
      half* V = V_base + layer_offset;
      for (int h = 0; h < head_num; h++) {
        step1[h]              = K + (h / kv_mul) * head_size_val;
        step1[head_num + h]   = Q + h * head_size_val;
        step1[2*head_num + h] = score_buf + (int64_t)h * kv_len * seq_len;
        step3[h]              = V + (h / kv_mul) * head_size_val;
        step3[head_num + h]   = score_buf + (int64_t)h * kv_len * seq_len;
        step3[2*head_num + h] = Q + h * head_size_val;
      }
    }
    cudaMemcpyAsync(d_ptr_buf, h_ptr_buf, total_ptrs * sizeof(half*),
                    cudaMemcpyHostToDevice, cuda_config_->stream);
    cudaStreamSynchronize(cuda_config_->stream);
    cudaFreeHost(h_ptr_buf);
  }

  tensor::Tensor* hidden_buffers[2] = {&hidden_buf0, &hidden_buf1};
  tensor::Tensor* final_hidden = nullptr;

  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    const tensor::Tensor* layer_input;
    tensor::Tensor* layer_output;
    if (layer_idx == 0) {
      layer_input = &input_embeddings;
      layer_output = hidden_buffers[0];
    } else {
      layer_input = hidden_buffers[(layer_idx - 1) % 2];
      layer_output = hidden_buffers[layer_idx % 2];
    }

    batched_attention_rms(layer_idx, *layer_input, rms_out, seq_len);
    batched_attention_qkv(layer_idx, rms_out, query_out, key_out, value_out, seq_len, start_pos);
    batched_attention_mha(layer_idx, query_out, mha_out, seq_len, start_pos,
                          score_buf, d_ptr_buf + layer_idx * ptrs_per_layer);
    STATUS_CHECK(qwen_layers_->batched_add_layer_->forward(*layer_input, mha_out, *layer_output));
    batched_feed_forward_optimized(layer_idx, *layer_output, ffn_norm_out, w1_out, w3_out, w2_out, seq_len);

    if (layer_idx < num_deepstack_layers && visual_pos_start_ >= 0) {
      half* hidden_ptr = layer_output->ptr<half>() + visual_pos_start_ * dim;
      const half* ds_ptr = deepstack_features_[layer_idx].ptr<half>();
      int num_visual_tokens = visual_pos_end_ - visual_pos_start_;
      STATUS_CHECK(qwen_layers_->batched_add_layer_->forward_raw(
          hidden_ptr, ds_ptr, hidden_ptr, num_visual_tokens * dim));
    }
    final_hidden = layer_output;
  }

  if (score_buf) cudaFree(score_buf);
  if (d_ptr_buf) cudaFree(d_ptr_buf);

  tensor::Tensor decode_input = get_buffer(ModelBufferType::kDecodeInput);
  void* last_token_ptr = final_hidden->ptr<uint8_t>() + (seq_len - 1) * dim * elem_size;
  cudaMemcpyAsync(decode_input.ptr<void>(), last_token_ptr,
                  dim * elem_size, cudaMemcpyDeviceToDevice, cuda_config_->stream);
  prefill_seq_len_ = seq_len;
  cudaStreamSynchronize(cuda_config_->stream);
  LOG(INFO) << "Batched Prefill complete";
  return base::error::Success();
}

int Qwen3VLModel::sample_first_token() const {
  tensor::Tensor input = get_buffer(ModelBufferType::kDecodeInput);
  tensor::Tensor pos_tensor = get_buffer(ModelBufferType::kInputPos);
  cls_logits(input);
  if (cuda_config_ && cuda_config_->stream) cudaStreamSynchronize(cuda_config_->stream);
  return post_processing(pos_tensor, false);
}

base::Status Qwen3VLModel::decode_step_optimized(int32_t pos, int& next) const {
  bool use_graph = cuda_config_ && cuda_config_->use_cuda_graph && cuda_config_->graph_context;

  if (use_graph) {
    auto& graph_ctx = cuda_config_->graph_context;
    auto& graph = graph_ctx->decode_graph;
    tensor::Tensor decode_input = get_buffer(ModelBufferType::kDecodeInput);
    tensor::Tensor pos_tensor_gpu = get_buffer(ModelBufferType::kInputPosGPU);
    tensor::Tensor kv_cache_pos_gpu = get_buffer(ModelBufferType::kKVCachePosGPU);
    tensor::Tensor pos_pinned = get_buffer(ModelBufferType::kInputPosPinned);
    tensor::Tensor kv_cache_pos_pinned = get_buffer(ModelBufferType::kKVCachePosPinned);
    tensor::Tensor argmax_output = get_buffer(ModelBufferType::kArgmaxOutput);
    tensor::Tensor argmax_pinned = get_buffer(ModelBufferType::kArgmaxOutputPinned);

    int32_t text_pos = mrope_max_text_pos_ + (pos - prefill_seq_len_) + 1;
    *const_cast<int32_t*>(pos_pinned.ptr<int32_t>()) = text_pos;
    cudaMemcpyAsync(const_cast<int32_t*>(pos_tensor_gpu.ptr<int32_t>()),
                    pos_pinned.ptr<int32_t>(), sizeof(int32_t),
                    cudaMemcpyHostToDevice, cuda_config_->stream);
    *const_cast<int32_t*>(kv_cache_pos_pinned.ptr<int32_t>()) = pos;
    cudaMemcpyAsync(const_cast<int32_t*>(kv_cache_pos_gpu.ptr<int32_t>()),
                    kv_cache_pos_pinned.ptr<int32_t>(), sizeof(int32_t),
                    cudaMemcpyHostToDevice, cuda_config_->stream);

    bool need_capture = graph_ctx->needs_recapture || !graph->is_valid();
    if (need_capture && !graph->is_disabled()) {
      cudaStreamSynchronize(cuda_config_->stream);
      if (graph->begin_capture(cuda_config_->stream)) {
        for (int32_t l = 0; l < config_->layer_num_; ++l) {
          attention_rms(l, decode_input);
          attention_qkv_with_graph(l, pos_tensor_gpu, kv_cache_pos_gpu);
          attention_mha_with_graph(l, pos_tensor_gpu, kv_cache_pos_gpu);
          feed_forward(l, decode_input);
        }
        cls_logits(decode_input);
        if (graph->end_capture(cuda_config_->stream)) {
          graph_ctx->graph_recaptures++;
          graph_ctx->needs_recapture = false;
        }
      }
    }

    if (graph->is_valid() && graph->launch(cuda_config_->stream)) {
      graph_ctx->graph_launches++;
      tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
      auto* argmax_sampler = dynamic_cast<sampler::ArgmaxSampler*>(sampler_.get());
      if (argmax_sampler) {
        argmax_sampler->sample_prealloc(
            forward_output.ptr<float>(), forward_output.size(),
            reinterpret_cast<size_t*>(const_cast<int32_t*>(argmax_output.ptr<int32_t>())),
            reinterpret_cast<size_t*>(const_cast<int32_t*>(argmax_pinned.ptr<int32_t>())),
            cuda_config_->stream);
        cudaStreamSynchronize(cuda_config_->stream);
        next = static_cast<int32_t>(*reinterpret_cast<size_t*>(
            const_cast<int32_t*>(argmax_pinned.ptr<int32_t>())));
      } else {
        cudaStreamSynchronize(cuda_config_->stream);
        next = post_processing(get_buffer(ModelBufferType::kInputPos), false);
      }
      return base::error::Success();
    }
    if (graph->is_valid()) graph_ctx->invalidate();
  }

  // Normal execution
  tensor::Tensor pos_tensor = get_buffer(ModelBufferType::kInputPos);
  pos_tensor.index<int32_t>(0) = pos;
  tensor::Tensor decode_input = get_buffer(ModelBufferType::kDecodeInput);

  for (int32_t l = 0; l < config_->layer_num_; ++l) {
    attention_rms(l, decode_input);
    attention_qkv(l, pos_tensor);
    attention_mha(l, pos_tensor);
    feed_forward(l, decode_input);
  }
  cls_logits(decode_input);
  if (cuda_config_ && cuda_config_->stream) cudaStreamSynchronize(cuda_config_->stream);
  next = post_processing(pos_tensor, false);
  return base::error::Success();
}

// ============================================================================
// Utility Methods
// ============================================================================

base::Status Qwen3VLModel::predict(const tensor::Tensor& input,
                                    const tensor::Tensor& pos_tensor,
                                    bool is_prompt, int& next) const {
  return forward(input, pos_tensor, next);
}

base::Status Qwen3VLModel::forward(const tensor::Tensor& input,
                                    const tensor::Tensor& pos_tensor,
                                    int& next) const {
  return base::error::Success();
}

op::EmbeddingOutput Qwen3VLModel::embedding(const std::vector<int>& tokens) const {
  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
  if (input_tokens.size() != tokens.size()) {
    input_tokens.reshape({static_cast<int32_t>(tokens.size())});
    input_embeddings.reshape({static_cast<int32_t>(tokens.size()), config_->dim_});
  }
  for (int32_t i = 0; i < static_cast<int32_t>(tokens.size()); ++i)
    input_tokens.index<int32_t>(i) = tokens.at(i);
  auto input_token_num = tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
  if (qwen_layers_->embedding_layer_)
    STATUS_CHECK(qwen_layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));
  return op::EmbeddingOutput(input_tokens, input_embeddings, input_token_num);
}

void Qwen3VLModel::embedding_to_decode_input(int token_id) const {
  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto decode_input = get_buffer(ModelBufferType::kDecodeInput);
  if (input_tokens.size() != 1) input_tokens.reshape({1});
  if (decode_input.dims_size() != 2 || decode_input.get_dim(0) != 1)
    decode_input.reshape({1, config_->dim_});
  input_tokens.index<int32_t>(0) = token_id;
  auto input_token_num = tensor::Tensor(base::DataType::kDataTypeInt32, 1);
  if (qwen_layers_->embedding_layer_)
    STATUS_CHECK(qwen_layers_->embedding_layer_->forward(input_tokens, input_token_num, decode_input));
  decode_input.reshape({config_->dim_});
}

void Qwen3VLModel::enable_cuda_graph(bool enable) {
  if (cuda_config_) {
    cuda_config_->use_cuda_graph = enable;
    if (enable && !cuda_config_->graph_context) {
      cuda_config_->graph_context = std::make_unique<base::CudaGraphContext>();
      cuda_config_->graph_context->needs_recapture = true;
    }
  }
}

// ============================================================================
// Batched Operations for Prefill (using BatchedGemmLayer)
// ============================================================================

void Qwen3VLModel::batched_attention_rms(int32_t layer_idx, const tensor::Tensor& input,
                                          const tensor::Tensor& output, int32_t seq_len) const {
  STATUS_CHECK(qwen_layers_->rmsnorm_layers_.at(layer_idx)->forward(input, output));
}

void Qwen3VLModel::batched_attention_qkv(int32_t layer_idx, const tensor::Tensor& rms_out,
                                          const tensor::Tensor& query_out,
                                          const tensor::Tensor& key_out,
                                          const tensor::Tensor& value_out,
                                          int32_t seq_len, int32_t start_pos) const {
  base::DataType activation_dtype = rms_out.data_type();
  size_t elem_size = (activation_dtype == base::DataType::kDataTypeFp16) ? sizeof(uint16_t) : sizeof(float);

  auto wq = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wq_layers_.at(layer_idx));
  auto wk = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wk_layers_.at(layer_idx));
  auto wv = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wv_layers_.at(layer_idx));

  // Q/K/V projections via BatchedGemmLayer
  vision_vl_layers_.batched_gemm_layer_->forward(
      wq->get_weight(0).ptr<half>(), rms_out.ptr<half>(), const_cast<half*>(query_out.ptr<half>()),
      config_->dim_, seq_len, config_->dim_, true, false, 1.0f, 0.0f,
      config_->dim_, config_->dim_, config_->dim_, cuda_config_.get());
  vision_vl_layers_.batched_gemm_layer_->forward(
      wk->get_weight(0).ptr<half>(), rms_out.ptr<half>(), const_cast<half*>(key_out.ptr<half>()),
      config_->kv_dim_, seq_len, config_->dim_, true, false, 1.0f, 0.0f,
      config_->dim_, config_->dim_, config_->kv_dim_, cuda_config_.get());
  vision_vl_layers_.batched_gemm_layer_->forward(
      wv->get_weight(0).ptr<half>(), rms_out.ptr<half>(), const_cast<half*>(value_out.ptr<half>()),
      config_->kv_dim_, seq_len, config_->dim_, true, false, 1.0f, 0.0f,
      config_->dim_, config_->dim_, config_->kv_dim_, cuda_config_.get());

  // q_norm / k_norm
  const auto& q_norm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx + 2 * config_->layer_num_ + 1);
  const auto& k_norm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx + 3 * config_->layer_num_ + 1);

  auto q_buffer = std::make_shared<base::Buffer>(
      seq_len * config_->dim_ * elem_size, nullptr, const_cast<void*>(query_out.get_buffer()->ptr()), true);
  tensor::Tensor q_reshaped(activation_dtype, seq_len * config_->head_num_, config_->head_size_, false, nullptr, nullptr);
  q_reshaped.assign(q_buffer);
  q_reshaped.set_device_type(base::DeviceType::kDeviceCUDA);

  auto k_buffer = std::make_shared<base::Buffer>(
      seq_len * config_->kv_dim_ * elem_size, nullptr, const_cast<void*>(key_out.get_buffer()->ptr()), true);
  tensor::Tensor k_reshaped(activation_dtype, seq_len * config_->kv_head_num_, config_->head_size_, false, nullptr, nullptr);
  k_reshaped.assign(k_buffer);
  k_reshaped.set_device_type(base::DeviceType::kDeviceCUDA);

  const auto& q_weight = std::dynamic_pointer_cast<op::RmsNormLayer>(q_norm_layer)->get_weight(0);
  const auto& k_weight = std::dynamic_pointer_cast<op::RmsNormLayer>(k_norm_layer)->get_weight(0);
  qwen_layers_->rmsnorm_dim_layer_->forward(q_reshaped, q_weight, q_reshaped, config_->head_size_);
  qwen_layers_->rmsnorm_dim_layer_->forward(k_reshaped, k_weight, k_reshaped, config_->head_size_);

  // Batched M-RoPE
  const auto& section = vl_config_.text.mrope_section;
  qwen_layers_->batched_mrope_layer_->forward(
      seq_len, config_->dim_, config_->kv_dim_, config_->head_size_,
      section[0], section[1], section[2],
      mrope_pos_t_gpu_ + start_pos, mrope_pos_h_gpu_ + start_pos, mrope_pos_w_gpu_ + start_pos,
      query_out, key_out,
      get_buffer(ModelBufferType::kSinCache), get_buffer(ModelBufferType::kCosCache));

  // Fused KV cache update
  qwen_layers_->fused_kv_cache_update_layer_->forward(
      key_out, value_out, get_buffer(ModelBufferType::kKeyCache), get_buffer(ModelBufferType::kValueCache),
      layer_idx, start_pos, seq_len, config_->kv_dim_, config_->seq_len_);
}

void Qwen3VLModel::batched_attention_mha(int32_t layer_idx, const tensor::Tensor& query,
                                          tensor::Tensor& mha_out,
                                          int32_t seq_len, int32_t start_pos,
                                          half* score_buf, half** d_ptr_buf) const {
  const int head_num = config_->head_num_;
  const int head_size = config_->head_size_;
  const int dim = config_->dim_;
  const int kv_dim = config_->kv_dim_;
  const int kv_len = start_pos + seq_len;
  const size_t ptrs_per_step = 3 * head_num;
  const float scale_f = 1.0f / sqrtf((float)head_size);

  // Step 1: Q·K^T via BatchedGemmLayer
  vision_vl_layers_.batched_gemm_layer_->forward_batched(
      (const half**)d_ptr_buf, (const half**)(d_ptr_buf + head_num),
      d_ptr_buf + 2 * head_num,
      kv_len, seq_len, head_size, true, false, scale_f, 0.0f,
      kv_dim, dim, kv_len, head_num, cuda_config_.get());

  // Step 2: Causal softmax
  vision_vl_layers_.causal_softmax_layer_->forward(
      score_buf, head_num, seq_len, kv_len, start_pos, cuda_config_->stream);

  // Step 3: Attn·V via BatchedGemmLayer
  half** step3 = d_ptr_buf + ptrs_per_step;
  vision_vl_layers_.batched_gemm_layer_->forward_batched(
      (const half**)step3, (const half**)(step3 + head_num),
      step3 + 2 * head_num,
      head_size, seq_len, kv_len, false, false, 1.0f, 0.0f,
      kv_dim, kv_len, dim, head_num, cuda_config_.get());

  // Step 4: WO projection via BatchedGemmLayer
  auto wo = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wo_layers_.at(layer_idx));
  vision_vl_layers_.batched_gemm_layer_->forward(
      wo->get_weight(0).ptr<half>(), const_cast<half*>(query.ptr<half>()), mha_out.ptr<half>(),
      dim, seq_len, dim, true, false, 1.0f, 0.0f, dim, dim, dim, cuda_config_.get());
}

void Qwen3VLModel::batched_feed_forward_optimized(
    int32_t layer_idx, const tensor::Tensor& input,
    tensor::Tensor& ffn_norm_out, tensor::Tensor& w1_out,
    tensor::Tensor& w3_out, tensor::Tensor& w2_out, int32_t seq_len) const {
  STATUS_CHECK(qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_)->forward(input, ffn_norm_out));

  auto w1 = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->w1_layers_.at(layer_idx));
  auto w3 = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->w3_layers_.at(layer_idx));
  int32_t hidden_dim = config_->hidden_dim_;

  // W1/W3 via BatchedGemmLayer
  vision_vl_layers_.batched_gemm_layer_->forward(
      w1->get_weight(0).ptr<half>(), ffn_norm_out.ptr<half>(), w1_out.ptr<half>(),
      hidden_dim, seq_len, config_->dim_, true, false, 1.0f, 0.0f,
      config_->dim_, config_->dim_, hidden_dim, cuda_config_.get());
  vision_vl_layers_.batched_gemm_layer_->forward(
      w3->get_weight(0).ptr<half>(), ffn_norm_out.ptr<half>(), w3_out.ptr<half>(),
      hidden_dim, seq_len, config_->dim_, true, false, 1.0f, 0.0f,
      config_->dim_, config_->dim_, hidden_dim, cuda_config_.get());

  STATUS_CHECK(qwen_layers_->batched_swiglu_layer_->forward(w1_out, w3_out, w1_out));

  // W2 via BatchedGemmLayer
  auto w2 = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->w2_layers_.at(layer_idx));
  vision_vl_layers_.batched_gemm_layer_->forward(
      w2->get_weight(0).ptr<half>(), w1_out.ptr<half>(), w2_out.ptr<half>(),
      config_->dim_, seq_len, hidden_dim, true, false, 1.0f, 0.0f,
      hidden_dim, hidden_dim, config_->dim_, cuda_config_.get());

  STATUS_CHECK(qwen_layers_->batched_add_layer_->forward(input, w2_out, input));
}

bool Qwen3VLModel::is_cuda_graph_enabled() const {
  return cuda_config_ && cuda_config_->use_cuda_graph;
}

void Qwen3VLModel::invalidate_cuda_graph() {
  if (cuda_config_) cuda_config_->invalidate_graph();
}

void Qwen3VLModel::clear_kv_cache() {
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor value_cache = get_buffer(ModelBufferType::kValueCache);
  if (device_type_ == base::DeviceType::kDeviceCUDA && cuda_config_) {
    cudaMemsetAsync(const_cast<void*>(key_cache.get_buffer()->ptr()), 0,
                    key_cache.size() * sizeof(uint16_t), cuda_config_->stream);
    cudaMemsetAsync(const_cast<void*>(value_cache.get_buffer()->ptr()), 0,
                    value_cache.size() * sizeof(uint16_t), cuda_config_->stream);
    cudaStreamSynchronize(cuda_config_->stream);
  }
}

void Qwen3VLModel::set_attention_type(base::AttentionType type) {
  Model::set_attention_type(type);
  if (qwen_layers_) {
    if (qwen_layers_->flash_attention_decode_layer_)
      qwen_layers_->flash_attention_decode_layer_->set_attention_type(type);
    if (qwen_layers_->flash_attention_prefill_layer_)
      qwen_layers_->flash_attention_prefill_layer_->set_attention_type(type);
    if (qwen_layers_->flash_attention_decode_gpu_pos_layer_)
      qwen_layers_->flash_attention_decode_gpu_pos_layer_->set_attention_type(type);
  }
}

}  // namespace model

#endif  // QWEN3_VL_SUPPORT
