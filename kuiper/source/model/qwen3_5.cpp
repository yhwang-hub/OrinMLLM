/**
 * @file qwen3_5.cpp
 * @brief Qwen3.5-9B Inference: Decode, Prefill, and Sampling
 *
 * Hybrid architecture: 24 GDN (Gated Delta Net) linear attention layers +
 * 8 full attention layers with output gating. Reuses Qwen3-VL's ViT pipeline.
 *
 * Base functions (init, load, memory, construction) are in qwen3_5_base.cpp.
 */

#ifdef QWEN3_VL_SUPPORT
#include "model/qwen3_5.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <cmath>
#include <chrono>

#include "op/matmul.h"
#include "op/rmsnorm.h"
#include "op/flash_attention.h"
#include "op/vision_layers.h"
#include "op/fused_ffn.h"
#include "op/add.h"
#include "op/gdn_layers.h"

#include "sampler/argmax_sampler.h"

using namespace base;

namespace model {

// ==========================================================================
// Decode: Full Attention Layer
// ==========================================================================

void Qwen35Model::full_attn_decode(int32_t layer_idx, const tensor::Tensor& input) const {
  auto stream = cuda_config_->stream;
  int type_idx = full_attn_type_idx(layer_idx);
  int n_layers = q35_config_.num_hidden_layers;
  int dim = q35_config_.hidden_size;
  int head_dim = q35_config_.head_dim;
  int n_heads = q35_config_.num_attention_heads;
  int n_kv_heads = q35_config_.num_key_value_heads;
  int kv_dim = q35_config_.kv_dim();
  int q_dim = q35_config_.q_dim();
  int q_gate_dim = q35_config_.q_gate_dim();
  int partial_dim = q35_config_.partial_rope_dim();

  // 1. Attention RMSNorm
  tensor::Tensor rms_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  qwen_layers_->rmsnorm_layers_[layer_idx]->forward(input, rms_output);

  // 2-3. Fused Q+K+V projection (single kernel launch)
  tensor::Tensor query_gate_buf = get_buffer(ModelBufferType::kQuery);  // [q_gate_dim=8192]
  tensor::Tensor temp_key = get_buffer(ModelBufferType::kTempKey);
  tensor::Tensor temp_value = get_buffer(ModelBufferType::kTempValue);

  auto wq_param = std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->wq_layers_[type_idx]);
  auto wk_param = std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->wk_layers_[type_idx]);
  auto wv_param = std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->wv_layers_[type_idx]);

  fused_qkv_gemv_layer_->forward(
      rms_output.ptr<half>(),
      wq_param->get_weight(0).ptr<half>(),
      wk_param->get_weight(0).ptr<half>(),
      wv_param->get_weight(0).ptr<half>(),
      query_gate_buf.ptr<half>(), temp_key.ptr<half>(), temp_value.ptr<half>(),
      dim, q_gate_dim, kv_dim);

  // Deinterleave Q and Gate per-head
  deinterleave_q_gate_layer_->forward(
      query_gate_buf.ptr<half>(), full_attn_q_.ptr<half>(), full_attn_gate_.ptr<half>(),
      n_heads, head_dim, 1);

  half* query_ptr = full_attn_q_.ptr<half>();
  half* gate_ptr = full_attn_gate_.ptr<half>();

  // 4. Q norm, K norm (per-head RMSNorm)
  int q_norm_idx = 2 * n_layers + 1 + type_idx;
  int k_norm_idx = 2 * n_layers + 1 + (int)q35_config_.full_attn_layer_indices.size() + type_idx;

  auto& q_norm_layer = qwen_layers_->rmsnorm_layers_[q_norm_idx];
  auto& k_norm_layer = qwen_layers_->rmsnorm_layers_[k_norm_idx];

  // Reshape to [n_heads, head_dim] for per-head norm then reshape back
  tensor::Tensor query_view(base::DataType::kDataTypeFp16, q_dim, false, nullptr, query_ptr);
  query_view.set_device_type(DeviceType::kDeviceCUDA);
  query_view.reshape({n_heads, head_dim});
  q_norm_layer->forward(query_view, query_view);
  query_view.reshape({q_dim});

  temp_key.reshape({n_kv_heads, head_dim});
  k_norm_layer->forward(temp_key, temp_key);
  temp_key.reshape({kv_dim});

  // 5. Partial M-RoPE (interleaved)
  tensor::Tensor pos_tensor = get_buffer(ModelBufferType::kInputPos);
  int pos = pos_tensor.index<int32_t>(0);
  int text_pos = mrope_max_text_pos_ + (pos - prefill_seq_len_) + 1;

  tensor::Tensor sin_cache = get_buffer(ModelBufferType::kSinCache);
  tensor::Tensor cos_cache = get_buffer(ModelBufferType::kCosCache);

  partial_mrope_layer_->forward(
      query_ptr, temp_key.ptr<half>(),
      sin_cache.ptr<float>(), cos_cache.ptr<float>(),
      text_pos, text_pos, text_pos,
      mrope_sections_cpu_, partial_dim,
      head_dim, n_heads, n_kv_heads);

  // 6. Write K, V to cache
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  size_t kv_offset = (static_cast<size_t>(type_idx) * config_->seq_len_ + pos) * kv_dim;
  cudaMemcpyAsync(key_cache.ptr<half>() + kv_offset, temp_key.ptr<half>(),
                  kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(val_cache.ptr<half>() + kv_offset, temp_value.ptr<half>(),
                  kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);

  // 7. Flash Attention Decode
  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);

  qwen_layers_->flash_attention_decode_layer_->forward(
      pos, n_heads, n_kv_heads, head_dim, config_->kv_mul_,
      type_idx, config_->seq_len_, kv_dim,
      query_view, mha_output, key_cache, val_cache);

  // 8. Apply sigmoid gate: mha_output *= sigmoid(gate)
  apply_sigmoid_gate_layer_->forward(mha_output.ptr<half>(), gate_ptr, q_dim);

  // 9. Output projection
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  qwen_layers_->wo_layers_[type_idx]->forward(mha_output, attn_output);
}

// ==========================================================================
// Decode: Full Attention Layer (CUDA Graph compatible)
// ==========================================================================

void Qwen35Model::full_attn_decode_graph(int32_t layer_idx, const tensor::Tensor& input,
                                         const int32_t* rope_pos_gpu, const int32_t* kv_pos_gpu) const {
  int type_idx = full_attn_type_idx(layer_idx);
  int n_layers = q35_config_.num_hidden_layers;
  int dim = q35_config_.hidden_size;
  int head_dim = q35_config_.head_dim;
  int n_heads = q35_config_.num_attention_heads;
  int n_kv_heads = q35_config_.num_key_value_heads;
  int kv_dim = q35_config_.kv_dim();
  int q_dim = q35_config_.q_dim();
  int partial_dim = q35_config_.partial_rope_dim();

  // 1. Attention RMSNorm
  tensor::Tensor rms_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  qwen_layers_->rmsnorm_layers_[layer_idx]->forward(input, rms_output);

  // 2-3. Fused Q+K+V projection (single kernel launch)
  tensor::Tensor query_gate_buf = get_buffer(ModelBufferType::kQuery);
  tensor::Tensor temp_key = get_buffer(ModelBufferType::kTempKey);
  tensor::Tensor temp_value = get_buffer(ModelBufferType::kTempValue);

  fused_qkv_gemv_layer_->forward(
      rms_output.ptr<half>(),
      std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->wq_layers_[type_idx])->get_weight(0).ptr<half>(),
      std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->wk_layers_[type_idx])->get_weight(0).ptr<half>(),
      std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->wv_layers_[type_idx])->get_weight(0).ptr<half>(),
      query_gate_buf.ptr<half>(), temp_key.ptr<half>(), temp_value.ptr<half>(),
      dim, q35_config_.q_gate_dim(), kv_dim);

  // Deinterleave Q and Gate
  deinterleave_q_gate_layer_->forward(
      query_gate_buf.ptr<half>(), full_attn_q_.ptr<half>(), full_attn_gate_.ptr<half>(),
      n_heads, head_dim, 1);

  half* query_ptr = full_attn_q_.ptr<half>();
  half* gate_ptr = full_attn_gate_.ptr<half>();

  // 4. Q norm, K norm
  int q_norm_idx = 2 * n_layers + 1 + type_idx;
  int k_norm_idx = 2 * n_layers + 1 + (int)q35_config_.full_attn_layer_indices.size() + type_idx;
  tensor::Tensor query_view(base::DataType::kDataTypeFp16, q_dim, false, nullptr, query_ptr);
  query_view.set_device_type(DeviceType::kDeviceCUDA);
  query_view.reshape({n_heads, head_dim});
  qwen_layers_->rmsnorm_layers_[q_norm_idx]->forward(query_view, query_view);
  query_view.reshape({q_dim});

  temp_key.reshape({n_kv_heads, head_dim});
  qwen_layers_->rmsnorm_layers_[k_norm_idx]->forward(temp_key, temp_key);
  temp_key.reshape({kv_dim});

  // 5. Partial M-RoPE with GPU position
  tensor::Tensor sin_cache = get_buffer(ModelBufferType::kSinCache);
  tensor::Tensor cos_cache = get_buffer(ModelBufferType::kCosCache);
  partial_mrope_layer_->forward_gpu_pos(
      query_ptr, temp_key.ptr<half>(),
      sin_cache.ptr<float>(), cos_cache.ptr<float>(),
      rope_pos_gpu, mrope_sections_cpu_, partial_dim,
      head_dim, n_heads, n_kv_heads);

  // 6. Write K, V to cache using GPU position
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
  kv_cache_write_gpu_pos_layer_->forward(key_cache.ptr<half>(), temp_key.ptr<half>(),
      kv_pos_gpu, kv_dim, type_idx, config_->seq_len_);
  kv_cache_write_gpu_pos_layer_->forward(val_cache.ptr<half>(), temp_value.ptr<half>(),
      kv_pos_gpu, kv_dim, type_idx, config_->seq_len_);

  // 7. Flash Attention Decode with GPU position
  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  qwen_layers_->flash_attention_decode_gpu_pos_layer_->forward(
      kv_pos_gpu, n_heads, n_kv_heads, head_dim, config_->kv_mul_,
      type_idx, config_->seq_len_, kv_dim,
      query_view, mha_output, key_cache, val_cache);

  // 8. Apply sigmoid gate
  apply_sigmoid_gate_layer_->forward(mha_output.ptr<half>(), gate_ptr, q_dim);

  // 9. Output projection
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  qwen_layers_->wo_layers_[type_idx]->forward(mha_output, attn_output);
}

// ==========================================================================
// Decode: Linear Attention Layer (GDN)
// ==========================================================================

void Qwen35Model::linear_attn_decode(int32_t layer_idx, const tensor::Tensor& input) const {
  int type_idx = linear_attn_type_idx(layer_idx);
  int n_layers = q35_config_.num_hidden_layers;
  int dim = q35_config_.hidden_size;
  int conv_dim = q35_config_.conv_dim();
  int n_k_heads = q35_config_.linear_num_key_heads;
  int n_v_heads = q35_config_.linear_num_value_heads;
  int k_head_dim = q35_config_.linear_key_head_dim;
  int v_head_dim = q35_config_.linear_value_head_dim;
  int ks = q35_config_.linear_conv_kernel_dim;
  
  auto& la = linear_attn_weights_->layers[type_idx];
  auto& state = gdn_states_[type_idx];

  // 1. RMSNorm
  tensor::Tensor rms_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  qwen_layers_->rmsnorm_layers_[layer_idx]->forward(input, rms_output);

  // 2. Projections: fused QKV+Z (single launch) + tiny A, B
  fused_gdn_proj_gemv_layer_->forward(
      rms_output.ptr<half>(),
      std::dynamic_pointer_cast<op::LayerParam>(la.in_proj_qkv)->get_weight(0).ptr<half>(),
      std::dynamic_pointer_cast<op::LayerParam>(la.in_proj_z)->get_weight(0).ptr<half>(),
      gdn_qkv_buf_.ptr<half>(), gdn_z_buf_.ptr<half>(),
      dim, conv_dim, dim);
  la.in_proj_a->forward(rms_output, gdn_alpha_buf_);     // [n_v_heads=32]
  la.in_proj_b->forward(rms_output, gdn_beta_buf_);      // [n_v_heads=32]

  // 3. Causal Conv1D + SiLU
  causal_conv1d_silu_layer_->forward(
      state.conv_state.ptr<half>(), gdn_qkv_buf_.ptr<half>(),
      la.conv_weight.ptr<half>(), gdn_conv_out_.ptr<half>(),
      conv_dim, ks);

  // 4. Split conv output into Q, K, V
  // Layout: [Q: n_k_heads*k_head_dim, K: n_k_heads*k_head_dim, V: n_v_heads*v_head_dim]
  int q_size = n_k_heads * k_head_dim;
  int k_size = n_k_heads * k_head_dim;
  int v_size = n_v_heads * v_head_dim;
  half* q_ptr = gdn_conv_out_.ptr<half>();
  half* k_ptr = q_ptr + q_size;
  half* v_ptr = k_ptr + k_size;

  // 5. L2 normalize Q and K
  l2_norm_per_head_layer_->forward(q_ptr, gdn_q_norm_.ptr<half>(), n_k_heads, k_head_dim, 1e-6f);
  l2_norm_per_head_layer_->forward(k_ptr, gdn_k_norm_.ptr<half>(), n_k_heads, k_head_dim, 1e-6f);

  // 6. Compute gates
  compute_gdn_gates_layer_->forward(
      gdn_alpha_buf_.ptr<half>(), la.dt_bias.ptr<half>(),
      la.A_log.ptr<float>(), gdn_beta_buf_.ptr<half>(),
      gdn_gate_buf_.ptr<float>(), gdn_beta_fp32_.ptr<float>(),
      n_v_heads);

  // 7. Delta Net decode step
  gdn_decode_step_layer_->forward(
      gdn_q_norm_.ptr<half>(), gdn_k_norm_.ptr<half>(), v_ptr,
      gdn_gate_buf_.ptr<float>(), gdn_beta_fp32_.ptr<float>(),
      state.ssm_state.ptr<float>(), gdn_attn_out_.ptr<half>(),
      n_k_heads, n_v_heads, k_head_dim, v_head_dim);

  // 8. Gated RMSNorm: RMSNorm(attn_out) * SiLU(z)
  gated_rmsnorm_layer_->forward(
      gdn_attn_out_.ptr<half>(), gdn_z_buf_.ptr<half>(),
      la.norm_weight.ptr<float>(), gdn_normed_out_.ptr<half>(),
      dim, q35_config_.rms_norm_eps);

  // 9. Output projection
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  la.out_proj->forward(gdn_normed_out_, attn_output);
}

// ==========================================================================
// Decode: Feed Forward
// ==========================================================================

void Qwen35Model::q35_feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  int n_layers = q35_config_.num_hidden_layers;

  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  
  // Residual add: input += attn_output (input buffer is mutable despite const ref)
  qwen_layers_->add_layer_->forward(input, attn_output, input);

  // FFN RMSNorm
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  qwen_layers_->rmsnorm_layers_[layer_idx + n_layers]->forward(input, ffn_norm_output);

  // Gate + Up projections with optional fused FFN
  tensor::Tensor w1_out = get_buffer(ModelBufferType::kW1Output);
  auto w1_mm = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->w1_layers_[layer_idx]);
  auto w3_mm = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->w3_layers_[layer_idx]);

  auto fused_ffn = qwen_layers_->fused_ffn_layer_;
  if (fused_ffn && w1_mm && w3_mm) {
    // Fused gate+up+SwiGLU in one kernel
    fused_ffn->set_use_fp16(true);
    fused_ffn->set_input(0, ffn_norm_output);
    fused_ffn->set_input(1, w1_mm->get_weight(0));
    fused_ffn->set_input(2, w3_mm->get_weight(0));
    fused_ffn->set_output(0, w1_out);
    fused_ffn->set_cuda_config(cuda_config_);
    fused_ffn->forward();
  } else {
    qwen_layers_->w1_layers_[layer_idx]->forward(ffn_norm_output, w1_out);
    tensor::Tensor w3_out = get_buffer(ModelBufferType::kW3Output);
    qwen_layers_->w3_layers_[layer_idx]->forward(ffn_norm_output, w3_out);
    qwen_layers_->swiglu_layer_->forward(w1_out, w3_out, w1_out);
  }

  // Down projection
  tensor::Tensor w2_out = get_buffer(ModelBufferType::kW2Output);
  qwen_layers_->w2_layers_[layer_idx]->forward(w1_out, w2_out);

  // Residual add
  qwen_layers_->add_layer_->forward(input, w2_out, input);
}

// ==========================================================================
// Decode: CLS Logits
// ==========================================================================

void Qwen35Model::q35_cls_logits(const tensor::Tensor& input) const {
  int n_layers = q35_config_.num_hidden_layers;
  tensor::Tensor rms_out = get_buffer(ModelBufferType::kOutputRMSNorm);
  qwen_layers_->rmsnorm_layers_[2 * n_layers]->forward(input, rms_out);
  
  tensor::Tensor forward_out = get_buffer(ModelBufferType::kForwardOutput);
  qwen_layers_->cls_layer_->forward(rms_out, forward_out);
}

// ==========================================================================
// Decode: Full Step
// ==========================================================================

base::Status Qwen35Model::decode_step_optimized(int32_t pos, int& next) const {
  auto stream = cuda_config_->stream;
  int n_layers = q35_config_.num_hidden_layers;
  bool use_graph = cuda_config_ && cuda_config_->use_cuda_graph;

  tensor::Tensor decode_input = get_buffer(ModelBufferType::kDecodeInput);

  if (use_graph) {
    auto& graph_ctx = cuda_config_->graph_context;
    auto& graph = graph_ctx->decode_graph;

    tensor::Tensor pos_gpu = get_buffer(ModelBufferType::kInputPosGPU);
    tensor::Tensor kv_pos_gpu = get_buffer(ModelBufferType::kKVCachePosGPU);
    tensor::Tensor pos_pinned = get_buffer(ModelBufferType::kInputPosPinned);
    tensor::Tensor kv_pos_pinned = get_buffer(ModelBufferType::kKVCachePosPinned);
    tensor::Tensor argmax_output = get_buffer(ModelBufferType::kArgmaxOutput);
    tensor::Tensor argmax_pinned = get_buffer(ModelBufferType::kArgmaxOutputPinned);

    // Compute text_pos for RoPE (same as non-graph path)
    int32_t text_pos = mrope_max_text_pos_ + (pos - prefill_seq_len_) + 1;

    // Update GPU positions via pinned H2D copy
    *const_cast<int32_t*>(pos_pinned.ptr<int32_t>()) = text_pos;
    cudaMemcpyAsync(const_cast<int32_t*>(pos_gpu.ptr<int32_t>()),
                    pos_pinned.ptr<int32_t>(), sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    *const_cast<int32_t*>(kv_pos_pinned.ptr<int32_t>()) = pos;
    cudaMemcpyAsync(const_cast<int32_t*>(kv_pos_gpu.ptr<int32_t>()),
                    kv_pos_pinned.ptr<int32_t>(), sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);

    bool need_capture = graph_ctx->needs_recapture || !graph->is_valid();

    if (need_capture && !graph->is_disabled()) {
      cudaStreamSynchronize(stream);

      if (graph->begin_capture(stream)) {
        for (int il = 0; il < n_layers; ++il) {
          if (q35_config_.is_full_attn_layer(il)) {
            full_attn_decode_graph(il, decode_input,
                                   pos_gpu.ptr<int32_t>(), kv_pos_gpu.ptr<int32_t>());
          } else {
            linear_attn_decode(il, decode_input);
          }
          q35_feed_forward(il, decode_input);
        }
        q35_cls_logits(decode_input);

        if (graph->end_capture(stream)) {
          graph_ctx->graph_recaptures++;
          graph_ctx->needs_recapture = false;
        }
      }
    }

    if (graph->is_valid()) {
      if (graph->launch(stream)) {
        graph_ctx->graph_launches++;

        tensor::Tensor forward_out = get_buffer(ModelBufferType::kForwardOutput);
        auto* argmax_sampler = dynamic_cast<sampler::ArgmaxSampler*>(sampler_.get());
        if (argmax_sampler) {
          argmax_sampler->sample_prealloc(
              forward_out.ptr<float>(), forward_out.size(),
              reinterpret_cast<size_t*>(const_cast<int32_t*>(argmax_output.ptr<int32_t>())),
              reinterpret_cast<size_t*>(const_cast<int32_t*>(argmax_pinned.ptr<int32_t>())),
              stream);
          cudaStreamSynchronize(stream);
          next = static_cast<int32_t>(*reinterpret_cast<size_t*>(
              const_cast<int32_t*>(argmax_pinned.ptr<int32_t>())));
        } else {
          cudaStreamSynchronize(stream);
          next = sampler_->sample(forward_out.ptr<float>(), forward_out.size(), stream);
        }
        return error::Success();
      }
      graph_ctx->invalidate();
    }
  }

  // Normal (non-graph) execution
  tensor::Tensor pos_tensor = get_buffer(ModelBufferType::kInputPos);
  *const_cast<int32_t*>(pos_tensor.ptr<int32_t>()) = pos;

  // Optional per-category timing (activated on step 5 to skip warmup)
  // Set Q35_PROFILE_DECODE=1 environment variable to enable
  static int profile_step_counter = 0;
  static bool profile_enabled = (std::getenv("Q35_PROFILE_DECODE") != nullptr);
  const bool do_profile = profile_enabled && (profile_step_counter == 5);
  profile_step_counter++;

  cudaEvent_t ev_start, ev_full_attn, ev_linear_attn, ev_ffn, ev_cls, ev_end;
  float t_full_attn_ms = 0, t_linear_attn_ms = 0, t_ffn_ms = 0, t_cls_ms = 0;
  if (do_profile) {
    cudaEventCreate(&ev_start); cudaEventCreate(&ev_full_attn);
    cudaEventCreate(&ev_linear_attn); cudaEventCreate(&ev_ffn);
    cudaEventCreate(&ev_cls); cudaEventCreate(&ev_end);
  }

  float total_full_attn = 0, total_linear_attn = 0, total_ffn = 0;

  for (int il = 0; il < n_layers; ++il) {
    if (do_profile) cudaEventRecord(ev_start, stream);

    if (q35_config_.is_full_attn_layer(il)) {
      full_attn_decode(il, decode_input);
      if (do_profile) {
        cudaEventRecord(ev_full_attn, stream);
        cudaEventSynchronize(ev_full_attn);
        float ms; cudaEventElapsedTime(&ms, ev_start, ev_full_attn);
        total_full_attn += ms;
      }
    } else {
      linear_attn_decode(il, decode_input);
      if (do_profile) {
        cudaEventRecord(ev_linear_attn, stream);
        cudaEventSynchronize(ev_linear_attn);
        float ms; cudaEventElapsedTime(&ms, ev_start, ev_linear_attn);
        total_linear_attn += ms;
      }
    }

    if (do_profile) cudaEventRecord(ev_start, stream);
    q35_feed_forward(il, decode_input);
    if (do_profile) {
      cudaEventRecord(ev_ffn, stream);
      cudaEventSynchronize(ev_ffn);
      float ms; cudaEventElapsedTime(&ms, ev_start, ev_ffn);
      total_ffn += ms;
    }
  }

  if (do_profile) cudaEventRecord(ev_start, stream);
  q35_cls_logits(decode_input);
  if (do_profile) {
    cudaEventRecord(ev_cls, stream);
    cudaEventSynchronize(ev_cls);
    float ms; cudaEventElapsedTime(&ms, ev_start, ev_cls);
    t_cls_ms = ms;
  }

  cudaStreamSynchronize(stream);

  if (do_profile) {
    float total = total_full_attn + total_linear_attn + total_ffn + t_cls_ms;
    LOG(INFO) << "\n=== Decode Step Profile (step 5) ===";
    LOG(INFO) << "  Full Attention (8 layers):  " << total_full_attn << " ms (" 
              << (100.0f * total_full_attn / total) << "%)";
    LOG(INFO) << "  Linear Attn/GDN (24 layers): " << total_linear_attn << " ms ("
              << (100.0f * total_linear_attn / total) << "%)";
    LOG(INFO) << "  FFN (32 layers):             " << total_ffn << " ms ("
              << (100.0f * total_ffn / total) << "%)";
    LOG(INFO) << "  LM Head (cls_logits):        " << t_cls_ms << " ms ("
              << (100.0f * t_cls_ms / total) << "%)";
    LOG(INFO) << "  Total GPU time:              " << total << " ms";
    LOG(INFO) << "=================================";
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_full_attn);
    cudaEventDestroy(ev_linear_attn); cudaEventDestroy(ev_ffn);
    cudaEventDestroy(ev_cls); cudaEventDestroy(ev_end);
  }

  tensor::Tensor forward_out = get_buffer(ModelBufferType::kForwardOutput);
  next = sampler_->sample(forward_out.ptr<float>(), forward_out.size(), stream);

  return error::Success();
}

// ==========================================================================
// Prefill
// ==========================================================================

base::Status Qwen35Model::prefill(const tensor::Tensor& input_embeddings,
                                  int32_t seq_len, int32_t start_pos) const {
  auto stream = cuda_config_->stream;
  int n_layers = q35_config_.num_hidden_layers;
  int dim = q35_config_.hidden_size;
  int intermediate = q35_config_.intermediate_size;
  int n_heads = q35_config_.num_attention_heads;
  int n_kv_heads = q35_config_.num_key_value_heads;
  int kv_dim = q35_config_.kv_dim();
  int q_dim = q35_config_.q_dim();
  int q_gate_dim = q35_config_.q_gate_dim();
  int head_dim = q35_config_.head_dim;
  int partial_dim = q35_config_.partial_rope_dim();
  int conv_dim = q35_config_.conv_dim();
  int n_k_heads = q35_config_.linear_num_key_heads;
  int n_v_heads = q35_config_.linear_num_value_heads;
  int k_head_dim = q35_config_.linear_key_head_dim;
  int v_head_dim = q35_config_.linear_value_head_dim;
  int ks = q35_config_.linear_conv_kernel_dim;
  int kv_mul = config_->kv_mul_;
  int n_full = static_cast<int>(q35_config_.full_attn_layer_indices.size());
  int q_size = n_k_heads * k_head_dim;
  int k_size = n_k_heads * k_head_dim;
  int v_size = n_v_heads * v_head_dim;

  // Upload M-RoPE positions (generate text-only sequential positions if needed)
  if (mrope_pos_t_.empty()) {
    mrope_pos_t_.resize(seq_len);
    mrope_pos_h_.resize(seq_len);
    mrope_pos_w_.resize(seq_len);
    for (int i = 0; i < seq_len; ++i) {
      mrope_pos_t_[i] = i;
      mrope_pos_h_[i] = i;
      mrope_pos_w_[i] = i;
    }
    mrope_max_text_pos_ = seq_len - 1;
  }
  vision_vl_layers_.generate_mrope_positions_layer_->upload(
      mrope_pos_t_, mrope_pos_h_, mrope_pos_w_,
      mrope_pos_t_gpu_, mrope_pos_h_gpu_, mrope_pos_w_gpu_,
      cuda_config_->stream);

  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  auto act = DataType::kDataTypeFp16;

  // ==========================================
  // Pre-allocate ALL working buffers (reused across layers)
  // ==========================================

  // Common buffers
  tensor::Tensor hidden0(act, seq_len, dim, true, alloc);
  tensor::Tensor hidden1(act, seq_len, dim, true, alloc);
  tensor::Tensor rms_out(act, seq_len, dim, true, alloc);
  tensor::Tensor ffn_norm(act, seq_len, dim, true, alloc);
  tensor::Tensor w1_out(act, seq_len, intermediate, true, alloc);
  tensor::Tensor w3_out(act, seq_len, intermediate, true, alloc);
  tensor::Tensor w2_out(act, seq_len, dim, true, alloc);
  tensor::Tensor attn_out_proj(act, seq_len, dim, true, alloc);

  // Full attention buffers (reused across 8 layers)
  tensor::Tensor query_gate(act, seq_len, q_gate_dim, true, alloc);
  tensor::Tensor key_buf(act, seq_len, kv_dim, true, alloc);
  tensor::Tensor val_buf(act, seq_len, kv_dim, true, alloc);
  tensor::Tensor mha_out(act, seq_len, dim, true, alloc);
  tensor::Tensor q_extracted(act, seq_len, q_dim, true, alloc);
  tensor::Tensor gate_extracted(act, seq_len, q_dim, true, alloc);

  int kv_len = start_pos + seq_len;
  half* score_buf = nullptr;
  cudaMalloc(&score_buf, (int64_t)n_heads * kv_len * seq_len * sizeof(half));

  // Linear attention buffers (reused across 24 layers)
  tensor::Tensor lin_qkv(act, seq_len, conv_dim, true, alloc);
  tensor::Tensor lin_z(act, seq_len, dim, true, alloc);
  tensor::Tensor lin_alpha(act, seq_len, n_v_heads, true, alloc);
  tensor::Tensor lin_beta(act, seq_len, n_v_heads, true, alloc);
  tensor::Tensor lin_conv_out(act, seq_len, conv_dim, true, alloc);
  tensor::Tensor lin_gdn_out(act, seq_len, n_v_heads * v_head_dim, true, alloc);
  tensor::Tensor lin_normed(act, seq_len, dim, true, alloc);
  tensor::Tensor gate_fp32(DataType::kDataTypeFp32, seq_len * n_v_heads, true, alloc);
  tensor::Tensor beta_fp32(DataType::kDataTypeFp32, seq_len * n_v_heads, true, alloc);
  tensor::Tensor q_normed(act, seq_len * q_size, true, alloc);
  tensor::Tensor k_normed(act, seq_len * k_size, true, alloc);
  tensor::Tensor v_gathered(act, seq_len * v_size, true, alloc);

  // Transposed state buffer for optimized coalesced GDN prefill
  // Standard: [v_head, v_dim, k_dim], Transposed: [v_head, k_dim, v_dim]
  tensor::Tensor state_transposed(DataType::kDataTypeFp32,
      n_v_heads * k_head_dim * v_head_dim, true, alloc);

  // ==========================================
  // Pre-compute batched attention pointer arrays for all full_attn layers
  // ==========================================
  const int ptrs_per_layer = 6 * n_heads;
  const size_t total_ptrs = n_full * ptrs_per_layer;

  half** d_attn_ptrs = nullptr;
  if (n_full > 0 && total_ptrs > 0) {
    cudaMalloc(&d_attn_ptrs, total_ptrs * sizeof(half*));
    half** h_ptrs = nullptr;
    cudaMallocHost(&h_ptrs, total_ptrs * sizeof(half*));

    half* kc_base = const_cast<half*>(get_buffer(ModelBufferType::kKeyCache).ptr<half>());
    half* vc_base = const_cast<half*>(get_buffer(ModelBufferType::kValueCache).ptr<half>());

    for (int ti = 0; ti < n_full; ++ti) {
      half* kc = kc_base + (int64_t)ti * config_->seq_len_ * kv_dim + start_pos * kv_dim;
      half* vc = vc_base + (int64_t)ti * config_->seq_len_ * kv_dim + start_pos * kv_dim;
      int base_off = ti * ptrs_per_layer;

      for (int h = 0; h < n_heads; ++h) {
        int kv_h = h / kv_mul;
        // Step 1: Q · K^T  (A=K, B=Q, C=Score)
        h_ptrs[base_off + h]                = kc + kv_h * head_dim;
        h_ptrs[base_off + n_heads + h]      = q_extracted.ptr<half>() + h * head_dim;
        h_ptrs[base_off + 2 * n_heads + h]  = score_buf + (int64_t)h * kv_len * seq_len;
        // Step 3: Score · V → mha_out  (A=V, B=Score, C=mha_out)
        h_ptrs[base_off + 3 * n_heads + h]  = vc + kv_h * head_dim;
        h_ptrs[base_off + 4 * n_heads + h]  = score_buf + (int64_t)h * kv_len * seq_len;
        h_ptrs[base_off + 5 * n_heads + h]  = mha_out.ptr<half>() + h * head_dim;
      }
    }

    cudaMemcpyAsync(d_attn_ptrs, h_ptrs, total_ptrs * sizeof(half*),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    cudaFreeHost(h_ptrs);
  }

  // ==========================================
  // Layer loop (double-buffered)
  // ==========================================
  tensor::Tensor* cur_in = nullptr;
  tensor::Tensor* cur_out = nullptr;

  for (int il = 0; il < n_layers; ++il) {
    if (il == 0) {
      cur_in = const_cast<tensor::Tensor*>(&input_embeddings);
      cur_out = &hidden0;
    } else {
      cur_in = (il % 2 == 1) ? &hidden0 : &hidden1;
      cur_out = (il % 2 == 1) ? &hidden1 : &hidden0;
    }

    // RMSNorm
    auto rms_il = std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->rmsnorm_layers_[il]);
    batched_rmsnorm_fp16_layer_->forward(cur_in->ptr<half>(), rms_out.ptr<half>(),
        rms_il->get_weight(0).ptr<half>(),
        dim, seq_len, q35_config_.rms_norm_eps);

    if (q35_config_.is_full_attn_layer(il)) {
      int ti = full_attn_type_idx(il);
      int q_norm_idx = 2 * n_layers + 1 + ti;
      int k_norm_idx = 2 * n_layers + 1 + n_full + ti;

      // Q (with gate), K, V projections via cuBLAS GEMM op
      auto wq = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wq_layers_[ti]);
      auto wk = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wk_layers_[ti]);
      auto wv = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wv_layers_[ti]);

      vision_vl_layers_.batched_gemm_layer_->forward(
          wq->get_weight(0).ptr<half>(), rms_out.ptr<half>(), query_gate.ptr<half>(),
          q_gate_dim, seq_len, dim, true, false, 1.0f, 0.0f,
          dim, dim, q_gate_dim, cuda_config_.get());

      vision_vl_layers_.batched_gemm_layer_->forward(
          wk->get_weight(0).ptr<half>(), rms_out.ptr<half>(), key_buf.ptr<half>(),
          kv_dim, seq_len, dim, true, false, 1.0f, 0.0f,
          dim, dim, kv_dim, cuda_config_.get());

      vision_vl_layers_.batched_gemm_layer_->forward(
          wv->get_weight(0).ptr<half>(), rms_out.ptr<half>(), val_buf.ptr<half>(),
          kv_dim, seq_len, dim, true, false, 1.0f, 0.0f,
          dim, dim, kv_dim, cuda_config_.get());

      // Deinterleave Q and Gate per-head
      deinterleave_q_gate_layer_->forward(
          query_gate.ptr<half>(), q_extracted.ptr<half>(), gate_extracted.ptr<half>(),
          n_heads, head_dim, seq_len);

      // Per-head Q/K norm
      auto rms_qn = std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->rmsnorm_layers_[q_norm_idx]);
      auto rms_kn = std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->rmsnorm_layers_[k_norm_idx]);
      batched_rmsnorm_fp16_layer_->forward_dim(
          q_extracted.ptr<half>(), q_extracted.ptr<half>(),
          rms_qn->get_weight(0).ptr<half>(),
          head_dim, seq_len * n_heads, q35_config_.rms_norm_eps);
      batched_rmsnorm_fp16_layer_->forward_dim(
          key_buf.ptr<half>(), key_buf.ptr<half>(),
          rms_kn->get_weight(0).ptr<half>(),
          head_dim, seq_len * n_kv_heads, q35_config_.rms_norm_eps);

      // Partial M-RoPE (interleaved)
      partial_mrope_layer_->forward_batched(
          q_extracted.ptr<half>(), key_buf.ptr<half>(),
          get_buffer(ModelBufferType::kSinCache).ptr<float>(),
          get_buffer(ModelBufferType::kCosCache).ptr<float>(),
          mrope_pos_t_gpu_ + start_pos, mrope_pos_h_gpu_ + start_pos,
          mrope_pos_w_gpu_ + start_pos,
          mrope_sections_cpu_, partial_dim, head_dim, n_heads, n_kv_heads, seq_len);

      // Write K, V to cache
      half* kc_ptr = const_cast<half*>(get_buffer(ModelBufferType::kKeyCache).ptr<half>())
          + static_cast<size_t>(ti) * config_->seq_len_ * kv_dim + start_pos * kv_dim;
      half* vc_ptr = const_cast<half*>(get_buffer(ModelBufferType::kValueCache).ptr<half>())
          + static_cast<size_t>(ti) * config_->seq_len_ * kv_dim + start_pos * kv_dim;
      cudaMemcpyAsync(kc_ptr, key_buf.ptr<half>(), seq_len * kv_dim * sizeof(half),
                      cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(vc_ptr, val_buf.ptr<half>(), seq_len * kv_dim * sizeof(half),
                      cudaMemcpyDeviceToDevice, stream);

      // Batched MHA via cuBLAS GEMM op (replaces per-head loop)
      float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
      half** layer_ptrs = d_attn_ptrs + ti * ptrs_per_layer;

      // Step 1: Q · K^T via BatchedGemmLayer (batched over all heads)
      vision_vl_layers_.batched_gemm_layer_->forward_batched(
          (const half**)layer_ptrs, (const half**)(layer_ptrs + n_heads),
          layer_ptrs + 2 * n_heads,
          kv_len, seq_len, head_dim, true, false, scale, 0.0f,
          kv_dim, q_dim, kv_len, n_heads, cuda_config_.get());

      // Step 2: Causal softmax (all heads at once)
      vision_vl_layers_.causal_softmax_layer_->forward(
          score_buf, n_heads, seq_len, kv_len, start_pos, stream);

      // Step 3: Attn · V → mha_out via BatchedGemmLayer (batched)
      half** step3 = layer_ptrs + 3 * n_heads;
      vision_vl_layers_.batched_gemm_layer_->forward_batched(
          (const half**)step3, (const half**)(step3 + n_heads),
          step3 + 2 * n_heads,
          head_dim, seq_len, kv_len, false, false, 1.0f, 0.0f,
          kv_dim, kv_len, dim, n_heads, cuda_config_.get());

      // Apply sigmoid gate
      apply_sigmoid_gate_layer_->forward_batched(mha_out.ptr<half>(), gate_extracted.ptr<half>(),
                                                  q_dim, seq_len);

      // WO projection via BatchedGemmLayer
      auto wo = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->wo_layers_[ti]);
      vision_vl_layers_.batched_gemm_layer_->forward(
          wo->get_weight(0).ptr<half>(), mha_out.ptr<half>(), attn_out_proj.ptr<half>(),
          dim, seq_len, q_dim, true, false, 1.0f, 0.0f,
          q_dim, q_dim, dim, cuda_config_.get());

      // Residual
      batched_add_fp16_layer_->forward(cur_in->ptr<half>(), attn_out_proj.ptr<half>(),
                                       cur_out->ptr<half>(), dim, seq_len);

    } else {
      // Linear attention layer (GDN)
      int ti = linear_attn_type_idx(il);
      auto& la = linear_attn_weights_->layers[ti];
      auto& state = gdn_states_[ti];

      // Projections via cuBLAS GEMM op
      auto proj_qkv = std::dynamic_pointer_cast<op::MatmulLayer>(la.in_proj_qkv);
      auto proj_z = std::dynamic_pointer_cast<op::MatmulLayer>(la.in_proj_z);
      auto proj_a = std::dynamic_pointer_cast<op::MatmulLayer>(la.in_proj_a);
      auto proj_b = std::dynamic_pointer_cast<op::MatmulLayer>(la.in_proj_b);

      vision_vl_layers_.batched_gemm_layer_->forward(
          proj_qkv->get_weight(0).ptr<half>(), rms_out.ptr<half>(), lin_qkv.ptr<half>(),
          conv_dim, seq_len, dim, true, false, 1.0f, 0.0f,
          dim, dim, conv_dim, cuda_config_.get());

      vision_vl_layers_.batched_gemm_layer_->forward(
          proj_z->get_weight(0).ptr<half>(), rms_out.ptr<half>(), lin_z.ptr<half>(),
          dim, seq_len, dim, true, false, 1.0f, 0.0f,
          dim, dim, dim, cuda_config_.get());

      vision_vl_layers_.batched_gemm_layer_->forward(
          proj_a->get_weight(0).ptr<half>(), rms_out.ptr<half>(), lin_alpha.ptr<half>(),
          n_v_heads, seq_len, dim, true, false, 1.0f, 0.0f,
          dim, dim, n_v_heads, cuda_config_.get());

      vision_vl_layers_.batched_gemm_layer_->forward(
          proj_b->get_weight(0).ptr<half>(), rms_out.ptr<half>(), lin_beta.ptr<half>(),
          n_v_heads, seq_len, dim, true, false, 1.0f, 0.0f,
          dim, dim, n_v_heads, cuda_config_.get());

      // Conv1D
      causal_conv1d_silu_layer_->forward_batched(
          state.conv_state.ptr<half>(), lin_qkv.ptr<half>(),
          la.conv_weight.ptr<half>(), lin_conv_out.ptr<half>(),
          conv_dim, ks, seq_len);

      // Extract Q, K, V from strided conv_out into contiguous buffers (single kernel each)
      gather_strided_layer_->forward(lin_conv_out.ptr<half>(), q_normed.ptr<half>(),
                                      q_size, conv_dim, 0, seq_len);
      gather_strided_layer_->forward(lin_conv_out.ptr<half>(), k_normed.ptr<half>(),
                                      k_size, conv_dim, q_size, seq_len);
      gather_strided_layer_->forward(lin_conv_out.ptr<half>(), v_gathered.ptr<half>(),
                                      v_size, conv_dim, q_size + k_size, seq_len);

      // L2 norm per-head for all tokens at once (single kernel launch)
      l2_norm_per_head_layer_->forward(q_normed.ptr<half>(), q_normed.ptr<half>(),
                                        seq_len * n_k_heads, k_head_dim, 1e-6f);
      l2_norm_per_head_layer_->forward(k_normed.ptr<half>(), k_normed.ptr<half>(),
                                        seq_len * n_k_heads, k_head_dim, 1e-6f);

      // Gates
      compute_gdn_gates_layer_->forward_batched(
          lin_alpha.ptr<half>(), la.dt_bias.ptr<half>(),
          la.A_log.ptr<float>(), lin_beta.ptr<half>(),
          gate_fp32.ptr<float>(), beta_fp32.ptr<float>(),
          n_v_heads, seq_len);

      // Transpose state [v_head, v_dim, k_dim] → [v_head, k_dim, v_dim] for coalesced access
      transpose_state_layer_->forward(state.ssm_state.ptr<float>(),
                                       state_transposed.ptr<float>(),
                                       n_v_heads, v_head_dim, k_head_dim);

      // Optimized GDN prefill with transposed state (coalesced memory access)
      gdn_prefill_transposed_layer_->forward(
          q_normed.ptr<half>(), k_normed.ptr<half>(), v_gathered.ptr<half>(),
          gate_fp32.ptr<float>(), beta_fp32.ptr<float>(),
          state_transposed.ptr<float>(), lin_gdn_out.ptr<half>(),
          n_k_heads, n_v_heads, k_head_dim, v_head_dim, seq_len);

      // Transpose state back [v_head, k_dim, v_dim] → [v_head, v_dim, k_dim] for decode
      transpose_state_layer_->forward(state_transposed.ptr<float>(),
                                       state.ssm_state.ptr<float>(),
                                       n_v_heads, k_head_dim, v_head_dim);

      // Gated RMSNorm
      gated_rmsnorm_layer_->forward_batched(
          lin_gdn_out.ptr<half>(), lin_z.ptr<half>(),
          la.norm_weight.ptr<float>(), lin_normed.ptr<half>(),
          dim, seq_len, q35_config_.rms_norm_eps);

      // Output projection via BatchedGemmLayer
      auto out_p = std::dynamic_pointer_cast<op::MatmulLayer>(la.out_proj);
      vision_vl_layers_.batched_gemm_layer_->forward(
          out_p->get_weight(0).ptr<half>(), lin_normed.ptr<half>(), attn_out_proj.ptr<half>(),
          dim, seq_len, dim, true, false, 1.0f, 0.0f,
          dim, dim, dim, cuda_config_.get());

      // Residual
      batched_add_fp16_layer_->forward(cur_in->ptr<half>(), attn_out_proj.ptr<half>(),
                                       cur_out->ptr<half>(), dim, seq_len);
    }

    // FFN (common for both layer types)
    auto rms_ffn = std::dynamic_pointer_cast<op::LayerParam>(qwen_layers_->rmsnorm_layers_[il + n_layers]);
    batched_rmsnorm_fp16_layer_->forward(cur_out->ptr<half>(), ffn_norm.ptr<half>(),
        rms_ffn->get_weight(0).ptr<half>(),
        dim, seq_len, q35_config_.rms_norm_eps);

    auto w1 = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->w1_layers_[il]);
    auto w2 = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->w2_layers_[il]);
    auto w3 = std::dynamic_pointer_cast<op::MatmulLayer>(qwen_layers_->w3_layers_[il]);

    // W1/W3 via BatchedGemmLayer
    vision_vl_layers_.batched_gemm_layer_->forward(
        w1->get_weight(0).ptr<half>(), ffn_norm.ptr<half>(), w1_out.ptr<half>(),
        intermediate, seq_len, dim, true, false, 1.0f, 0.0f,
        dim, dim, intermediate, cuda_config_.get());
    vision_vl_layers_.batched_gemm_layer_->forward(
        w3->get_weight(0).ptr<half>(), ffn_norm.ptr<half>(), w3_out.ptr<half>(),
        intermediate, seq_len, dim, true, false, 1.0f, 0.0f,
        dim, dim, intermediate, cuda_config_.get());

    // Batched SwiGLU
    qwen_layers_->batched_swiglu_layer_->forward(w1_out, w3_out, w1_out);

    // W2 via BatchedGemmLayer
    vision_vl_layers_.batched_gemm_layer_->forward(
        w2->get_weight(0).ptr<half>(), w1_out.ptr<half>(), w2_out.ptr<half>(),
        dim, seq_len, intermediate, true, false, 1.0f, 0.0f,
        intermediate, intermediate, dim, cuda_config_.get());

    // Residual add
    batched_add_fp16_layer_->forward(cur_out->ptr<half>(), w2_out.ptr<half>(),
                                     cur_out->ptr<half>(), dim, seq_len);
  }

  // Free batched attention buffers
  if (score_buf) cudaFree(score_buf);
  if (d_attn_ptrs) cudaFree(d_attn_ptrs);

  // Copy last token's hidden state to decode input buffer
  tensor::Tensor& final_hidden = (n_layers % 2 == 1) ? hidden0 : hidden1;
  void* decode_dst = const_cast<void*>(get_buffer(ModelBufferType::kDecodeInput).ptr<void>());
  
  if (n_layers == 0) {
    cudaMemcpyAsync(decode_dst,
                    input_embeddings.ptr<half>() + (seq_len - 1) * dim,
                    dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
  } else {
    cudaMemcpyAsync(decode_dst,
                    final_hidden.ptr<half>() + (seq_len - 1) * dim,
                    dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
  }

  prefill_seq_len_ = seq_len;
  cudaStreamSynchronize(stream);
  return error::Success();
}

// ==========================================================================
// Sample First Token (after prefill)
// ==========================================================================

int Qwen35Model::sample_first_token() const {
  tensor::Tensor decode_input = get_buffer(ModelBufferType::kDecodeInput);
  q35_cls_logits(decode_input);
  cudaStreamSynchronize(cuda_config_->stream);
  
  tensor::Tensor forward_out = get_buffer(ModelBufferType::kForwardOutput);
  
  return sampler_->sample(forward_out.ptr<float>(), forward_out.size(), cuda_config_->stream);
}

}  // namespace model

#endif  // QWEN3_VL_SUPPORT
