#include "op/gdn_layers.h"
#include "kernels/cuda/gdn_kernel.cuh"

namespace op {

// ==================== DeinterleaveQGateLayer ====================

DeinterleaveQGateLayer::DeinterleaveQGateLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "DeinterleaveQGate") {
  reset_input_size(1);
  reset_output_size(2);
}

base::Status DeinterleaveQGateLayer::check() const {
  return base::error::Success();
}

base::Status DeinterleaveQGateLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status DeinterleaveQGateLayer::forward(const half* interleaved, half* q_out, half* gate_out,
                                             int n_heads, int head_dim, int seq_len) {
  kernel::deinterleave_q_gate_cu(interleaved, q_out, gate_out, n_heads, head_dim, seq_len,
                                 cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== PartialMRoPEInterleavedLayer ====================

PartialMRoPEInterleavedLayer::PartialMRoPEInterleavedLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "PartialMRoPEInterleaved") {
  reset_input_size(4);
  reset_output_size(0);
}

base::Status PartialMRoPEInterleavedLayer::check() const {
  return base::error::Success();
}

base::Status PartialMRoPEInterleavedLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status PartialMRoPEInterleavedLayer::forward(
    half* q, half* k, const float* sin_cache, const float* cos_cache,
    int pos_t, int pos_h, int pos_w,
    const int* sections, int partial_dim,
    int head_dim, int num_heads, int num_kv_heads) {
  kernel::partial_mrope_interleaved_cu(q, k, sin_cache, cos_cache,
                                       pos_t, pos_h, pos_w,
                                       sections, partial_dim,
                                       head_dim, num_heads, num_kv_heads,
                                       cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

base::Status PartialMRoPEInterleavedLayer::forward_gpu_pos(
    half* q, half* k, const float* sin_cache, const float* cos_cache,
    const int32_t* pos_gpu, const int* sections, int partial_dim,
    int head_dim, int num_heads, int num_kv_heads) {
  kernel::partial_mrope_interleaved_gpu_pos_cu(q, k, sin_cache, cos_cache,
                                                pos_gpu, sections, partial_dim,
                                                head_dim, num_heads, num_kv_heads,
                                                cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

base::Status PartialMRoPEInterleavedLayer::forward_batched(
    half* q, half* k, const float* sin_cache, const float* cos_cache,
    const int* pos_t, const int* pos_h, const int* pos_w,
    const int* sections, int partial_dim,
    int head_dim, int num_heads, int num_kv_heads, int seq_len) {
  kernel::batched_partial_mrope_interleaved_cu(q, k, sin_cache, cos_cache,
                                                pos_t, pos_h, pos_w,
                                                sections, partial_dim,
                                                head_dim, num_heads, num_kv_heads,
                                                seq_len,
                                                cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== KVCacheWriteGpuPosLayer ====================

KVCacheWriteGpuPosLayer::KVCacheWriteGpuPosLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "KVCacheWriteGpuPos") {
  reset_input_size(1);
  reset_output_size(0);
}

base::Status KVCacheWriteGpuPosLayer::check() const {
  return base::error::Success();
}

base::Status KVCacheWriteGpuPosLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status KVCacheWriteGpuPosLayer::forward(
    half* kv_cache, const half* kv_data, const int32_t* pos_gpu,
    int kv_dim, int layer_idx, int max_seq_len) {
  kernel::kv_cache_write_gpu_pos_cu(kv_cache, kv_data, pos_gpu,
                                     kv_dim, layer_idx, max_seq_len,
                                     cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== ApplySigmoidGateLayer ====================

ApplySigmoidGateLayer::ApplySigmoidGateLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "ApplySigmoidGate") {
  reset_input_size(2);
  reset_output_size(0);
}

base::Status ApplySigmoidGateLayer::check() const {
  return base::error::Success();
}

base::Status ApplySigmoidGateLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status ApplySigmoidGateLayer::forward(half* attn_output, const half* gate, int dim) {
  kernel::apply_sigmoid_gate_cu(attn_output, gate, dim,
                                cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

base::Status ApplySigmoidGateLayer::forward_batched(half* attn_output, const half* gate,
                                                     int dim, int seq_len) {
  kernel::batched_apply_sigmoid_gate_cu(attn_output, gate, dim, seq_len,
                                         cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== CausalConv1dSiluLayer ====================

CausalConv1dSiluLayer::CausalConv1dSiluLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "CausalConv1dSilu") {
  reset_input_size(2);
  reset_output_size(1);
}

base::Status CausalConv1dSiluLayer::check() const {
  return base::error::Success();
}

base::Status CausalConv1dSiluLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status CausalConv1dSiluLayer::forward(
    half* conv_state, const half* new_input, const half* conv_weight, half* output,
    int conv_dim, int kernel_size) {
  kernel::causal_conv1d_silu_cu(conv_state, new_input, conv_weight, output,
                                 conv_dim, kernel_size,
                                 cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

base::Status CausalConv1dSiluLayer::forward_batched(
    half* conv_state, const half* input, const half* conv_weight, half* output,
    int conv_dim, int kernel_size, int seq_len) {
  kernel::batched_causal_conv1d_silu_cu(conv_state, input, conv_weight, output,
                                         conv_dim, kernel_size, seq_len,
                                         cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== L2NormPerHeadLayer ====================

L2NormPerHeadLayer::L2NormPerHeadLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "L2NormPerHead") {
  reset_input_size(1);
  reset_output_size(1);
}

base::Status L2NormPerHeadLayer::check() const {
  return base::error::Success();
}

base::Status L2NormPerHeadLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status L2NormPerHeadLayer::forward(const half* input, half* output,
                                          int num_heads, int head_dim, float eps) {
  kernel::l2_norm_per_head_cu(input, output, num_heads, head_dim, eps,
                               cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== ComputeGDNGatesLayer ====================

ComputeGDNGatesLayer::ComputeGDNGatesLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "ComputeGDNGates") {
  reset_input_size(4);
  reset_output_size(2);
}

base::Status ComputeGDNGatesLayer::check() const {
  return base::error::Success();
}

base::Status ComputeGDNGatesLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status ComputeGDNGatesLayer::forward(
    const half* alpha_raw, const half* dt_bias, const float* A_log,
    const half* beta_raw, float* gate_out, float* beta_out, int num_v_heads) {
  kernel::compute_gdn_gates_cu(alpha_raw, dt_bias, A_log, beta_raw,
                                gate_out, beta_out, num_v_heads,
                                cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

base::Status ComputeGDNGatesLayer::forward_batched(
    const half* alpha_raw, const half* dt_bias, const float* A_log,
    const half* beta_raw, float* gate_out, float* beta_out,
    int num_v_heads, int seq_len) {
  kernel::batched_compute_gdn_gates_cu(alpha_raw, dt_bias, A_log, beta_raw,
                                        gate_out, beta_out, num_v_heads, seq_len,
                                        cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== GDNDecodeStepLayer ====================

GDNDecodeStepLayer::GDNDecodeStepLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "GDNDecodeStep") {
  reset_input_size(5);
  reset_output_size(1);
}

base::Status GDNDecodeStepLayer::check() const {
  return base::error::Success();
}

base::Status GDNDecodeStepLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status GDNDecodeStepLayer::forward(
    const half* q, const half* k, const half* v,
    const float* gate, const float* beta,
    float* state, half* output,
    int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim) {
  kernel::gdn_decode_step_cu(q, k, v, gate, beta, state, output,
                              num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                              cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== GatedRMSNormLayer ====================

GatedRMSNormLayer::GatedRMSNormLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerRMSNorm, "GatedRMSNorm") {
  reset_input_size(2);
  reset_output_size(1);
}

base::Status GatedRMSNormLayer::check() const {
  return base::error::Success();
}

base::Status GatedRMSNormLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status GatedRMSNormLayer::forward(const half* x, const half* z, const float* weight,
                                         half* output, int dim, float eps) {
  kernel::gated_rmsnorm_cu(x, z, weight, output, dim, eps,
                            cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

base::Status GatedRMSNormLayer::forward_batched(const half* x, const half* z,
                                                 const float* weight, half* output,
                                                 int dim, int seq_len, float eps) {
  kernel::batched_gated_rmsnorm_cu(x, z, weight, output, dim, seq_len, eps,
                                    cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== BatchedAddFP16Layer ====================

BatchedAddFP16Layer::BatchedAddFP16Layer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerAdd, "BatchedAddFP16") {
  reset_input_size(2);
  reset_output_size(1);
}

base::Status BatchedAddFP16Layer::check() const {
  return base::error::Success();
}

base::Status BatchedAddFP16Layer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status BatchedAddFP16Layer::forward(const half* a, const half* b, half* output,
                                           int dim, int seq_len) {
  kernel::batched_add_fp16_cu(a, b, output, dim, seq_len,
                               cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== BatchedRMSNormFP16Layer ====================

BatchedRMSNormFP16Layer::BatchedRMSNormFP16Layer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerRMSNorm, "BatchedRMSNormFP16") {
  reset_input_size(2);
  reset_output_size(1);
}

base::Status BatchedRMSNormFP16Layer::check() const {
  return base::error::Success();
}

base::Status BatchedRMSNormFP16Layer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status BatchedRMSNormFP16Layer::forward(const half* input, half* output,
                                               const half* weight,
                                               int dim, int seq_len, float eps) {
  kernel::batched_rmsnorm_fp16_cu(input, output, weight, dim, seq_len, eps,
                                   cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

base::Status BatchedRMSNormFP16Layer::forward_dim(const half* input, half* output,
                                                    const half* weight,
                                                    int dim, int total_rows, float eps) {
  kernel::batched_rmsnorm_dim_fp16_cu(input, output, weight, dim, total_rows, eps,
                                       cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== GatherStridedLayer ====================

GatherStridedLayer::GatherStridedLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "GatherStrided") {
  reset_input_size(1);
  reset_output_size(1);
}

base::Status GatherStridedLayer::check() const {
  return base::error::Success();
}

base::Status GatherStridedLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status GatherStridedLayer::forward(const half* src, half* dst,
                                          int inner_dim, int outer_stride,
                                          int src_offset, int count) {
  kernel::gather_strided_cu(src, dst, inner_dim, outer_stride, src_offset, count,
                             cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== TransposeStateLayer ====================

TransposeStateLayer::TransposeStateLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "TransposeState") {
  reset_input_size(1);
  reset_output_size(1);
}

base::Status TransposeStateLayer::check() const {
  return base::error::Success();
}

base::Status TransposeStateLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status TransposeStateLayer::forward(const float* in, float* out,
                                           int num_heads, int dim1, int dim2) {
  kernel::transpose_state_cu(in, out, num_heads, dim1, dim2,
                              cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== GDNPrefillTransposedLayer ====================

GDNPrefillTransposedLayer::GDNPrefillTransposedLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "GDNPrefillTransposed") {
  reset_input_size(5);
  reset_output_size(1);
}

base::Status GDNPrefillTransposedLayer::check() const {
  return base::error::Success();
}

base::Status GDNPrefillTransposedLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status GDNPrefillTransposedLayer::forward(
    const half* q, const half* k, const half* v,
    const float* gate, const float* beta,
    float* state_transposed, half* output,
    int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int seq_len) {
  kernel::gdn_prefill_transposed_cu(q, k, v, gate, beta, state_transposed, output,
                                     num_k_heads, num_v_heads,
                                     head_k_dim, head_v_dim, seq_len,
                                     cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== FusedQKVGemvLayer ====================

FusedQKVGemvLayer::FusedQKVGemvLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "FusedQKVGemv") {
  reset_input_size(1);
  reset_output_size(3);
}

base::Status FusedQKVGemvLayer::check() const {
  return base::error::Success();
}

base::Status FusedQKVGemvLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status FusedQKVGemvLayer::forward(
    const half* input,
    const half* q_weight, const half* k_weight, const half* v_weight,
    half* q_output, half* k_output, half* v_output,
    int dim, int q_dim, int kv_dim) {
  kernel::fused_fp16_qkv_gemv_cu(input, q_weight, k_weight, v_weight,
                                  q_output, k_output, v_output,
                                  dim, q_dim, kv_dim,
                                  cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

// ==================== FusedGDNProjGemvLayer ====================

FusedGDNProjGemvLayer::FusedGDNProjGemvLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerUnknown, "FusedGDNProjGemv") {
  reset_input_size(1);
  reset_output_size(2);
}

base::Status FusedGDNProjGemvLayer::check() const {
  return base::error::Success();
}

base::Status FusedGDNProjGemvLayer::forward() {
  return base::error::InvalidArgument("Use forward(...) with parameters");
}

base::Status FusedGDNProjGemvLayer::forward(
    const half* input,
    const half* qkv_weight, const half* z_weight,
    half* qkv_output, half* z_output,
    int dim, int qkv_dim, int z_dim) {
  kernel::fused_fp16_gdn_proj_gemv_cu(input, qkv_weight, z_weight,
                                       qkv_output, z_output,
                                       dim, qkv_dim, z_dim,
                                       cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

}  // namespace op
