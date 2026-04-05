#ifndef KUIPER_INCLUDE_OP_GDN_LAYERS_H_
#define KUIPER_INCLUDE_OP_GDN_LAYERS_H_

#include "layer.h"
#include <cuda_fp16.h>

namespace op {

// Deinterleave Q and Gate from per-head interleaved format
class DeinterleaveQGateLayer : public Layer {
 public:
  explicit DeinterleaveQGateLayer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;
  base::Status forward(const half* interleaved, half* q_out, half* gate_out,
                       int n_heads, int head_dim, int seq_len);
};

// Partial M-RoPE with interleaved format (CPU positions, decode)
class PartialMRoPEInterleavedLayer : public Layer {
 public:
  explicit PartialMRoPEInterleavedLayer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;
  base::Status forward(half* q, half* k,
                       const float* sin_cache, const float* cos_cache,
                       int pos_t, int pos_h, int pos_w,
                       const int* sections, int partial_dim,
                       int head_dim, int num_heads, int num_kv_heads);
  // GPU position variant (CUDA Graph compatible)
  base::Status forward_gpu_pos(half* q, half* k,
                               const float* sin_cache, const float* cos_cache,
                               const int32_t* pos_gpu,
                               const int* sections, int partial_dim,
                               int head_dim, int num_heads, int num_kv_heads);
  // Batched variant (prefill)
  base::Status forward_batched(half* q, half* k,
                               const float* sin_cache, const float* cos_cache,
                               const int* pos_t, const int* pos_h, const int* pos_w,
                               const int* sections, int partial_dim,
                               int head_dim, int num_heads, int num_kv_heads, int seq_len);
};

// Write KV data to cache using GPU position
class KVCacheWriteGpuPosLayer : public Layer {
 public:
  explicit KVCacheWriteGpuPosLayer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;
  base::Status forward(half* kv_cache, const half* kv_data,
                       const int32_t* pos_gpu,
                       int kv_dim, int layer_idx, int max_seq_len);
};

// Apply sigmoid gate: output *= sigmoid(gate)
class ApplySigmoidGateLayer : public Layer {
 public:
  explicit ApplySigmoidGateLayer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;
  base::Status forward(half* attn_output, const half* gate, int dim);
  base::Status forward_batched(half* attn_output, const half* gate, int dim, int seq_len);
};

// Causal Conv1D + SiLU activation
class CausalConv1dSiluLayer : public Layer {
 public:
  explicit CausalConv1dSiluLayer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;
  base::Status forward(half* conv_state, const half* new_input,
                       const half* conv_weight, half* output,
                       int conv_dim, int kernel_size);
  base::Status forward_batched(half* conv_state, const half* input,
                               const half* conv_weight, half* output,
                               int conv_dim, int kernel_size, int seq_len);
};

// Per-head L2 normalization
class L2NormPerHeadLayer : public Layer {
 public:
  explicit L2NormPerHeadLayer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;
  base::Status forward(const half* input, half* output,
                       int num_heads, int head_dim, float eps);
};

// Compute GDN gates (softplus + exp + sigmoid)
class ComputeGDNGatesLayer : public Layer {
 public:
  explicit ComputeGDNGatesLayer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;
  base::Status forward(const half* alpha_raw, const half* dt_bias,
                       const float* A_log, const half* beta_raw,
                       float* gate_out, float* beta_out, int num_v_heads);
  base::Status forward_batched(const half* alpha_raw, const half* dt_bias,
                               const float* A_log, const half* beta_raw,
                               float* gate_out, float* beta_out,
                               int num_v_heads, int seq_len);
};

// GDN delta net decode step
class GDNDecodeStepLayer : public Layer {
 public:
  explicit GDNDecodeStepLayer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;
  base::Status forward(const half* q, const half* k, const half* v,
                       const float* gate, const float* beta,
                       float* state, half* output,
                       int num_k_heads, int num_v_heads,
                       int head_k_dim, int head_v_dim);
};

// Gated RMSNorm: RMSNorm(x, weight) * SiLU(z)
class GatedRMSNormLayer : public Layer {
 public:
  explicit GatedRMSNormLayer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;
  base::Status forward(const half* x, const half* z, const float* weight,
                       half* output, int dim, float eps);
  base::Status forward_batched(const half* x, const half* z, const float* weight,
                               half* output, int dim, int seq_len, float eps);
};

// Batched FP16 element-wise add
class BatchedAddFP16Layer : public Layer {
 public:
  explicit BatchedAddFP16Layer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;
  base::Status forward(const half* a, const half* b, half* output,
                       int dim, int seq_len);
};

// Batched RMSNorm FP16
class BatchedRMSNormFP16Layer : public Layer {
 public:
  explicit BatchedRMSNormFP16Layer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;
  base::Status forward(const half* input, half* output, const half* weight,
                       int dim, int seq_len, float eps);
  // Per-dimension variant (for Q/K norms)
  base::Status forward_dim(const half* input, half* output, const half* weight,
                           int dim, int total_rows, float eps);
};

// Gather strided data
class GatherStridedLayer : public Layer {
 public:
  explicit GatherStridedLayer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;
  base::Status forward(const half* src, half* dst,
                       int inner_dim, int outer_stride, int src_offset, int count);
};

// Transpose state between [v_head, dim1, dim2] and [v_head, dim2, dim1]
class TransposeStateLayer : public Layer {
 public:
  explicit TransposeStateLayer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;
  base::Status forward(const float* in, float* out,
                       int num_heads, int dim1, int dim2);
};

// GDN prefill with transposed state
class GDNPrefillTransposedLayer : public Layer {
 public:
  explicit GDNPrefillTransposedLayer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;
  base::Status forward(const half* q, const half* k, const half* v,
                       const float* gate, const float* beta,
                       float* state_transposed, half* output,
                       int num_k_heads, int num_v_heads,
                       int head_k_dim, int head_v_dim, int seq_len);
};

// Fused QKV GEMV: 3 GEMV (Q, K, V) in single kernel launch
class FusedQKVGemvLayer : public Layer {
 public:
  explicit FusedQKVGemvLayer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;
  base::Status forward(const half* input,
                       const half* q_weight, const half* k_weight, const half* v_weight,
                       half* q_output, half* k_output, half* v_output,
                       int dim, int q_dim, int kv_dim);
};

// Fused GDN projection GEMV: QKV + Z in single kernel launch
class FusedGDNProjGemvLayer : public Layer {
 public:
  explicit FusedGDNProjGemvLayer(base::DeviceType device_type);
  base::Status check() const override;
  base::Status forward() override;
  base::Status forward(const half* input,
                       const half* qkv_weight, const half* z_weight,
                       half* qkv_output, half* z_output,
                       int dim, int qkv_dim, int z_dim);
};

}  // namespace op

#endif  // KUIPER_INCLUDE_OP_GDN_LAYERS_H_
