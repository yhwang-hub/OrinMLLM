#ifndef KUIPER_SOURCE_OP_KERNELS_CUDA_GDN_KERNEL_CUH_
#define KUIPER_SOURCE_OP_KERNELS_CUDA_GDN_KERNEL_CUH_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace kernel {

/**
 * Causal depthwise Conv1D with SiLU activation.
 * Shifts conv_state left, inserts new_input, computes output = SiLU(conv(state)).
 *
 * @param conv_state   [conv_dim, kernel_size-1] FP16, in-place updated
 * @param new_input    [conv_dim] FP16, new input to insert
 * @param conv_weight  [conv_dim, kernel_size] FP16, depthwise conv weights
 * @param output       [conv_dim] FP16, output after conv+SiLU
 * @param conv_dim     Number of channels (e.g., 8192)
 * @param kernel_size  Conv kernel size (e.g., 4)
 */
void causal_conv1d_silu_cu(half* conv_state, const half* new_input,
                           const half* conv_weight, half* output,
                           int conv_dim, int kernel_size,
                           cudaStream_t stream);

/**
 * Batched causal Conv1D with SiLU for prefill.
 * Processes seq_len tokens through the conv, updating state.
 *
 * @param conv_state   [conv_dim, kernel_size-1] FP16, updated with last (kernel_size-1) tokens
 * @param input        [seq_len, conv_dim] FP16
 * @param conv_weight  [conv_dim, kernel_size] FP16
 * @param output       [seq_len, conv_dim] FP16
 */
void batched_causal_conv1d_silu_cu(half* conv_state, const half* input,
                                    const half* conv_weight, half* output,
                                    int conv_dim, int kernel_size, int seq_len,
                                    cudaStream_t stream);

/**
 * Per-head L2 normalization.
 * For each head: x_normalized = x / max(||x||_2, eps)
 *
 * @param input   [num_heads, head_dim] FP16
 * @param output  [num_heads, head_dim] FP16
 * @param num_heads  Number of heads
 * @param head_dim   Dimension per head
 * @param eps        Epsilon for numerical stability (default 1e-6)
 */
void l2_norm_per_head_cu(const half* input, half* output,
                         int num_heads, int head_dim, float eps,
                         cudaStream_t stream);

/**
 * Gated Delta Net decode step (single token, autoregressive).
 *
 * For each value head i:
 *   key_head = i / kv_mul;   // GQA mapping
 *   q_i = q[key_head]        // [head_k_dim]
 *   k_i = k[key_head]        // [head_k_dim]
 *   v_i = v[i]               // [head_v_dim]
 *   gate_i = exp(gate[i])    // scalar decay
 *   beta_i = beta[i]         // scalar write gate
 *   state[i] = state[i] * gate_i + beta_i * outer(k_i, v_i - state[i]^T @ k_i)
 *   output[i] = state[i]^T @ q_i  // [head_v_dim]
 *
 * @param q       [num_k_heads, head_k_dim] FP16, L2-normalized
 * @param k       [num_k_heads, head_k_dim] FP16, L2-normalized
 * @param v       [num_v_heads, head_v_dim] FP16
 * @param gate    [num_v_heads] FP32, pre-computed exp(softplus(alpha+dt_bias)*A_log)
 * @param beta    [num_v_heads] FP32, sigmoid(beta)
 * @param state   [num_v_heads, head_v_dim, head_k_dim] FP32, updated in-place
 * @param output  [num_v_heads, head_v_dim] FP16
 */
void gdn_decode_step_cu(const half* q, const half* k, const half* v,
                        const float* gate, const float* beta,
                        float* state, half* output,
                        int num_k_heads, int num_v_heads,
                        int head_k_dim, int head_v_dim,
                        cudaStream_t stream);

/**
 * Gated Delta Net prefill (chunked processing).
 * Processes seq_len tokens through the delta net, updating state.
 *
 * @param q       [seq_len, num_k_heads, head_k_dim] FP16
 * @param k       [seq_len, num_k_heads, head_k_dim] FP16
 * @param v       [seq_len, num_v_heads, head_v_dim] FP16
 * @param gate    [seq_len, num_v_heads] FP32
 * @param beta    [seq_len, num_v_heads] FP32
 * @param state   [num_v_heads, head_v_dim, head_k_dim] FP32, updated
 * @param output  [seq_len, num_v_heads, head_v_dim] FP16
 */
void gdn_prefill_cu(const half* q, const half* k, const half* v,
                    const float* gate, const float* beta,
                    float* state, half* output,
                    int num_k_heads, int num_v_heads,
                    int head_k_dim, int head_v_dim, int seq_len,
                    cudaStream_t stream);

/**
 * Gated RMSNorm: output = RMSNorm(x, weight) * SiLU(z)
 *
 * @param x       [dim] FP16
 * @param z       [dim] FP16 (gate)
 * @param weight  [dim] FP32 (norm weight)
 * @param output  [dim] FP16
 * @param dim     Dimension
 * @param eps     Epsilon
 */
void gated_rmsnorm_cu(const half* x, const half* z, const float* weight,
                      half* output, int dim, float eps,
                      cudaStream_t stream);

/**
 * Batched gated RMSNorm for prefill.
 *
 * @param x       [seq_len, dim] FP16
 * @param z       [seq_len, dim] FP16
 * @param weight  [dim] FP32
 * @param output  [seq_len, dim] FP16
 */
void batched_gated_rmsnorm_cu(const half* x, const half* z, const float* weight,
                              half* output, int dim, int seq_len, float eps,
                              cudaStream_t stream);

/**
 * Compute GDN gate values: gate = exp(softplus(alpha + dt_bias) * (-exp(A_log)))
 * And beta = sigmoid(beta_raw)
 *
 * @param alpha_raw  [num_v_heads] FP16, raw alpha projection
 * @param dt_bias    [num_v_heads] FP16
 * @param A_log      [num_v_heads] FP32
 * @param beta_raw   [num_v_heads] FP16, raw beta projection
 * @param gate_out   [num_v_heads] FP32, output gate (decay factor)
 * @param beta_out   [num_v_heads] FP32, output beta (write gate)
 */
void compute_gdn_gates_cu(const half* alpha_raw, const half* dt_bias,
                          const float* A_log, const half* beta_raw,
                          float* gate_out, float* beta_out,
                          int num_v_heads, cudaStream_t stream);

/**
 * Batched gate computation for prefill.
 */
void batched_compute_gdn_gates_cu(const half* alpha_raw, const half* dt_bias,
                                  const float* A_log, const half* beta_raw,
                                  float* gate_out, float* beta_out,
                                  int num_v_heads, int seq_len,
                                  cudaStream_t stream);

/**
 * Apply sigmoid gate to attention output: output *= sigmoid(gate)
 * For gated full-attention layers.
 *
 * @param attn_output  [dim] FP16, in-place modified
 * @param gate         [dim] FP16, sigmoid applied internally
 * @param dim          Total dimension
 */
void apply_sigmoid_gate_cu(half* attn_output, const half* gate, int dim,
                          cudaStream_t stream);

/**
 * Batched sigmoid gate for prefill.
 */
void batched_apply_sigmoid_gate_cu(half* attn_output, const half* gate,
                                   int dim, int seq_len, cudaStream_t stream);

/**
 * Partial RoPE: apply M-RoPE only to first partial_dim of each head,
 * leave remaining untouched. For interleaved rope format.
 *
 * @param q            [num_heads, head_dim] FP16
 * @param k            [num_kv_heads, head_dim] FP16  
 * @param sin_cache    RoPE sin cache
 * @param cos_cache    RoPE cos cache
 * @param pos_t/h/w    3D position indices
 * @param sections     [3] mrope sections e.g. [11,11,10]
 * @param partial_dim  Number of dims with RoPE (head_dim * partial_rotary_factor)
 * @param head_dim     Full head dimension
 * @param num_heads    Number of Q heads
 * @param num_kv_heads Number of KV heads
 */
void partial_mrope_interleaved_cu(half* q, half* k,
                                  const float* sin_cache, const float* cos_cache,
                                  int pos_t, int pos_h, int pos_w,
                                  const int* sections, int partial_dim,
                                  int head_dim, int num_heads, int num_kv_heads,
                                  cudaStream_t stream);

/**
 * Batched partial M-RoPE for prefill.
 */
void batched_partial_mrope_interleaved_cu(half* q, half* k,
                                          const float* sin_cache, const float* cos_cache,
                                          const int* pos_t, const int* pos_h, const int* pos_w,
                                          const int* sections, int partial_dim,
                                          int head_dim, int num_heads, int num_kv_heads,
                                          int seq_len, cudaStream_t stream);

/**
 * GPU-position variant of partial_mrope for CUDA Graph support.
 * Position is read from GPU memory pointed to by pos_gpu.
 */
void partial_mrope_interleaved_gpu_pos_cu(half* q, half* k,
                                          const float* sin_cache, const float* cos_cache,
                                          const int32_t* pos_gpu,
                                          const int* sections, int partial_dim,
                                          int head_dim, int num_heads, int num_kv_heads,
                                          cudaStream_t stream);

/**
 * GPU-position KV cache write kernel.
 * Reads position from GPU pointer and writes to correct KV cache offset.
 */
void kv_cache_write_gpu_pos_cu(half* kv_cache, const half* kv_data,
                                const int32_t* pos_gpu,
                                int kv_dim, int layer_idx, int max_seq_len,
                                cudaStream_t stream);

}  // namespace kernel

// =============================================================
// Batched helper kernels needed by Qwen3.5 prefill
// =============================================================
namespace kernel {

/**
 * Batched RMSNorm: apply RMSNorm to each row of [seq_len, dim] input.
 * weight is [dim] FP16.
 */
void batched_rmsnorm_fp16_cu(const half* input, half* output, const half* weight,
                             int dim, int seq_len, float eps, cudaStream_t stream);

/**
 * Batched per-dim RMSNorm: apply to [total_rows, dim] input (e.g., seq*heads, head_dim).
 */
void batched_rmsnorm_dim_fp16_cu(const half* input, half* output, const half* weight,
                                 int dim, int total_rows, float eps, cudaStream_t stream);

/**
 * Batched FP16 vector add: output[i] = a[i] + b[i] for total elements = dim * seq_len.
 */
void batched_add_fp16_cu(const half* a, const half* b, half* output,
                         int dim, int seq_len, cudaStream_t stream);

/**
 * Deinterleave Q and Gate from q_proj output.
 * Input layout per token: [h0_q(hd), h0_gate(hd), h1_q(hd), h1_gate(hd), ...]
 * Output: q_out = [h0_q(hd), h1_q(hd), ...], gate_out = [h0_gate(hd), h1_gate(hd), ...]
 *
 * @param interleaved  [seq_len, n_heads * head_dim * 2] FP16 input (q_proj output)
 * @param q_out        [seq_len, n_heads * head_dim] FP16 output
 * @param gate_out     [seq_len, n_heads * head_dim] FP16 output
 * @param n_heads      Number of heads
 * @param head_dim     Dimension per head
 * @param seq_len      Sequence length (1 for decode)
 */
void deinterleave_q_gate_cu(const half* interleaved, half* q_out, half* gate_out,
                            int n_heads, int head_dim, int seq_len,
                            cudaStream_t stream);

/**
 * Gather strided sub-arrays into contiguous output.
 * For each of `count` rows, copies `inner_dim` elements from
 * src[row * outer_stride + src_offset] to dst[row * inner_dim].
 *
 * Used to extract Q/K/V from interleaved conv output [seq_len, conv_dim].
 */
void gather_strided_cu(const half* src, half* dst,
                       int inner_dim, int outer_stride, int src_offset,
                       int count, cudaStream_t stream);

/**
 * Transpose state for GDN prefill optimization.
 * in:  [num_heads, dim1, dim2]   (standard: [v_head, v_dim, k_dim])
 * out: [num_heads, dim2, dim1]   (transposed: [v_head, k_dim, v_dim])
 * Enables coalesced memory access when threads iterate over v_dim.
 */
void transpose_state_cu(const float* in, float* out,
                        int num_heads, int dim1, int dim2,
                        cudaStream_t stream);

/**
 * Optimized GDN prefill with transposed state layout.
 * State is [v_head, k_dim, v_dim] instead of [v_head, v_dim, k_dim].
 * This ensures coalesced memory access: threads handling adjacent v-elements
 * access contiguous memory addresses.
 */
void gdn_prefill_transposed_cu(const half* q, const half* k, const half* v,
                                const float* gate, const float* beta,
                                float* state_transposed, half* output,
                                int num_k_heads, int num_v_heads,
                                int head_k_dim, int head_v_dim, int seq_len,
                                cudaStream_t stream);

// Fused FP16 QKV GEMV (single launch for Q, K, V projections)
void fused_fp16_qkv_gemv_cu(
    const half* input,
    const half* q_weight, const half* k_weight, const half* v_weight,
    half* q_output, half* k_output, half* v_output,
    int dim, int q_dim, int kv_dim, cudaStream_t stream);

// Fused FP16 GDN projections (qkv + z in single launch)
void fused_fp16_gdn_proj_gemv_cu(
    const half* input,
    const half* qkv_weight, const half* z_weight,
    half* qkv_output, half* z_output,
    int dim, int qkv_dim, int z_dim, cudaStream_t stream);

}  // namespace kernel

#endif  // KUIPER_SOURCE_OP_KERNELS_CUDA_GDN_KERNEL_CUH_
