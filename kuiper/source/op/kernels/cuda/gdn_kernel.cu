/**
 * @file gdn_kernel.cu
 * @brief CUDA kernels for Gated Delta Net (GDN) linear attention
 * 
 * Implements the core operations for the hybrid Qwen3.5 architecture:
 * - Causal depthwise Conv1D with SiLU
 * - Per-head L2 normalization
 * - Delta Net decode step (autoregressive)
 * - Delta Net prefill (sequential for correctness)
 * - Gated RMSNorm
 * - GDN gate computation (softplus + exp + sigmoid)
 * - Sigmoid gate for full attention output
 * - Partial interleaved M-RoPE
 */

#include "gdn_kernel.cuh"
#include <cuda_fp16.h>
#include <cmath>

namespace kernel {

// ============================================================
// Causal Conv1D with SiLU
// ============================================================

__global__ void causal_conv1d_silu_kernel(half* conv_state, const half* new_input,
                                          const half* conv_weight, half* output,
                                          int conv_dim, int kernel_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= conv_dim) return;
  
  int state_cols = kernel_size - 1;
  
  // Compute convolution FIRST (before updating state)
  // state holds [input[t-K+1], ..., input[t-1]], and new_input is input[t]
  // Conv: sum = state[0]*w[0] + state[1]*w[1] + ... + state[K-2]*w[K-2] + new_input*w[K-1]
  float sum = 0.0f;
  for (int j = 0; j < state_cols; ++j) {
    sum += __half2float(conv_state[idx * state_cols + j]) * 
           __half2float(conv_weight[idx * kernel_size + j]);
  }
  sum += __half2float(new_input[idx]) * __half2float(conv_weight[idx * kernel_size + kernel_size - 1]);
  
  // THEN update state: shift left, insert new input
  for (int j = 0; j < state_cols - 1; ++j) {
    conv_state[idx * state_cols + j] = conv_state[idx * state_cols + j + 1];
  }
  conv_state[idx * state_cols + state_cols - 1] = new_input[idx];
  
  // SiLU activation: x * sigmoid(x)
  float sigmoid_val = 1.0f / (1.0f + expf(-sum));
  output[idx] = __float2half(sum * sigmoid_val);
}

void causal_conv1d_silu_cu(half* conv_state, const half* new_input,
                           const half* conv_weight, half* output,
                           int conv_dim, int kernel_size,
                           cudaStream_t stream) {
  int threads = 256;
  int blocks = (conv_dim + threads - 1) / threads;
  causal_conv1d_silu_kernel<<<blocks, threads, 0, stream>>>(
      conv_state, new_input, conv_weight, output, conv_dim, kernel_size);
}

// Batched version for prefill
__global__ void batched_causal_conv1d_silu_kernel(half* conv_state, const half* input,
                                                   const half* conv_weight, half* output,
                                                   int conv_dim, int kernel_size, int seq_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= conv_dim) return;
  
  int state_cols = kernel_size - 1;
  
  // Process each token sequentially
  for (int t = 0; t < seq_len; ++t) {
    half new_val = input[t * conv_dim + idx];
    
    // Compute convolution FIRST (before updating state)
    float sum = 0.0f;
    for (int j = 0; j < state_cols; ++j) {
      sum += __half2float(conv_state[idx * state_cols + j]) * 
             __half2float(conv_weight[idx * kernel_size + j]);
    }
    sum += __half2float(new_val) * __half2float(conv_weight[idx * kernel_size + kernel_size - 1]);
    
    // THEN update state: shift left, insert new input
    for (int j = 0; j < state_cols - 1; ++j) {
      conv_state[idx * state_cols + j] = conv_state[idx * state_cols + j + 1];
    }
    conv_state[idx * state_cols + state_cols - 1] = new_val;
    
    // SiLU
    float sigmoid_val = 1.0f / (1.0f + expf(-sum));
    output[t * conv_dim + idx] = __float2half(sum * sigmoid_val);
  }
}

void batched_causal_conv1d_silu_cu(half* conv_state, const half* input,
                                    const half* conv_weight, half* output,
                                    int conv_dim, int kernel_size, int seq_len,
                                    cudaStream_t stream) {
  int threads = 256;
  int blocks = (conv_dim + threads - 1) / threads;
  batched_causal_conv1d_silu_kernel<<<blocks, threads, 0, stream>>>(
      conv_state, input, conv_weight, output, conv_dim, kernel_size, seq_len);
}

// ============================================================
// Per-head L2 Normalization
// ============================================================

__global__ void l2_norm_per_head_kernel(const half* input, half* output,
                                         int num_heads, int head_dim, float eps) {
  int head_idx = blockIdx.x;
  if (head_idx >= num_heads) return;
  
  const half* head_in = input + head_idx * head_dim;
  half* head_out = output + head_idx * head_dim;
  
  // Compute L2 norm using shared memory reduction
  extern __shared__ float sdata[];
  
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float val = __half2float(head_in[i]);
    local_sum += val * val;
  }
  sdata[threadIdx.x] = local_sum;
  __syncthreads();
  
  // Warp reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }
  
  // HF: inv_norm = rsqrt(sum(x*x) + eps) — matches FLA library l2norm
  float inv_norm = rsqrtf(sdata[0] + eps);
  
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float val = __half2float(head_in[i]);
    head_out[i] = __float2half(val * inv_norm);
  }
}

void l2_norm_per_head_cu(const half* input, half* output,
                         int num_heads, int head_dim, float eps,
                         cudaStream_t stream) {
  int threads = min(256, head_dim);
  // Round up to power of 2 for reduction
  int t = 1;
  while (t < threads) t <<= 1;
  threads = t;
  if (threads > 256) threads = 256;
  
  l2_norm_per_head_kernel<<<num_heads, threads, threads * sizeof(float), stream>>>(
      input, output, num_heads, head_dim, eps);
}

// ============================================================
// GDN Decode Step (Autoregressive Delta Net)
// ============================================================

// Each block handles one value head
__global__ void gdn_decode_step_kernel(const half* q, const half* k, const half* v,
                                        const float* gate, const float* beta,
                                        float* state, half* output,
                                        int num_k_heads, int num_v_heads,
                                        int head_k_dim, int head_v_dim) {
  int v_head = blockIdx.x;
  if (v_head >= num_v_heads) return;
  
  int kv_mul = num_v_heads / num_k_heads;
  int k_head = v_head / kv_mul;
  
  float gate_val = gate[v_head];  // Already exp(softplus(alpha+dt_bias) * A)
  float beta_val = beta[v_head];  // Already sigmoid(beta_raw)
  float q_scale = rsqrtf((float)head_k_dim);  // 1/sqrt(k_head_dim)
  
  // State shape: [num_v_heads, head_v_dim, head_k_dim]
  float* state_head = state + v_head * head_v_dim * head_k_dim;
  
  const half* q_head = q + k_head * head_k_dim;
  const half* k_head_ptr = k + k_head * head_k_dim;
  const half* v_head_ptr = v + v_head * head_v_dim;
  half* out_head = output + v_head * head_v_dim;
  
  // HF reference order:
  // 1. Decay state: state' = state * gate
  // 2. Compute memory: kv_mem = state' @ k
  // 3. Delta: delta = beta * (v - kv_mem)
  // 4. Update: state = state' + outer(k, delta)
  // 5. Output: output = state @ (q * scale)
  
  for (int vi = threadIdx.x; vi < head_v_dim; vi += blockDim.x) {
    float* state_row = state_head + vi * head_k_dim;
    
    // 1. Decay state and compute kv_mem from DECAYED state
    float sk_dot = 0.0f;
    for (int kj = 0; kj < head_k_dim; ++kj) {
      state_row[kj] *= gate_val;
      sk_dot += state_row[kj] * __half2float(k_head_ptr[kj]);
    }
    
    // 2. Delta from decayed state
    float v_val = __half2float(v_head_ptr[vi]);
    float delta = beta_val * (v_val - sk_dot);
    
    // 3. Update state and compute scaled output
    float dot_q = 0.0f;
    for (int kj = 0; kj < head_k_dim; ++kj) {
      float k_val = __half2float(k_head_ptr[kj]);
      state_row[kj] += delta * k_val;
      dot_q += state_row[kj] * (__half2float(q_head[kj]) * q_scale);
    }
    
    out_head[vi] = __float2half(dot_q);
  }
}

void gdn_decode_step_cu(const half* q, const half* k, const half* v,
                        const float* gate, const float* beta,
                        float* state, half* output,
                        int num_k_heads, int num_v_heads,
                        int head_k_dim, int head_v_dim,
                        cudaStream_t stream) {
  // One block per value head, threads process v-dimensions
  int threads = min(128, head_v_dim);
  gdn_decode_step_kernel<<<num_v_heads, threads, 0, stream>>>(
      q, k, v, gate, beta, state, output, num_k_heads, num_v_heads, head_k_dim, head_v_dim);
}

// ============================================================
// GDN Prefill (Sequential for correctness)
// ============================================================

// Process tokens sequentially, each token does the decode step
__global__ void gdn_prefill_kernel(const half* q, const half* k, const half* v,
                                    const float* gate, const float* beta,
                                    float* state, half* output,
                                    int num_k_heads, int num_v_heads,
                                    int head_k_dim, int head_v_dim, int seq_len) {
  int v_head = blockIdx.x;
  if (v_head >= num_v_heads) return;
  
  int kv_mul = num_v_heads / num_k_heads;
  int k_head = v_head / kv_mul;
  float q_scale = rsqrtf((float)head_k_dim);  // 1/sqrt(k_head_dim)
  
  float* state_head = state + v_head * head_v_dim * head_k_dim;
  
  for (int t = 0; t < seq_len; ++t) {
    float gate_val = gate[t * num_v_heads + v_head];
    float beta_val = beta[t * num_v_heads + v_head];
    
    const half* q_t = q + (t * num_k_heads + k_head) * head_k_dim;
    const half* k_t = k + (t * num_k_heads + k_head) * head_k_dim;
    const half* v_t = v + (t * num_v_heads + v_head) * head_v_dim;
    half* out_t = output + (t * num_v_heads + v_head) * head_v_dim;
    
    for (int vi = threadIdx.x; vi < head_v_dim; vi += blockDim.x) {
      float* state_row = state_head + vi * head_k_dim;
      
      // 1. Decay state and compute kv_mem from DECAYED state
      float sk_dot = 0.0f;
      for (int kj = 0; kj < head_k_dim; ++kj) {
        state_row[kj] *= gate_val;
        sk_dot += state_row[kj] * __half2float(k_t[kj]);
      }
      
      // 2. Delta from decayed state
      float v_val = __half2float(v_t[vi]);
      float delta = beta_val * (v_val - sk_dot);
      
      // 3. Update and compute scaled output
      float dot_q = 0.0f;
      for (int kj = 0; kj < head_k_dim; ++kj) {
        float k_val = __half2float(k_t[kj]);
        state_row[kj] += delta * k_val;
        dot_q += state_row[kj] * (__half2float(q_t[kj]) * q_scale);
      }
      
      out_t[vi] = __float2half(dot_q);
    }
    __syncthreads();
  }
}

void gdn_prefill_cu(const half* q, const half* k, const half* v,
                    const float* gate, const float* beta,
                    float* state, half* output,
                    int num_k_heads, int num_v_heads,
                    int head_k_dim, int head_v_dim, int seq_len,
                    cudaStream_t stream) {
  int threads = min(128, head_v_dim);
  gdn_prefill_kernel<<<num_v_heads, threads, 0, stream>>>(
      q, k, v, gate, beta, state, output, num_k_heads, num_v_heads, head_k_dim, head_v_dim, seq_len);
}

// ============================================================
// Gated RMSNorm: RMSNorm(x) * SiLU(z)
// ============================================================

// Gated RMSNorm: per-head RMSNorm(x) * SiLU(z)
// x: [n_heads * head_dim], weight: [head_dim], z: [n_heads * head_dim]
// RMS computed independently per-head over head_dim elements
__global__ void gated_rmsnorm_kernel(const half* x, const half* z, const float* weight,
                                      half* output, int n_heads, int head_dim, float eps) {
  // One block per head
  int head = blockIdx.x;
  if (head >= n_heads) return;
  
  extern __shared__ float sdata[];
  
  const half* x_h = x + head * head_dim;
  const half* z_h = z + head * head_dim;
  half* out_h = output + head * head_dim;
  
  // Compute RMS for this head
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float val = __half2float(x_h[i]);
    local_sum += val * val;
  }
  sdata[threadIdx.x] = local_sum;
  __syncthreads();
  
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }
  
  float rms = sqrtf(sdata[0] / head_dim + eps);
  float inv_rms = 1.0f / rms;
  
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float x_val = __half2float(x_h[i]) * inv_rms * weight[i];
    float z_val = __half2float(z_h[i]);
    float silu_z = z_val / (1.0f + expf(-z_val));
    out_h[i] = __float2half(x_val * silu_z);
  }
}

void gated_rmsnorm_cu(const half* x, const half* z, const float* weight,
                      half* output, int dim, float eps,
                      cudaStream_t stream) {
  // dim = n_heads * head_dim, but we need both values
  // For Qwen3.5: n_v_heads=32, v_head_dim=128, dim=4096
  // We pass dim and compute from the weight size
  // Updated signature: pass n_heads and head_dim separately
  // For backward compat, dim is total (n_heads * head_dim)
  // Launch one block per head
  int n_heads = 32;  // TODO: make configurable
  int head_dim = dim / n_heads;
  int threads = min(256, head_dim);
  int t = 1;
  while (t < threads) t <<= 1;
  threads = t;
  if (threads > 256) threads = 256;
  
  gated_rmsnorm_kernel<<<n_heads, threads, threads * sizeof(float), stream>>>(
      x, z, weight, output, n_heads, head_dim, eps);
}

// Batched version: x: [seq_len, n_heads * head_dim], weight: [head_dim], z: [seq_len, n_heads * head_dim]
__global__ void batched_gated_rmsnorm_kernel(const half* x, const half* z, const float* weight,
                                              half* output, int n_heads, int head_dim, int seq_len, float eps) {
  // Grid: (n_heads, seq_len)
  int head = blockIdx.x;
  int token_idx = blockIdx.y;
  if (head >= n_heads || token_idx >= seq_len) return;
  
  extern __shared__ float sdata[];
  int dim = n_heads * head_dim;
  
  const half* x_t = x + token_idx * dim + head * head_dim;
  const half* z_t = z + token_idx * dim + head * head_dim;
  half* out_t = output + token_idx * dim + head * head_dim;
  
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float val = __half2float(x_t[i]);
    local_sum += val * val;
  }
  sdata[threadIdx.x] = local_sum;
  __syncthreads();
  
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }
  
  float rms = sqrtf(sdata[0] / head_dim + eps);
  float inv_rms = 1.0f / rms;
  
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float x_val = __half2float(x_t[i]) * inv_rms * weight[i];
    float z_val = __half2float(z_t[i]);
    float silu_z = z_val / (1.0f + expf(-z_val));
    out_t[i] = __float2half(x_val * silu_z);
  }
}

void batched_gated_rmsnorm_cu(const half* x, const half* z, const float* weight,
                              half* output, int dim, int seq_len, float eps,
                              cudaStream_t stream) {
  int n_heads = 32;
  int head_dim = dim / n_heads;
  int threads = min(256, head_dim);
  int t = 1;
  while (t < threads) t <<= 1;
  threads = t;
  if (threads > 256) threads = 256;
  
  dim3 grid(n_heads, seq_len);
  batched_gated_rmsnorm_kernel<<<grid, threads, threads * sizeof(float), stream>>>(
      x, z, weight, output, n_heads, head_dim, seq_len, eps);
}

// ============================================================
// GDN Gate Computation: gate = exp(softplus(alpha + dt_bias) * (-exp(A_log)))
//                       beta = sigmoid(beta_raw)
// ============================================================

__global__ void compute_gdn_gates_kernel(const half* alpha_raw, const half* dt_bias,
                                          const float* A_log, const half* beta_raw,
                                          float* gate_out, float* beta_out,
                                          int num_v_heads) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_v_heads) return;
  
  float alpha = __half2float(alpha_raw[idx]) + __half2float(dt_bias[idx]);
  float softplus_alpha = logf(1.0f + expf(alpha));
  float neg_exp_A = -expf(A_log[idx]);
  gate_out[idx] = expf(softplus_alpha * neg_exp_A);
  
  float beta = __half2float(beta_raw[idx]);
  beta_out[idx] = 1.0f / (1.0f + expf(-beta));
}

void compute_gdn_gates_cu(const half* alpha_raw, const half* dt_bias,
                          const float* A_log, const half* beta_raw,
                          float* gate_out, float* beta_out,
                          int num_v_heads, cudaStream_t stream) {
  int threads = min(256, num_v_heads);
  int blocks = (num_v_heads + threads - 1) / threads;
  compute_gdn_gates_kernel<<<blocks, threads, 0, stream>>>(
      alpha_raw, dt_bias, A_log, beta_raw, gate_out, beta_out, num_v_heads);
}

__global__ void batched_compute_gdn_gates_kernel(const half* alpha_raw, const half* dt_bias,
                                                  const float* A_log, const half* beta_raw,
                                                  float* gate_out, float* beta_out,
                                                  int num_v_heads, int seq_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_v_heads * seq_len;
  if (idx >= total) return;
  
  int head_idx = idx % num_v_heads;
  
  float alpha = __half2float(alpha_raw[idx]) + __half2float(dt_bias[head_idx]);
  float softplus_alpha = logf(1.0f + expf(alpha));
  float neg_exp_A = -expf(A_log[head_idx]);
  gate_out[idx] = expf(softplus_alpha * neg_exp_A);
  
  float beta = __half2float(beta_raw[idx]);
  beta_out[idx] = 1.0f / (1.0f + expf(-beta));
}

void batched_compute_gdn_gates_cu(const half* alpha_raw, const half* dt_bias,
                                  const float* A_log, const half* beta_raw,
                                  float* gate_out, float* beta_out,
                                  int num_v_heads, int seq_len,
                                  cudaStream_t stream) {
  int total = num_v_heads * seq_len;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  batched_compute_gdn_gates_kernel<<<blocks, threads, 0, stream>>>(
      alpha_raw, dt_bias, A_log, beta_raw, gate_out, beta_out, num_v_heads, seq_len);
}

// ============================================================
// Sigmoid Gate for full attention output
// ============================================================

__global__ void apply_sigmoid_gate_kernel(half* attn_output, const half* gate, int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= dim) return;
  
  float out_val = __half2float(attn_output[idx]);
  float gate_val = __half2float(gate[idx]);
  float sigmoid_gate = 1.0f / (1.0f + expf(-gate_val));
  attn_output[idx] = __float2half(out_val * sigmoid_gate);
}

void apply_sigmoid_gate_cu(half* attn_output, const half* gate, int dim,
                          cudaStream_t stream) {
  int threads = 256;
  int blocks = (dim + threads - 1) / threads;
  apply_sigmoid_gate_kernel<<<blocks, threads, 0, stream>>>(attn_output, gate, dim);
}

void batched_apply_sigmoid_gate_cu(half* attn_output, const half* gate,
                                   int dim, int seq_len, cudaStream_t stream) {
  int total = dim * seq_len;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  // Reuse single kernel since the data is contiguous
  apply_sigmoid_gate_kernel<<<blocks, threads, 0, stream>>>(attn_output, gate, total);
}

// ============================================================
// Partial Half-Split M-RoPE
// For full attention layers: only rotate first partial_dim of each head
// Using half-split RoPE format: pair (q[i], q[i+half]) where half = partial_dim/2
// MRoPE section interleaving: [T,H,W,T,H,W,...] pattern
// ============================================================

__global__ void partial_mrope_interleaved_kernel(half* q, half* k,
                                                  const float* sin_cache, const float* cos_cache,
                                                  int pos_t, int pos_h, int pos_w,
                                                  int section0, int section1, int section2,
                                                  int partial_dim, int head_dim,
                                                  int num_heads, int num_kv_heads) {
  int total_heads = num_heads + num_kv_heads;
  int pair_idx = threadIdx.x;
  int head_idx = blockIdx.x;
  
  if (head_idx >= total_heads) return;
  
  int num_pairs = partial_dim / 2;
  if (pair_idx >= num_pairs) return;
  
  // Interleaved MRoPE: pair i%3==0 → T, i%3==1 → H, i%3==2 → W
  // With sections [11,11,10]: last pair 30→T, 31→H (W runs out)
  int pos;
  int mod3 = pair_idx % 3;
  int div3 = pair_idx / 3;
  if (mod3 == 0 && div3 < section0) {
    pos = pos_t;
  } else if (mod3 == 1 && div3 < section1) {
    pos = pos_h;
  } else if (mod3 == 2 && div3 < section2) {
    pos = pos_w;
  } else {
    // Fallback for remaining pairs: use T
    pos = pos_t;
  }
  
  // Cache layout: [max_seq_len, num_pairs]
  float sin_val = sin_cache[pos * num_pairs + pair_idx];
  float cos_val = cos_cache[pos * num_pairs + pair_idx];
  
  // Half-split format: pair (element[i], element[i + num_pairs])
  half* target;
  if (head_idx < num_heads) {
    target = q + head_idx * head_dim;
  } else {
    target = k + (head_idx - num_heads) * head_dim;
  }
  
  int idx0 = pair_idx;
  int idx1 = pair_idx + num_pairs;
  float re = __half2float(target[idx0]);
  float im = __half2float(target[idx1]);
  
  // Rotation: (re, im) -> (re*cos - im*sin, re*sin + im*cos)
  target[idx0] = __float2half(re * cos_val - im * sin_val);
  target[idx1] = __float2half(re * sin_val + im * cos_val);
}

void partial_mrope_interleaved_cu(half* q, half* k,
                                  const float* sin_cache, const float* cos_cache,
                                  int pos_t, int pos_h, int pos_w,
                                  const int* sections, int partial_dim,
                                  int head_dim, int num_heads, int num_kv_heads,
                                  cudaStream_t stream) {
  int total_pairs = partial_dim / 2;
  int total_heads = num_heads + num_kv_heads;
  
  // sections is on CPU
  int section0 = sections[0];
  int section1 = sections[1];
  int section2 = sections[2];
  
  partial_mrope_interleaved_kernel<<<total_heads, total_pairs, 0, stream>>>(
      q, k, sin_cache, cos_cache, pos_t, pos_h, pos_w,
      section0, section1, section2, partial_dim, head_dim, num_heads, num_kv_heads);
}

__global__ void batched_partial_mrope_interleaved_kernel(half* q, half* k,
                                                          const float* sin_cache, const float* cos_cache,
                                                          const int* pos_t_arr, const int* pos_h_arr, const int* pos_w_arr,
                                                          int section0, int section1, int section2,
                                                          int partial_dim, int head_dim,
                                                          int num_heads, int num_kv_heads, int seq_len) {
  // Grid: (total_heads, seq_len), threads: num_pairs
  int head_idx = blockIdx.x;
  int token_idx = blockIdx.y;
  int pair_idx = threadIdx.x;
  
  int total_heads = num_heads + num_kv_heads;
  if (head_idx >= total_heads || token_idx >= seq_len) return;
  
  int num_pairs = partial_dim / 2;
  if (pair_idx >= num_pairs) return;
  
  // Interleaved MRoPE section pattern
  int pos;
  int mod3 = pair_idx % 3;
  int div3 = pair_idx / 3;
  if (mod3 == 0 && div3 < section0) {
    pos = pos_t_arr[token_idx];
  } else if (mod3 == 1 && div3 < section1) {
    pos = pos_h_arr[token_idx];
  } else if (mod3 == 2 && div3 < section2) {
    pos = pos_w_arr[token_idx];
  } else {
    pos = pos_t_arr[token_idx];
  }
  
  // Cache layout: [max_seq_len, num_pairs]
  float sin_val = sin_cache[pos * num_pairs + pair_idx];
  float cos_val = cos_cache[pos * num_pairs + pair_idx];
  
  // Half-split format: pair (element[i], element[i + num_pairs])
  half* target;
  if (head_idx < num_heads) {
    target = q + (token_idx * num_heads + head_idx) * head_dim;
  } else {
    target = k + (token_idx * num_kv_heads + (head_idx - num_heads)) * head_dim;
  }
  
  int idx0 = pair_idx;
  int idx1 = pair_idx + num_pairs;
  float re = __half2float(target[idx0]);
  float im = __half2float(target[idx1]);
  
  target[idx0] = __float2half(re * cos_val - im * sin_val);
  target[idx1] = __float2half(re * sin_val + im * cos_val);
}

void batched_partial_mrope_interleaved_cu(half* q, half* k,
                                          const float* sin_cache, const float* cos_cache,
                                          const int* pos_t, const int* pos_h, const int* pos_w,
                                          const int* sections, int partial_dim,
                                          int head_dim, int num_heads, int num_kv_heads,
                                          int seq_len, cudaStream_t stream) {
  int total_pairs = partial_dim / 2;
  int total_heads = num_heads + num_kv_heads;
  
  int section0 = sections[0];
  int section1 = sections[1];
  int section2 = sections[2];
  
  dim3 grid(total_heads, seq_len);
  batched_partial_mrope_interleaved_kernel<<<grid, total_pairs, 0, stream>>>(
      q, k, sin_cache, cos_cache, pos_t, pos_h, pos_w,
      section0, section1, section2, partial_dim, head_dim, num_heads, num_kv_heads, seq_len);
}

// GPU-position variant of partial MRoPE for CUDA Graph
__global__ void partial_mrope_interleaved_gpu_pos_kernel(half* q, half* k,
    const float* sin_cache, const float* cos_cache,
    const int32_t* pos_gpu,
    int section0, int section1, int section2,
    int partial_dim, int head_dim, int num_heads, int num_kv_heads) {
  int total_heads = num_heads + num_kv_heads;
  int pair_idx = threadIdx.x;
  int head_idx = blockIdx.x;
  if (head_idx >= total_heads) return;
  int num_pairs = partial_dim / 2;
  if (pair_idx >= num_pairs) return;

  int text_pos = pos_gpu[0];  // Read position from GPU memory

  int pos;
  int mod3 = pair_idx % 3;
  int div3 = pair_idx / 3;
  if (mod3 == 0 && div3 < section0) pos = text_pos;
  else if (mod3 == 1 && div3 < section1) pos = text_pos;
  else if (mod3 == 2 && div3 < section2) pos = text_pos;
  else pos = text_pos;

  float sin_val = sin_cache[pos * num_pairs + pair_idx];
  float cos_val = cos_cache[pos * num_pairs + pair_idx];

  half* target;
  if (head_idx < num_heads) target = q + head_idx * head_dim;
  else target = k + (head_idx - num_heads) * head_dim;

  int idx0 = pair_idx;
  int idx1 = pair_idx + num_pairs;
  float re = __half2float(target[idx0]);
  float im = __half2float(target[idx1]);
  target[idx0] = __float2half(re * cos_val - im * sin_val);
  target[idx1] = __float2half(re * sin_val + im * cos_val);
}

void partial_mrope_interleaved_gpu_pos_cu(half* q, half* k,
    const float* sin_cache, const float* cos_cache,
    const int32_t* pos_gpu,
    const int* sections, int partial_dim,
    int head_dim, int num_heads, int num_kv_heads,
    cudaStream_t stream) {
  int total_pairs = partial_dim / 2;
  int total_heads = num_heads + num_kv_heads;
  int section0 = sections[0], section1 = sections[1], section2 = sections[2];
  partial_mrope_interleaved_gpu_pos_kernel<<<total_heads, total_pairs, 0, stream>>>(
      q, k, sin_cache, cos_cache, pos_gpu,
      section0, section1, section2, partial_dim, head_dim, num_heads, num_kv_heads);
}

// GPU-position KV cache write kernel
__global__ void kv_cache_write_gpu_pos_kernel(half* kv_cache, const half* kv_data,
    const int32_t* pos_gpu, int kv_dim, int layer_idx, int max_seq_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= kv_dim) return;
  int pos = pos_gpu[0];
  size_t offset = (static_cast<size_t>(layer_idx) * max_seq_len + pos) * kv_dim + idx;
  kv_cache[offset] = kv_data[idx];
}

void kv_cache_write_gpu_pos_cu(half* kv_cache, const half* kv_data,
    const int32_t* pos_gpu, int kv_dim, int layer_idx, int max_seq_len,
    cudaStream_t stream) {
  int threads = 256;
  int blocks = (kv_dim + threads - 1) / threads;
  kv_cache_write_gpu_pos_kernel<<<blocks, threads, 0, stream>>>(
      kv_cache, kv_data, pos_gpu, kv_dim, layer_idx, max_seq_len);
}

}  // namespace kernel

// =============================================================
// Batched helpers for Qwen3.5 prefill
// =============================================================
namespace kernel {

__global__ void batched_rmsnorm_fp16_kernel(const half* input, half* output, const half* weight,
                                             int dim, int seq_len, float eps) {
  int row = blockIdx.x;
  if (row >= seq_len) return;
  extern __shared__ float sdata[];
  
  const half* in_row = input + row * dim;
  half* out_row = output + row * dim;
  
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = __half2float(in_row[i]);
    local_sum += v * v;
  }
  sdata[threadIdx.x] = local_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  float inv_rms = rsqrtf(sdata[0] / dim + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    out_row[i] = __float2half(__half2float(in_row[i]) * inv_rms * __half2float(weight[i]));
  }
}

void batched_rmsnorm_fp16_cu(const half* input, half* output, const half* weight,
                             int dim, int seq_len, float eps, cudaStream_t stream) {
  int threads = min(256, dim);
  int t = 1; while (t < threads) t <<= 1; threads = min(t, 256);
  batched_rmsnorm_fp16_kernel<<<seq_len, threads, threads * sizeof(float), stream>>>(
      input, output, weight, dim, seq_len, eps);
}

void batched_rmsnorm_dim_fp16_cu(const half* input, half* output, const half* weight,
                                 int dim, int total_rows, float eps, cudaStream_t stream) {
  // Same kernel, just different interpretation (total_rows instead of seq_len)
  batched_rmsnorm_fp16_cu(input, output, weight, dim, total_rows, eps, stream);
}

__global__ void batched_add_fp16_kernel(const half* a, const half* b, half* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  output[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
}

void batched_add_fp16_cu(const half* a, const half* b, half* output,
                         int dim, int seq_len, cudaStream_t stream) {
  int n = dim * seq_len;
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  batched_add_fp16_kernel<<<blocks, threads, 0, stream>>>(a, b, output, n);
}

// ============================================================
// Q/Gate Deinterleave
// ============================================================

__global__ void deinterleave_q_gate_kernel(const half* interleaved, half* q_out, half* gate_out,
                                            int n_heads, int head_dim, int seq_len) {
  // Thread per output element
  int total = seq_len * n_heads * head_dim;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  int q_dim = n_heads * head_dim;
  int t = idx / q_dim;          // token index
  int rem = idx % q_dim;
  int h = rem / head_dim;        // head index
  int d = rem % head_dim;        // dim within head

  // Input layout per token: [h0_q(hd), h0_gate(hd), h1_q(hd), h1_gate(hd), ...]
  int src_q_offset = t * (n_heads * head_dim * 2) + h * head_dim * 2 + d;
  int src_g_offset = src_q_offset + head_dim;

  q_out[idx] = interleaved[src_q_offset];
  gate_out[idx] = interleaved[src_g_offset];
}

void deinterleave_q_gate_cu(const half* interleaved, half* q_out, half* gate_out,
                            int n_heads, int head_dim, int seq_len,
                            cudaStream_t stream) {
  int total = seq_len * n_heads * head_dim;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  deinterleave_q_gate_kernel<<<blocks, threads, 0, stream>>>(
      interleaved, q_out, gate_out, n_heads, head_dim, seq_len);
}

// ============================================================
// Strided Gather
// ============================================================

__global__ void gather_strided_kernel(const half* src, half* dst,
                                       int inner_dim, int outer_stride,
                                       int src_offset, int total_elems) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elems) return;
  int row = idx / inner_dim;
  int col = idx % inner_dim;
  dst[idx] = src[row * outer_stride + src_offset + col];
}

void gather_strided_cu(const half* src, half* dst,
                       int inner_dim, int outer_stride, int src_offset,
                       int count, cudaStream_t stream) {
  int total = count * inner_dim;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  gather_strided_kernel<<<blocks, threads, 0, stream>>>(
      src, dst, inner_dim, outer_stride, src_offset, total);
}

// ============================================================
// State Transpose for GDN Prefill Optimization
// ============================================================

__global__ void transpose_state_kernel(const float* in, float* out,
                                        int num_heads, int dim1, int dim2) {
  // in: [num_heads, dim1, dim2], out: [num_heads, dim2, dim1]
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int plane_size = dim1 * dim2;
  int total = num_heads * plane_size;
  if (idx >= total) return;

  int head = idx / plane_size;
  int rem = idx % plane_size;
  int r = rem / dim2;
  int c = rem % dim2;

  out[head * plane_size + c * dim1 + r] = in[idx];
}

void transpose_state_cu(const float* in, float* out,
                        int num_heads, int dim1, int dim2,
                        cudaStream_t stream) {
  int total = num_heads * dim1 * dim2;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  transpose_state_kernel<<<blocks, threads, 0, stream>>>(
      in, out, num_heads, dim1, dim2);
}

// ============================================================
// Optimized GDN Prefill with Transposed State
// ============================================================

// State layout: [v_head, k_dim, v_dim] — threads over v_dim are coalesced
__global__ void gdn_prefill_transposed_kernel(
    const half* q, const half* k, const half* v,
    const float* gate, const float* beta,
    float* state, half* output,
    int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int seq_len) {

  int v_head = blockIdx.x;
  if (v_head >= num_v_heads) return;

  int kv_mul = num_v_heads / num_k_heads;
  int k_head = v_head / kv_mul;
  float q_scale = rsqrtf((float)head_k_dim);

  // state_head points to transposed state: [k_dim, v_dim]
  float* state_head = state + v_head * head_k_dim * head_v_dim;

  extern __shared__ float smem[];
  float* sh_k = smem;                    // [head_k_dim]
  float* sh_q = smem + head_k_dim;       // [head_k_dim]

  for (int t = 0; t < seq_len; ++t) {
    const half* k_t = k + (t * num_k_heads + k_head) * head_k_dim;
    const half* q_t = q + (t * num_k_heads + k_head) * head_k_dim;

    // Cooperatively load k and q into shared memory
    for (int i = threadIdx.x; i < head_k_dim; i += blockDim.x) {
      sh_k[i] = __half2float(k_t[i]);
      sh_q[i] = __half2float(q_t[i]) * q_scale;
    }
    __syncthreads();

    float gate_val = gate[t * num_v_heads + v_head];
    float beta_val = beta[t * num_v_heads + v_head];

    for (int vi = threadIdx.x; vi < head_v_dim; vi += blockDim.x) {
      // Pass 1: Decay state and compute sk_dot
      // Access: state_head[kj * head_v_dim + vi] — COALESCED across threads
      float sk_dot = 0.0f;
      for (int kj = 0; kj < head_k_dim; ++kj) {
        float* s_ptr = state_head + kj * head_v_dim + vi;
        float s = *s_ptr * gate_val;
        *s_ptr = s;
        sk_dot += s * sh_k[kj];
      }

      // Delta rule
      float v_val = __half2float(v[(t * num_v_heads + v_head) * head_v_dim + vi]);
      float delta = beta_val * (v_val - sk_dot);

      // Pass 2: Update state and compute output
      float dot_q = 0.0f;
      for (int kj = 0; kj < head_k_dim; ++kj) {
        float* s_ptr = state_head + kj * head_v_dim + vi;
        float s = *s_ptr + delta * sh_k[kj];
        *s_ptr = s;
        dot_q += s * sh_q[kj];
      }

      output[(t * num_v_heads + v_head) * head_v_dim + vi] = __float2half(dot_q);
    }
    __syncthreads();
  }
}

void gdn_prefill_transposed_cu(const half* q, const half* k, const half* v,
                                const float* gate, const float* beta,
                                float* state_transposed, half* output,
                                int num_k_heads, int num_v_heads,
                                int head_k_dim, int head_v_dim, int seq_len,
                                cudaStream_t stream) {
  int threads = min(128, head_v_dim);
  size_t smem_size = 2 * head_k_dim * sizeof(float);
  gdn_prefill_transposed_kernel<<<num_v_heads, threads, smem_size, stream>>>(
      q, k, v, gate, beta, state_transposed, output,
      num_k_heads, num_v_heads, head_k_dim, head_v_dim, seq_len);
}

// ============================================================
// Fused FP16 QKV GEMV (block-dispatch: Q, K, V in single launch)
// ============================================================

template <int WARP_SIZE = 32, int WARPS_PER_BLOCK = 8>
__global__ void fused_fp16_qkv_gemv_kernel(
    const half* __restrict__ input,
    const half* __restrict__ q_weight, const half* __restrict__ k_weight,
    const half* __restrict__ v_weight,
    half* __restrict__ q_output, half* __restrict__ k_output,
    half* __restrict__ v_output,
    const int dim, const int q_dim, const int kv_dim)
{
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;

  const int q_blocks = (q_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  const int k_blocks = (kv_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

  const half* weight;
  half* output;
  int N, local_block;

  if (blockIdx.x < q_blocks) {
    local_block = blockIdx.x;
    weight = q_weight; output = q_output; N = q_dim;
  } else if (blockIdx.x < q_blocks + k_blocks) {
    local_block = blockIdx.x - q_blocks;
    weight = k_weight; output = k_output; N = kv_dim;
  } else {
    local_block = blockIdx.x - q_blocks - k_blocks;
    weight = v_weight; output = v_output; N = kv_dim;
  }

  const int row = local_block * WARPS_PER_BLOCK + warp_id;
  if (row >= N) return;

  const half* row_ptr = weight + static_cast<int64_t>(row) * dim;
  float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

  const int num_float4 = dim / 8;
  const float4* weight_f4 = reinterpret_cast<const float4*>(row_ptr);
  const float4* input_f4 = reinterpret_cast<const float4*>(input);

  #pragma unroll 4
  for (int i = lane_id; i < num_float4; i += WARP_SIZE) {
    float4 w = __ldg(weight_f4 + i);
    float4 x = __ldg(input_f4 + i);
    const half2* w_h2 = reinterpret_cast<const half2*>(&w);
    const half2* x_h2 = reinterpret_cast<const half2*>(&x);
    float2 wf0 = __half22float2(w_h2[0]); float2 xf0 = __half22float2(x_h2[0]);
    float2 wf1 = __half22float2(w_h2[1]); float2 xf1 = __half22float2(x_h2[1]);
    float2 wf2 = __half22float2(w_h2[2]); float2 xf2 = __half22float2(x_h2[2]);
    float2 wf3 = __half22float2(w_h2[3]); float2 xf3 = __half22float2(x_h2[3]);
    sum0 = fmaf(wf0.x, xf0.x, fmaf(wf0.y, xf0.y, sum0));
    sum1 = fmaf(wf1.x, xf1.x, fmaf(wf1.y, xf1.y, sum1));
    sum2 = fmaf(wf2.x, xf2.x, fmaf(wf2.y, xf2.y, sum2));
    sum3 = fmaf(wf3.x, xf3.x, fmaf(wf3.y, xf3.y, sum3));
  }
  float sum = sum0 + sum1 + sum2 + sum3;

  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if (lane_id == 0) output[row] = __float2half(sum);
}

void fused_fp16_qkv_gemv_cu(
    const half* input,
    const half* q_weight, const half* k_weight, const half* v_weight,
    half* q_output, half* k_output, half* v_output,
    int dim, int q_dim, int kv_dim, cudaStream_t stream)
{
  constexpr int WPB = 8;
  int total = (q_dim + WPB - 1) / WPB + 2 * ((kv_dim + WPB - 1) / WPB);
  fused_fp16_qkv_gemv_kernel<32, WPB><<<total, 256, 0, stream>>>(
      input, q_weight, k_weight, v_weight,
      q_output, k_output, v_output, dim, q_dim, kv_dim);
}

void fused_fp16_gdn_proj_gemv_cu(
    const half* input,
    const half* qkv_weight, const half* z_weight,
    half* qkv_output, half* z_output,
    int dim, int qkv_dim, int z_dim, cudaStream_t stream)
{
  constexpr int WPB = 8;
  // 2-way fusion: Q-blocks handle qkv, K-blocks handle z, no V-blocks
  int total = (qkv_dim + WPB - 1) / WPB + (z_dim + WPB - 1) / WPB;
  fused_fp16_qkv_gemv_kernel<32, WPB><<<total, 256, 0, stream>>>(
      input, qkv_weight, z_weight, z_weight,
      qkv_output, z_output, z_output,
      dim, qkv_dim, z_dim);
}

}  // namespace kernel
