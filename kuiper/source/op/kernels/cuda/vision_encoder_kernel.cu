/**
 * @file vision_encoder_kernel.cu
 * @brief CUDA kernel implementations for Qwen3-VL Vision Encoder
 */

#include "vision_encoder_kernel.cuh"
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cmath>
#include <cfloat>  // for FLT_MAX

namespace kernel {

// ============================================================================
// Helper Functions
// ============================================================================

__device__ __forceinline__ float gelu_approx(float x) {
  // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
  // Optimized with FMA instructions for Orin SM87
  const float sqrt_2_over_pi = 0.7978845608028654f;
  const float coeff = 0.044715f;
  float x_sq = x * x;
  float inner = sqrt_2_over_pi * __fmaf_rn(coeff, x_sq * x, x);
  return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ half gelu_approx_fp16(half x) {
  return __float2half(gelu_approx(__half2float(x)));
}

// ============================================================================
// LayerNorm with Bias Implementation
// ============================================================================

__global__ void layernorm_with_bias_fp16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int num_tokens,
    int hidden_size,
    float eps) {
  
  // Each block handles one token
  // Optimized: half2 vectorized loads + warp shuffle reduction (no dynamic shared memory)
  int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;
  
  const half* token_input = input + token_idx * hidden_size;
  half* token_output = output + token_idx * hidden_size;
  
  // Phase 1: Compute sum and sum_sq using half2 vectorized loads
  const int hidden_half2 = hidden_size >> 1;
  const half2* input_h2 = reinterpret_cast<const half2*>(token_input);
  
  float local_sum = 0.0f;
  float local_sum_sq = 0.0f;
  
  for (int i = threadIdx.x; i < hidden_half2; i += blockDim.x) {
    float2 fval = __half22float2(input_h2[i]);
    local_sum += fval.x + fval.y;
    local_sum_sq = __fmaf_rn(fval.x, fval.x, local_sum_sq);
    local_sum_sq = __fmaf_rn(fval.y, fval.y, local_sum_sq);
  }
  // Handle odd element
  if ((hidden_size & 1) && threadIdx.x == 0) {
    float val = __half2float(token_input[hidden_size - 1]);
    local_sum += val;
    local_sum_sq = __fmaf_rn(val, val, local_sum_sq);
  }
  
  // Warp-level reduction using shuffle
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    local_sum_sq += __shfl_xor_sync(0xffffffff, local_sum_sq, offset);
  }
  
  // Cross-warp reduction with minimal shared memory
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
  const int num_warps = blockDim.x >> 5;
  
  __shared__ float s_sum[32];
  __shared__ float s_sum_sq[32];
  
  if (lane_id == 0) {
    s_sum[warp_id] = local_sum;
    s_sum_sq[warp_id] = local_sum_sq;
  }
  __syncthreads();
  
  if (warp_id == 0) {
    local_sum = (lane_id < num_warps) ? s_sum[lane_id] : 0.0f;
    local_sum_sq = (lane_id < num_warps) ? s_sum_sq[lane_id] : 0.0f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
      local_sum_sq += __shfl_xor_sync(0xffffffff, local_sum_sq, offset);
    }
    if (lane_id == 0) {
      s_sum[0] = local_sum;
      s_sum_sq[0] = local_sum_sq;
    }
  }
  __syncthreads();
  
  // Compute mean and variance
  float mean = s_sum[0] / hidden_size;
  float variance = s_sum_sq[0] / hidden_size - mean * mean;
  float inv_std = rsqrtf(variance + eps);
  
  // Phase 2: Normalize with half2 vectorized access + FMA
  const half2* weight_h2 = reinterpret_cast<const half2*>(weight);
  const half2* bias_h2 = reinterpret_cast<const half2*>(bias);
  half2* output_h2 = reinterpret_cast<half2*>(token_output);
  
  for (int i = threadIdx.x; i < hidden_half2; i += blockDim.x) {
    float2 fval = __half22float2(input_h2[i]);
    float2 fw = __half22float2(weight_h2[i]);
    float2 fb = __half22float2(bias_h2[i]);
    
    float norm_x = (fval.x - mean) * inv_std;
    float norm_y = (fval.y - mean) * inv_std;
    
    float2 result;
    result.x = __fmaf_rn(norm_x, fw.x, fb.x);
    result.y = __fmaf_rn(norm_y, fw.y, fb.y);
    output_h2[i] = __floats2half2_rn(result.x, result.y);
  }
  // Handle odd element
  if ((hidden_size & 1) && threadIdx.x == 0) {
    float val = __half2float(token_input[hidden_size - 1]);
    float normalized = (val - mean) * inv_std;
    float w = __half2float(weight[hidden_size - 1]);
    float b = __half2float(bias[hidden_size - 1]);
    token_output[hidden_size - 1] = __float2half(__fmaf_rn(normalized, w, b));
  }
}

void layernorm_with_bias_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& weight,
    const tensor::Tensor& bias,
    tensor::Tensor& output,
    float eps,
    cudaStream_t stream) {
  
  int num_tokens = input.get_dim(0);
  int hidden_size = input.get_dim(1);
  
  int block_size = 256;
  int grid_size = num_tokens;
  
  layernorm_with_bias_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
      input.ptr<half>(),
      weight.ptr<half>(),
      bias.ptr<half>(),
      output.ptr<half>(),
      num_tokens,
      hidden_size,
      eps);
}

// ============================================================================
// Fused Bias + GELU kernel for MLP optimization
// ============================================================================

__global__ void bias_gelu_fp16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int size,
    int bias_size) {
  
  // Optimized: process 8 elements per thread using float4 (128-bit) loads/stores
  const int VEC = 8;
  int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
  
  if (base_idx + VEC <= size) {
    // Vectorized path: load 8 halfs (16 bytes) at once
    float4 in_data = *reinterpret_cast<const float4*>(&input[base_idx]);
    const half* in_h = reinterpret_cast<const half*>(&in_data);
    
    int bias_idx = base_idx % bias_size;
    float4 b_data = *reinterpret_cast<const float4*>(&bias[bias_idx]);
    const half* b_h = reinterpret_cast<const half*>(&b_data);
    
    half result[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      float val = __half2float(in_h[i]) + __half2float(b_h[i]);
      result[i] = __float2half(gelu_approx(val));
    }
    
    *reinterpret_cast<float4*>(&output[base_idx]) = *reinterpret_cast<const float4*>(result);
  } else {
    // Scalar fallback for remainder
    for (int i = base_idx; i < size; i++) {
      int bias_idx = i % bias_size;
      float val = __half2float(input[i]) + __half2float(bias[bias_idx]);
      output[i] = __float2half(gelu_approx(val));
    }
  }
}

void bias_gelu_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& bias,
    tensor::Tensor& output,
    cudaStream_t stream) {
  
  int size = static_cast<int>(input.size());
  int bias_size = static_cast<int>(bias.size());
  int block_size = 256;
  int num_vecs = (size + 7) / 8;
  int grid_size = (num_vecs + block_size - 1) / block_size;
  
  bias_gelu_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
      input.ptr<half>(),
      bias.ptr<half>(),
      output.ptr<half>(),
      size,
      bias_size);
}

// ============================================================================
// GELU Activation Implementation
// ============================================================================

__global__ void gelu_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int size) {
  
  // Optimized: process 8 elements per thread using float4 loads/stores
  const int VEC = 8;
  int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
  
  if (base_idx + VEC <= size) {
    float4 in_data = *reinterpret_cast<const float4*>(&input[base_idx]);
    const half* in_h = reinterpret_cast<const half*>(&in_data);
    
    half result[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      result[i] = __float2half(gelu_approx(__half2float(in_h[i])));
    }
    
    *reinterpret_cast<float4*>(&output[base_idx]) = *reinterpret_cast<const float4*>(result);
  } else {
    for (int i = base_idx; i < size; i++) {
      output[i] = gelu_approx_fp16(input[i]);
    }
  }
}

void gelu_cu(
    const tensor::Tensor& input,
    tensor::Tensor& output,
    cudaStream_t stream) {
  
  int size = static_cast<int>(input.size());
  int block_size = 256;
  int num_vecs = (size + 7) / 8;
  int grid_size = (num_vecs + block_size - 1) / block_size;
  
  gelu_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
      input.ptr<half>(),
      output.ptr<half>(),
      size);
}

// ============================================================================
// Bias Add with Residual Connection
// ============================================================================

__global__ void bias_add_residual_fp16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ bias,
    const half* __restrict__ residual,
    half* __restrict__ output,
    int size,
    int bias_size) {
  
  // Optimized: process 8 elements per thread using float4 (128-bit) loads/stores
  const int VEC = 8;
  int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
  
  if (base_idx + VEC <= size) {
    float4 in_data = *reinterpret_cast<const float4*>(&input[base_idx]);
    const half* in_h = reinterpret_cast<const half*>(&in_data);
    
    int bias_idx = base_idx % bias_size;
    float4 b_data = *reinterpret_cast<const float4*>(&bias[bias_idx]);
    const half* b_h = reinterpret_cast<const half*>(&b_data);
    
    half result[8];
    
    if (residual != nullptr) {
      float4 r_data = *reinterpret_cast<const float4*>(&residual[base_idx]);
      const half* r_h = reinterpret_cast<const half*>(&r_data);
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        result[i] = __float2half(__half2float(in_h[i]) + __half2float(b_h[i]) + __half2float(r_h[i]));
      }
    } else {
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        result[i] = __float2half(__half2float(in_h[i]) + __half2float(b_h[i]));
      }
    }
    
    *reinterpret_cast<float4*>(&output[base_idx]) = *reinterpret_cast<const float4*>(result);
  } else {
    // Scalar fallback for remainder
    for (int i = base_idx; i < size; i++) {
      int bias_idx = i % bias_size;
      float val = __half2float(input[i]) + __half2float(bias[bias_idx]);
      if (residual != nullptr) val += __half2float(residual[i]);
      output[i] = __float2half(val);
    }
  }
}

void bias_add_residual_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& bias,
    const tensor::Tensor& residual,
    tensor::Tensor& output,
    cudaStream_t stream) {
  
  int size = static_cast<int>(input.size());
  int bias_size = static_cast<int>(bias.size());
  int block_size = 256;
  int num_vecs = (size + 7) / 8;
  int grid_size = (num_vecs + block_size - 1) / block_size;
  
  bias_add_residual_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
      input.ptr<half>(),
      bias.ptr<half>(),
      residual.is_empty() ? nullptr : residual.ptr<half>(),
      output.ptr<half>(),
      size,
      bias_size);
}

// ============================================================================
// Position Embedding with Bilinear Interpolation
// ============================================================================

__global__ void pos_embed_interpolate_fp16_kernel(
    const half* __restrict__ patch_embeds,
    const half* __restrict__ pos_embed,
    half* __restrict__ output,
    int num_patches,
    int hidden_size,
    int grid_h,
    int grid_w,
    int grid_t,
    int num_grid_per_side,
    int spatial_merge_size) {
  
  int patch_idx = blockIdx.x;
  int hidden_idx = threadIdx.x + blockIdx.y * blockDim.x;
  
  if (patch_idx >= num_patches || hidden_idx >= hidden_size) return;
  
  // Calculate grid position for this patch
  // Input patch_embeds is in 2x2 block-interleaved order (after patch_embed + pixel reordering)
  // We need to compute the position embedding for this patch
  
  int patches_per_frame = grid_h * grid_w;
  int frame_idx = patch_idx / patches_per_frame;
  int in_frame_idx = patch_idx % patches_per_frame;
  
  // The patch_embeds is in 2x2 block-interleaved order:
  // For patch_idx in the input, find its (h_pos, w_pos) in the original grid
  int w_div = grid_w / spatial_merge_size;  // number of 2x2 blocks in width
  
  int block_idx = in_frame_idx / (spatial_merge_size * spatial_merge_size);
  int in_block_idx = in_frame_idx % (spatial_merge_size * spatial_merge_size);
  
  int block_h = block_idx / w_div;
  int block_w = block_idx % w_div;
  int local_h = in_block_idx / spatial_merge_size;
  int local_w = in_block_idx % spatial_merge_size;
  
  // Original grid position for this patch
  int h_pos = block_h * spatial_merge_size + local_h;
  int w_pos = block_w * spatial_merge_size + local_w;
  
  // Bilinear interpolation from base grid (48x48)
  // HuggingFace uses torch.linspace(0, num_grid_per_side-1, grid_h)
  // This produces indices: 0, step, 2*step, ..., (grid_h-1)*step = num_grid_per_side-1
  // where step = (num_grid_per_side - 1) / (grid_h - 1)
  float h_idx = (grid_h > 1) ? (static_cast<float>(h_pos) * (num_grid_per_side - 1) / (grid_h - 1)) : 0.0f;
  float w_idx = (grid_w > 1) ? (static_cast<float>(w_pos) * (num_grid_per_side - 1) / (grid_w - 1)) : 0.0f;
  
  int h_floor = static_cast<int>(h_idx);
  int w_floor = static_cast<int>(w_idx);
  int h_ceil = min(h_floor + 1, num_grid_per_side - 1);
  int w_ceil = min(w_floor + 1, num_grid_per_side - 1);
  
  float dh = h_idx - h_floor;
  float dw = w_idx - w_floor;
  
  // Bilinear weights
  float w00 = (1 - dh) * (1 - dw);
  float w01 = (1 - dh) * dw;
  float w10 = dh * (1 - dw);
  float w11 = dh * dw;
  
  // Fetch position embeddings from base grid (row-major order)
  int idx00 = (h_floor * num_grid_per_side + w_floor) * hidden_size + hidden_idx;
  int idx01 = (h_floor * num_grid_per_side + w_ceil) * hidden_size + hidden_idx;
  int idx10 = (h_ceil * num_grid_per_side + w_floor) * hidden_size + hidden_idx;
  int idx11 = (h_ceil * num_grid_per_side + w_ceil) * hidden_size + hidden_idx;
  
  // Bilinear interpolation with FMA for pos_embed
  float pos_val = __fmaf_rn(w00, __half2float(pos_embed[idx00]),
                  __fmaf_rn(w01, __half2float(pos_embed[idx01]),
                  __fmaf_rn(w10, __half2float(pos_embed[idx10]),
                            w11 * __half2float(pos_embed[idx11]))));
  
  // Add to patch embedding
  int out_idx = patch_idx * hidden_size + hidden_idx;
  float patch_val = __half2float(patch_embeds[out_idx]);
  output[out_idx] = __float2half(patch_val + pos_val);
}

void pos_embed_interpolate_cu(
    const tensor::Tensor& patch_embeds,
    const tensor::Tensor& pos_embed,
    tensor::Tensor& output,
    int grid_h,
    int grid_w,
    int grid_t,
    int num_grid_per_side,
    int spatial_merge_size,
    cudaStream_t stream) {
  
  int num_patches = patch_embeds.get_dim(0);
  int hidden_size = patch_embeds.get_dim(1);
  
  dim3 block_size(256);
  dim3 grid_size(num_patches, (hidden_size + 255) / 256);
  
  pos_embed_interpolate_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
      patch_embeds.ptr<half>(),
      pos_embed.ptr<half>(),
      output.ptr<half>(),
      num_patches,
      hidden_size,
      grid_h,
      grid_w,
      grid_t,
      num_grid_per_side,
      spatial_merge_size);
}

// ============================================================================
// Spatial Merge Implementation
// ============================================================================

// Spatial merge for 2x2 block-interleaved input
// Input is already in block order: patches 0,1,2,3 form block 0, patches 4,5,6,7 form block 1, etc.
// HuggingFace does: x.view(-1, hidden_size * 4) which simply concatenates every 4 consecutive patches
// So we just need to reshape: [num_patches, hidden] -> [num_patches/4, hidden*4]
__global__ void spatial_merge_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int num_patches,
    int hidden_size,
    int merge_area) {
  
  // Optimized: float4 vectorized copy (8 halfs per thread)
  int num_out_tokens = num_patches / merge_area;
  int out_hidden_size = hidden_size * merge_area;
  
  int token_idx = blockIdx.x;
  int base_idx = (threadIdx.x + blockIdx.y * blockDim.x) * 8;
  
  if (token_idx >= num_out_tokens || base_idx >= out_hidden_size) return;
  
  // For hidden_size divisible by 8, float4 never crosses patch boundaries
  int local_patch = base_idx / hidden_size;
  int local_hidden = base_idx % hidden_size;
  
  int in_patch_idx = token_idx * merge_area + local_patch;
  int in_idx = in_patch_idx * hidden_size + local_hidden;
  int out_idx = token_idx * out_hidden_size + base_idx;
  
  if (base_idx + 8 <= out_hidden_size && local_hidden + 8 <= hidden_size) {
    float4 data = *reinterpret_cast<const float4*>(&input[in_idx]);
    *reinterpret_cast<float4*>(&output[out_idx]) = data;
  } else {
    // Scalar fallback
    for (int i = 0; i < 8 && base_idx + i < out_hidden_size; i++) {
      int h_idx = base_idx + i;
      int lp = h_idx / hidden_size;
      int lh = h_idx % hidden_size;
      int ip = token_idx * merge_area + lp;
      output[token_idx * out_hidden_size + h_idx] = input[ip * hidden_size + lh];
    }
  }
}

void spatial_merge_cu(
    const tensor::Tensor& input,
    tensor::Tensor& output,
    int grid_t,
    int grid_h,
    int grid_w,
    int hidden_size,
    int spatial_merge_size,
    cudaStream_t stream) {
  
  int merge_area = spatial_merge_size * spatial_merge_size;  // 4
  int num_patches = input.get_dim(0);  // grid_t * grid_h * grid_w
  int num_out_tokens = num_patches / merge_area;
  int out_hidden_size = hidden_size * merge_area;
  
  dim3 block_size(256);
  int num_vecs = (out_hidden_size + 7) / 8;
  dim3 grid_size(num_out_tokens, (num_vecs + 255) / 256);
  
  spatial_merge_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
      input.ptr<half>(),
      output.ptr<half>(),
      num_patches,
      hidden_size,
      merge_area);
}

// ============================================================================
// Vision MLP Implementation
// ============================================================================
void vision_mlp_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& fc1_weight,
    const tensor::Tensor& fc1_bias,
    const tensor::Tensor& fc2_weight,
    const tensor::Tensor& fc2_bias,
    const tensor::Tensor& residual,
    tensor::Tensor& output,
    tensor::Tensor& intermediate,
    const kernel::CudaConfig* config) {
  
  // Optimized Vision MLP with fused operations:
  // 1. fc1 GEMM
  // 2. fused bias + GELU (saves one memory pass)
  // 3. fc2 GEMM
  // 4. bias + residual
  
  int num_tokens = input.get_dim(0);
  int hidden_size = input.get_dim(1);
  int intermediate_size = fc1_weight.get_dim(0);
  
  const half alpha = __float2half(1.0f);
  const half beta = __float2half(0.0f);
  
  // Step 1: fc1 GEMM
  cublasHgemm(config->cublas_handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              intermediate_size, num_tokens, hidden_size,
              &alpha,
              fc1_weight.ptr<half>(), hidden_size,
              input.ptr<half>(), hidden_size,
              &beta,
              intermediate.ptr<half>(), intermediate_size);
  
  // Step 2: Fused bias + GELU (replaces separate bias_add + gelu)
  bias_gelu_cu(intermediate, fc1_bias, intermediate, config->stream);
  
  // Step 3: fc2 GEMM
  // Reuse input buffer as fc2_out temp buffer
  tensor::Tensor& fc2_out = const_cast<tensor::Tensor&>(input);
  
  cublasHgemm(config->cublas_handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              hidden_size, num_tokens, intermediate_size,
              &alpha,
              fc2_weight.ptr<half>(), intermediate_size,
              intermediate.ptr<half>(), intermediate_size,
              &beta,
              fc2_out.ptr<half>(), hidden_size);
  
  // Step 4: bias + residual
  bias_add_residual_cu(fc2_out, fc2_bias, residual, output, config->stream);
}

// ============================================================================
// Vision Merger Implementation
// ============================================================================

// Vision merger MLP only (LayerNorm is done separately before spatial merge)
void vision_merger_mlp_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& fc1_weight,
    const tensor::Tensor& fc1_bias,
    const tensor::Tensor& fc2_weight,
    const tensor::Tensor& fc2_bias,
    tensor::Tensor& output,
    tensor::Tensor& intermediate,
    const kernel::CudaConfig* config) {
  
  int num_tokens = input.get_dim(0);
  int merged_hidden = input.get_dim(1);  // 4608
  int out_hidden = fc2_weight.get_dim(0); // 4096
  
  const half alpha = __float2half(1.0f);
  const half beta = __float2half(0.0f);
  
  // fc1: [num_tokens, merged_hidden] -> [num_tokens, merged_hidden]
  cublasHgemm(config->cublas_handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              merged_hidden, num_tokens, merged_hidden,
              &alpha,
              fc1_weight.ptr<half>(), merged_hidden,
              input.ptr<half>(), merged_hidden,
              &beta,
              intermediate.ptr<half>(), merged_hidden);
  
  bias_add_residual_cu(intermediate, fc1_bias, tensor::Tensor(), intermediate, config->stream);
  gelu_cu(intermediate, intermediate, config->stream);
  
  // fc2: [num_tokens, merged_hidden] -> [num_tokens, out_hidden]
  cublasHgemm(config->cublas_handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              out_hidden, num_tokens, merged_hidden,
              &alpha,
              fc2_weight.ptr<half>(), merged_hidden,
              intermediate.ptr<half>(), merged_hidden,
              &beta,
              output.ptr<half>(), out_hidden);
  
  bias_add_residual_cu(output, fc2_bias, tensor::Tensor(), output, config->stream);
}

// ============================================================================
// Fused Split + RoPE + Transpose Kernel
// ============================================================================

/**
 * Super-fused kernel: split_qkv + RoPE + transpose in ONE kernel
 * 
 * Input:  qkv [num_tokens, 3 * hidden_size]  (hidden_size = num_heads * head_dim)
 * Output: q_trans [num_heads, num_tokens, head_dim]  (with RoPE applied)
 *         k_trans [num_heads, num_tokens, head_dim]  (with RoPE applied)
 *         v_trans [num_heads, num_tokens, head_dim]  (no RoPE)
 * 
 * This fuses 3 operations into 1:
 * 1. Split QKV (1 global read)
 * 2. Apply RoPE to Q, K
 * 3. Transpose to [num_heads, seq_len, head_dim] layout for batched GEMM
 * 
 * Saves 5 kernel launches and multiple global memory passes.
 */
__global__ void fused_split_rope_transpose_kernel(
    const half* __restrict__ qkv,           // [num_tokens, 3 * hidden_size]
    const half* __restrict__ cos_cache,     // [num_tokens, head_dim]
    const half* __restrict__ sin_cache,     // [num_tokens, head_dim]
    half* __restrict__ q_trans,             // [num_heads, num_tokens, head_dim]
    half* __restrict__ k_trans,             // [num_heads, num_tokens, head_dim]
    half* __restrict__ v_trans,             // [num_heads, num_tokens, head_dim]
    int num_tokens,
    int num_heads,
    int head_dim) {
  
  // Grid: [num_heads, num_tokens]
  // Block: [head_dim] or (head_dim + blockDim - 1) / blockDim threads
  const int head_idx = blockIdx.x;
  const int token_idx = blockIdx.y;
  const int hidden_size = num_heads * head_dim;
  const int half_head_dim = head_dim / 2;
  
  if (head_idx >= num_heads || token_idx >= num_tokens) return;
  
  // Base pointers for this token in qkv
  const half* qkv_token = qkv + token_idx * 3 * hidden_size;
  const half* q_in = qkv_token + head_idx * head_dim;
  const half* k_in = qkv_token + hidden_size + head_idx * head_dim;
  const half* v_in = qkv_token + 2 * hidden_size + head_idx * head_dim;
  
  // RoPE cache for this token
  const half* cos_ptr = cos_cache + token_idx * head_dim;
  const half* sin_ptr = sin_cache + token_idx * head_dim;
  
  // Output base pointers (transposed layout)
  // Output layout: [head_idx, token_idx, dim] = [head_idx * num_tokens * head_dim + token_idx * head_dim + dim]
  int out_offset = head_idx * num_tokens * head_dim + token_idx * head_dim;
  half* q_out = q_trans + out_offset;
  half* k_out = k_trans + out_offset;
  half* v_out = v_trans + out_offset;
  
  // Process dimensions in parallel
  for (int d = threadIdx.x; d < half_head_dim; d += blockDim.x) {
    // Load Q, K pairs from input
    float q1 = __half2float(q_in[d]);
    float q2 = __half2float(q_in[d + half_head_dim]);
    float k1 = __half2float(k_in[d]);
    float k2 = __half2float(k_in[d + half_head_dim]);
    
    // Load rotation angles
    float cos_val = __half2float(cos_ptr[d]);
    float sin_val = __half2float(sin_ptr[d]);
    float cos_val2 = __half2float(cos_ptr[d + half_head_dim]);
    float sin_val2 = __half2float(sin_ptr[d + half_head_dim]);
    
    // Apply RoPE rotation with FMA instructions
    // rotate_half: [x1, x2] -> [-x2, x1]
    float q1_rot = __fmaf_rn(q1, cos_val, -(q2 * sin_val));
    float q2_rot = __fmaf_rn(q1, sin_val2, q2 * cos_val2);
    float k1_rot = __fmaf_rn(k1, cos_val, -(k2 * sin_val));
    float k2_rot = __fmaf_rn(k1, sin_val2, k2 * cos_val2);
    
    // Write to transposed output
    q_out[d] = __float2half(q1_rot);
    q_out[d + half_head_dim] = __float2half(q2_rot);
    k_out[d] = __float2half(k1_rot);
    k_out[d + half_head_dim] = __float2half(k2_rot);
  }
  
  // V doesn't need RoPE, just copy with transpose using float4 vectorization
  const int head_dim_f4 = head_dim / 8;  // 72/8 = 9
  if (head_dim_f4 > 0) {
    const float4* v_in_f4 = reinterpret_cast<const float4*>(v_in);
    float4* v_out_f4 = reinterpret_cast<float4*>(v_out);
    for (int d = threadIdx.x; d < head_dim_f4; d += blockDim.x) {
      v_out_f4[d] = v_in_f4[d];
    }
    // Handle remainder (head_dim=72 is divisible by 8, so no remainder)
    for (int d = head_dim_f4 * 8 + threadIdx.x; d < head_dim; d += blockDim.x) {
      v_out[d] = v_in[d];
    }
  } else {
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      v_out[d] = v_in[d];
    }
  }
}

void fused_split_rope_transpose_cu(
    const tensor::Tensor& qkv,
    const tensor::Tensor& cos_cache,
    const tensor::Tensor& sin_cache,
    tensor::Tensor& q_trans,
    tensor::Tensor& k_trans,
    tensor::Tensor& v_trans,
    int num_tokens,
    int num_heads,
    int head_dim,
    cudaStream_t stream) {
  
  // Launch with grid [num_heads, num_tokens], block [128]
  // Each block handles one (head, token) pair with improved occupancy
  dim3 grid(num_heads, num_tokens);
  dim3 block(128);  // Increased from 64 for better SM utilization
  
  fused_split_rope_transpose_kernel<<<grid, block, 0, stream>>>(
      qkv.ptr<half>(),
      cos_cache.ptr<half>(),
      sin_cache.ptr<half>(),
      q_trans.ptr<half>(),
      k_trans.ptr<half>(),
      v_trans.ptr<half>(),
      num_tokens,
      num_heads,
      head_dim);
}

// ============================================================================
// Vision Self-Attention Implementation
// ============================================================================


/**
 * Transpose kernel: [num_heads, total_tokens, head_dim] -> [total_tokens, num_heads * head_dim]
 * Optimized with float4 vectorization (8 halfs per thread) for aligned head_dim
 */
__global__ void transpose_head_token_kernel(
    const half* __restrict__ input,   // [num_heads, total_tokens, head_dim]
    half* __restrict__ output,        // [total_tokens, num_heads * head_dim]
    int total_tokens,
    int num_heads,
    int head_dim) {
  
  const int vec_size = ((head_dim & 7) == 0) ? 8 : 2;
  const int head_dim_vec = head_dim / vec_size;
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_vecs = total_tokens * num_heads * head_dim_vec;
  
  if (idx >= total_vecs) return;
  
  int d_vec = idx % head_dim_vec;
  int temp = idx / head_dim_vec;
  int t = temp % total_tokens;
  int h = temp / total_tokens;
  
  int input_off = h * total_tokens * head_dim + t * head_dim + d_vec * vec_size;
  int output_off = t * (num_heads * head_dim) + h * head_dim + d_vec * vec_size;
  
  if (vec_size == 8) {
    *reinterpret_cast<float4*>(&output[output_off]) = 
        *reinterpret_cast<const float4*>(&input[input_off]);
  } else {
    *reinterpret_cast<half2*>(&output[output_off]) = 
        *reinterpret_cast<const half2*>(&input[input_off]);
  }
}




/**
 * Simple softmax kernel for attention scores
 */

/**
 * Optimized vision softmax kernel with vectorized access and better parallelism
 * Each block processes one row of one head's attention matrix
 * Uses half2 vectorization for memory access and warp-level primitives
 */
__global__ void vision_softmax_fp16_kernel(
    half* __restrict__ scores,   // [num_heads, num_tokens, num_tokens]
    int num_tokens,
    int num_heads,
    float scale) {
  
  // blockIdx.x = [0, num_heads * num_tokens)
  // Each block processes one row of one head's attention matrix
  int head_idx = blockIdx.x / num_tokens;
  int row = blockIdx.x % num_tokens;
  
  if (head_idx >= num_heads || row >= num_tokens) return;
  
  // Calculate correct offset into the [num_heads, num_tokens, num_tokens] tensor
  long long head_offset = static_cast<long long>(head_idx) * num_tokens * num_tokens;
  half* row_scores = scores + head_offset + static_cast<long long>(row) * num_tokens;
  
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;
  
  // Use vectorized access where possible
  const int num_tokens_h2 = num_tokens >> 1;
  half2* row_scores_h2 = reinterpret_cast<half2*>(row_scores);
  
  // Find max for numerical stability using vectorized loads
  float max_val = -INFINITY;
  for (int i = tid; i < num_tokens_h2; i += blockDim.x) {
    float2 vals = __half22float2(row_scores_h2[i]);
    vals.x *= scale;
    vals.y *= scale;
    max_val = fmaxf(max_val, fmaxf(vals.x, vals.y));
  }
  // Handle odd element if num_tokens is odd
  if ((num_tokens & 1) && (tid == 0)) {
    int last_idx = num_tokens - 1;
    float val = __half2float(row_scores[last_idx]) * scale;
    max_val = fmaxf(max_val, val);
  }
  
  // Warp reduce max
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
  }
  
  // Block reduce max using shared memory
  __shared__ float shared_max[32];
  if (lane == 0 && warp_id < 32) shared_max[warp_id] = max_val;
  __syncthreads();
  
  // Thread 0 performs final reduction
  if (tid == 0) {
    float final_max = shared_max[0];
    for (int i = 1; i < num_warps && i < 32; i++) {
      final_max = fmaxf(final_max, shared_max[i]);
    }
    shared_max[0] = final_max;
  }
  __syncthreads();
  max_val = shared_max[0];
  
  // Compute exp and sum using vectorized access
  float sum = 0.0f;
  for (int i = tid; i < num_tokens_h2; i += blockDim.x) {
    float2 vals = __half22float2(row_scores_h2[i]);
    vals.x = expf(vals.x * scale - max_val);
    vals.y = expf(vals.y * scale - max_val);
    row_scores_h2[i] = __floats2half2_rn(vals.x, vals.y);
    sum += vals.x + vals.y;
  }
  // Handle odd element
  if ((num_tokens & 1) && (tid == 0)) {
    int last_idx = num_tokens - 1;
    float val = expf(__half2float(row_scores[last_idx]) * scale - max_val);
    row_scores[last_idx] = __float2half(val);
    sum += val;
  }
  
  // Warp reduce sum
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  }
  
  // Block reduce sum
  __shared__ float shared_sum[32];
  if (lane == 0 && warp_id < 32) shared_sum[warp_id] = sum;
  __syncthreads();
  
  // Thread 0 performs final reduction
  if (tid == 0) {
    float final_sum = shared_sum[0];
    for (int i = 1; i < num_warps && i < 32; i++) {
      final_sum += shared_sum[i];
    }
    shared_sum[0] = final_sum;
  }
  __syncthreads();
  sum = shared_sum[0];
  
  // Normalize using vectorized access
  float inv_sum = 1.0f / (sum + 1e-10f);
  for (int i = tid; i < num_tokens_h2; i += blockDim.x) {
    float2 vals = __half22float2(row_scores_h2[i]);
    vals.x *= inv_sum;
    vals.y *= inv_sum;
    row_scores_h2[i] = __floats2half2_rn(vals.x, vals.y);
  }
  // Handle odd element
  if ((num_tokens & 1) && (tid == 0)) {
    int last_idx = num_tokens - 1;
    row_scores[last_idx] = __float2half(__half2float(row_scores[last_idx]) * inv_sum);
  }
}

/**
 * @brief Optimized vision attention with pre-transposed Q, K, V
 * 
 * This version skips the input transpose step because the caller
 * uses fused_split_rope_transpose_cu to produce already-transposed tensors.
 * 
 * Input: Q, K, V already in [num_heads, num_tokens, head_dim] layout
 * Output: attention output in [num_tokens, hidden_size] layout
 */
void vision_attention_pretransposed_cu(
    const tensor::Tensor& q_trans,      // [num_heads, num_tokens, head_dim]
    const tensor::Tensor& k_trans,      // [num_heads, num_tokens, head_dim]
    const tensor::Tensor& v_trans,      // [num_heads, num_tokens, head_dim]
    tensor::Tensor& output,             // [num_tokens, hidden_size]
    tensor::Tensor& out_transposed,     // workspace [num_heads, num_tokens, head_dim]
    tensor::Tensor& scores,             // workspace [num_heads, num_tokens, num_tokens]
    int num_tokens,
    int num_heads,
    int head_dim,
    float softmax_scale,
    const kernel::CudaConfig* config) {
  
  cudaStream_t stream = config->stream;
  cublasHandle_t handle = config->cublas_handle;
  
  const half alpha_h = __float2half(1.0f);
  const half beta_h = __float2half(0.0f);
  
  long long stride_q = num_tokens * head_dim;
  long long stride_k = num_tokens * head_dim;
  long long stride_s = static_cast<long long>(num_tokens) * num_tokens;
  
  // Batched GEMM for scores: Q @ K^T for all heads
  cublasHgemmStridedBatched(
      handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      num_tokens, num_tokens, head_dim,
      &alpha_h,
      k_trans.ptr<half>(), head_dim, stride_k,
      q_trans.ptr<half>(), head_dim, stride_q,
      &beta_h,
      scores.ptr<half>(), num_tokens, stride_s,
      num_heads);
  
  // Apply softmax with scaling
  dim3 softmax_grid(num_tokens * num_heads);
  dim3 softmax_block(256);
  vision_softmax_fp16_kernel<<<softmax_grid, softmax_block, 0, stream>>>(
      scores.ptr<half>(),
      num_tokens,
      num_heads,
      softmax_scale);
  
  // Batched GEMM for output: scores @ V
  long long stride_v = num_tokens * head_dim;
  long long stride_o = num_tokens * head_dim;
  
  cublasHgemmStridedBatched(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      head_dim, num_tokens, num_tokens,
      &alpha_h,
      v_trans.ptr<half>(), head_dim, stride_v,
      scores.ptr<half>(), num_tokens, stride_s,
      &beta_h,
      out_transposed.ptr<half>(), head_dim, stride_o,
      num_heads);
  
  // Transpose output back: [num_heads, num_tokens, head_dim] -> [num_tokens, num_heads * head_dim]
  int vec_div = ((head_dim & 7) == 0) ? 8 : 2;
  int total_vecs = num_tokens * num_heads * (head_dim / vec_div);
  dim3 trans_block(256);
  dim3 trans_grid((total_vecs + 255) / 256);
  
  transpose_head_token_kernel<<<trans_grid, trans_block, 0, stream>>>(
      out_transposed.ptr<half>(), output.ptr<half>(), num_tokens, num_heads, head_dim);
}

}  // namespace kernel
