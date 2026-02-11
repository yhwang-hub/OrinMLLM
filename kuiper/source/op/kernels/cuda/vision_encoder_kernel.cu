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
  const float sqrt_2_over_pi = 0.7978845608028654f;
  const float coeff = 0.044715f;
  float x3 = x * x * x;
  float inner = sqrt_2_over_pi * (x + coeff * x3);
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
  int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;
  
  const half* token_input = input + token_idx * hidden_size;
  half* token_output = output + token_idx * hidden_size;
  
  // Shared memory for reduction
  extern __shared__ float shared[];
  float* s_sum = shared;
  float* s_sum_sq = shared + blockDim.x;
  
  // Compute local sum and sum of squares
  float local_sum = 0.0f;
  float local_sum_sq = 0.0f;
  
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float val = __half2float(token_input[i]);
    local_sum += val;
    local_sum_sq += val * val;
  }
  
  // Store in shared memory
  s_sum[threadIdx.x] = local_sum;
  s_sum_sq[threadIdx.x] = local_sum_sq;
  __syncthreads();
  
  // Block reduction
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
      s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }
  
  // Compute mean and variance
  float mean = s_sum[0] / hidden_size;
  float variance = s_sum_sq[0] / hidden_size - mean * mean;
  float inv_std = rsqrtf(variance + eps);
  
  // Normalize and apply weight/bias
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float val = __half2float(token_input[i]);
    float normalized = (val - mean) * inv_std;
    float w = __half2float(weight[i]);
    float b = __half2float(bias[i]);
    token_output[i] = __float2half(normalized * w + b);
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
  size_t shared_mem_size = 2 * block_size * sizeof(float);
  
  layernorm_with_bias_fp16_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
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
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Process 2 elements at a time using half2
  int idx2 = idx * 2;
  if (idx2 + 1 < size) {
    int bias_idx = idx2 % bias_size;
    half2 in = *reinterpret_cast<const half2*>(&input[idx2]);
    half2 b = *reinterpret_cast<const half2*>(&bias[bias_idx]);
    
    // Add bias
    float val0 = __half2float(in.x) + __half2float(b.x);
    float val1 = __half2float(in.y) + __half2float(b.y);
    
    // Apply GELU
    half2 out;
    out.x = __float2half(gelu_approx(val0));
    out.y = __float2half(gelu_approx(val1));
    *reinterpret_cast<half2*>(&output[idx2]) = out;
  } else if (idx2 < size) {
    int bias_idx = idx2 % bias_size;
    float val = __half2float(input[idx2]) + __half2float(bias[bias_idx]);
    output[idx2] = __float2half(gelu_approx(val));
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
  int grid_size = (size / 2 + block_size - 1) / block_size;
  
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
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Process 2 elements at a time using half2
  int idx2 = idx * 2;
  if (idx2 + 1 < size) {
    half2 in = *reinterpret_cast<const half2*>(&input[idx2]);
    half2 out;
    out.x = gelu_approx_fp16(in.x);
    out.y = gelu_approx_fp16(in.y);
    *reinterpret_cast<half2*>(&output[idx2]) = out;
  } else if (idx2 < size) {
    output[idx2] = gelu_approx_fp16(input[idx2]);
  }
}

void gelu_cu(
    const tensor::Tensor& input,
    tensor::Tensor& output,
    cudaStream_t stream) {
  
  int size = static_cast<int>(input.size());
  int block_size = 256;
  int grid_size = (size / 2 + block_size - 1) / block_size;
  
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
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;
  
  int bias_idx = idx % bias_size;
  float val = __half2float(input[idx]) + __half2float(bias[bias_idx]);
  if (residual != nullptr) {
    val += __half2float(residual[idx]);
  }
  output[idx] = __float2half(val);
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
  int grid_size = (size + block_size - 1) / block_size;
  
  bias_add_residual_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
      input.ptr<half>(),
      bias.ptr<half>(),
      residual.is_empty() ? nullptr : residual.ptr<half>(),
      output.ptr<half>(),
      size,
      bias_size);
}

// ============================================================================
// Patch Embedding (Conv3D) Implementation
// ============================================================================

__global__ void patch_embed_conv3d_fp16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int num_patches,
    int hidden_size,
    int patch_dim,
    int in_channels,
    int temporal_patch_size,
    int patch_size) {
  
  // Each thread computes one output element
  int patch_idx = blockIdx.x;
  int out_idx = threadIdx.x + blockIdx.y * blockDim.x;
  
  if (patch_idx >= num_patches || out_idx >= hidden_size) return;
  
  const half* patch_input = input + patch_idx * patch_dim;
  const half* filter = weight + out_idx * patch_dim;  // [hidden_size, patch_dim]
  
  // Compute dot product
  float sum = 0.0f;
  for (int i = 0; i < patch_dim; ++i) {
    sum += __half2float(patch_input[i]) * __half2float(filter[i]);
  }
  
  // Add bias
  sum += __half2float(bias[out_idx]);
  
  // Write output
  output[patch_idx * hidden_size + out_idx] = __float2half(sum);
}

void patch_embed_conv3d_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& weight,
    const tensor::Tensor& bias,
    tensor::Tensor& output,
    int in_channels,
    int temporal_patch_size,
    int patch_size,
    cudaStream_t stream) {
  
  int num_patches = input.get_dim(0);
  int patch_dim = input.get_dim(1);
  int hidden_size = weight.get_dim(0);
  
  dim3 block_size(256);
  dim3 grid_size(num_patches, (hidden_size + 255) / 256);
  
  patch_embed_conv3d_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
      input.ptr<half>(),
      weight.ptr<half>(),
      bias.ptr<half>(),
      output.ptr<half>(),
      num_patches,
      hidden_size,
      patch_dim,
      in_channels,
      temporal_patch_size,
      patch_size);
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
  int h_div = grid_h / spatial_merge_size;  // number of 2x2 blocks in height
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
  
  float pos_val = w00 * __half2float(pos_embed[idx00]) +
                  w01 * __half2float(pos_embed[idx01]) +
                  w10 * __half2float(pos_embed[idx10]) +
                  w11 * __half2float(pos_embed[idx11]);
  
  // Add to patch embedding (both are in 2x2 block-interleaved order)
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
  
  int num_out_tokens = num_patches / merge_area;
  int out_hidden_size = hidden_size * merge_area;
  
  int token_idx = blockIdx.x;
  int hidden_idx = threadIdx.x + blockIdx.y * blockDim.x;
  
  if (token_idx >= num_out_tokens || hidden_idx >= out_hidden_size) return;
  
  // For block-interleaved input, every 4 consecutive patches form one output token
  // hidden_idx = [0, out_hidden_size) = [0, hidden_size * 4)
  // Split into: which patch within the block (0-3) and which element within the patch
  int local_patch = hidden_idx / hidden_size;  // 0, 1, 2, or 3
  int local_hidden = hidden_idx % hidden_size;
  
  // Input patch index: token_idx * 4 + local_patch
  int in_patch_idx = token_idx * merge_area + local_patch;
  
  // Copy value
  int in_idx = in_patch_idx * hidden_size + local_hidden;
  int out_idx = token_idx * out_hidden_size + hidden_idx;
  output[out_idx] = input[in_idx];
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
  dim3 grid_size(num_out_tokens, (out_hidden_size + 255) / 256);
  
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

void vision_merger_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& norm_weight,
    const tensor::Tensor& norm_bias,
    const tensor::Tensor& fc1_weight,
    const tensor::Tensor& fc1_bias,
    const tensor::Tensor& fc2_weight,
    const tensor::Tensor& fc2_bias,
    tensor::Tensor& output,
    tensor::Tensor& intermediate,
    float eps,
    bool use_postshuffle_norm,
    const kernel::CudaConfig* config) {
  
  int num_tokens = input.get_dim(0);
  int merged_hidden = input.get_dim(1);
  int out_hidden = fc2_weight.get_dim(0);
  
  // For deepstack merger: norm is applied after spatial shuffle
  // For main merger: norm is applied before spatial shuffle
  
  // Step 1: LayerNorm
  tensor::Tensor normed(input.data_type(), num_tokens, merged_hidden, true, input.get_buffer()->allocator());
  if (use_postshuffle_norm) {
    // Norm after shuffle: input is already merged [num_tokens, merged_hidden]
    layernorm_with_bias_cu(input, norm_weight, norm_bias, normed, eps, config->stream);
  } else {
    // Norm before shuffle: need to reshape, norm, then flatten
    // For simplicity, we still apply LayerNorm per token
    layernorm_with_bias_cu(input, norm_weight, norm_bias, normed, eps, config->stream);
  }
  
  // Step 2: fc1 + GELU
  const half alpha = __float2half(1.0f);
  const half beta = __float2half(0.0f);
  
  cublasHgemm(config->cublas_handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              merged_hidden, num_tokens, merged_hidden,
              &alpha,
              fc1_weight.ptr<half>(), merged_hidden,
              normed.ptr<half>(), merged_hidden,
              &beta,
              intermediate.ptr<half>(), merged_hidden);
  
  bias_add_residual_cu(intermediate, fc1_bias, tensor::Tensor(), intermediate, config->stream);
  gelu_cu(intermediate, intermediate, config->stream);
  
  // Step 3: fc2 + bias
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
// Image Token Replacement
// ============================================================================

__global__ void replace_image_tokens_kernel(
    const half* __restrict__ text_embeds,
    const half* __restrict__ visual_embeds,
    half* __restrict__ output,
    const int32_t* __restrict__ token_ids,
    int seq_len,
    int image_token_id,
    int num_vision_tokens,
    int hidden_size) {
  
  // This kernel needs to be more sophisticated for production
  // For now, it's a placeholder that copies embeddings
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = seq_len * hidden_size;
  
  if (idx < total_size) {
    // Simple copy for now - real implementation needs to track positions
    output[idx] = text_embeds[idx];
  }
}

void replace_image_tokens_cu(
    const tensor::Tensor& text_embeds,
    const tensor::Tensor& visual_embeds,
    tensor::Tensor& output,
    const int32_t* token_ids,
    int seq_len,
    int image_token_id,
    int num_vision_tokens,
    int hidden_size,
    cudaStream_t stream) {
  
  int block_size = 256;
  int grid_size = (seq_len * hidden_size + block_size - 1) / block_size;
  
  replace_image_tokens_kernel<<<grid_size, block_size, 0, stream>>>(
      text_embeds.ptr<half>(),
      visual_embeds.ptr<half>(),
      output.ptr<half>(),
      token_ids,
      seq_len,
      image_token_id,
      num_vision_tokens,
      hidden_size);
}

// ============================================================================
// Split QKV from interleaved layout
// ============================================================================

__global__ void split_qkv_fp16_kernel(
    const half* __restrict__ qkv,    // [num_tokens, 3 * hidden_size]
    half* __restrict__ q,            // [num_tokens, hidden_size]
    half* __restrict__ k,            // [num_tokens, hidden_size]
    half* __restrict__ v,            // [num_tokens, hidden_size]
    int num_tokens,
    int hidden_size) {
  
  int token_idx = blockIdx.x;
  int dim_idx = threadIdx.x + blockIdx.y * blockDim.x;
  
  if (token_idx >= num_tokens || dim_idx >= hidden_size) return;
  
  int qkv_offset = token_idx * 3 * hidden_size;
  int output_offset = token_idx * hidden_size + dim_idx;
  
  q[output_offset] = qkv[qkv_offset + dim_idx];
  k[output_offset] = qkv[qkv_offset + hidden_size + dim_idx];
  v[output_offset] = qkv[qkv_offset + 2 * hidden_size + dim_idx];
}

/**
 * Fused Split QKV + Transpose kernel
 * Input: qkv [num_tokens, 3 * num_heads * head_dim]
 * Output: q, k, v each [num_heads, num_tokens, head_dim]
 * 
 * This avoids separate split and transpose operations
 */
__global__ void split_qkv_transpose_fp16_kernel(
    const half* __restrict__ qkv,    // [num_tokens, 3 * hidden_size]
    half* __restrict__ q_out,        // [num_heads, num_tokens, head_dim]
    half* __restrict__ k_out,        // [num_heads, num_tokens, head_dim]
    half* __restrict__ v_out,        // [num_heads, num_tokens, head_dim]
    int num_tokens,
    int num_heads,
    int head_dim) {
  
  int hidden_size = num_heads * head_dim;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = num_tokens * hidden_size;
  
  if (idx >= total_elements) return;
  
  // Decode indices for output [num_heads, num_tokens, head_dim]
  int d = idx % head_dim;
  int temp = idx / head_dim;
  int t = temp % num_tokens;
  int h = temp / num_tokens;
  
  // Input index: qkv[t, q/k/v, h, d] = qkv[t * 3 * hidden_size + offset + h * head_dim + d]
  int input_base = t * 3 * hidden_size + h * head_dim + d;
  
  q_out[idx] = qkv[input_base];
  k_out[idx] = qkv[input_base + hidden_size];
  v_out[idx] = qkv[input_base + 2 * hidden_size];
}

void split_qkv_transpose_cu(
    const tensor::Tensor& qkv,
    tensor::Tensor& q_out,
    tensor::Tensor& k_out,
    tensor::Tensor& v_out,
    int num_tokens,
    int num_heads,
    int head_dim,
    cudaStream_t stream) {
  
  int hidden_size = num_heads * head_dim;
  int total_elements = num_tokens * hidden_size;
  int block_size = 256;
  int grid_size = (total_elements + block_size - 1) / block_size;
  
  split_qkv_transpose_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
      qkv.ptr<half>(),
      q_out.ptr<half>(),
      k_out.ptr<half>(),
      v_out.ptr<half>(),
      num_tokens,
      num_heads,
      head_dim);
}

void split_qkv_cu(
    const tensor::Tensor& qkv,
    tensor::Tensor& q,
    tensor::Tensor& k,
    tensor::Tensor& v,
    int num_tokens,
    int hidden_size,
    cudaStream_t stream) {
  
  dim3 block_size(256);
  dim3 grid_size(num_tokens, (hidden_size + 255) / 256);
  
  split_qkv_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
      qkv.ptr<half>(),
      q.ptr<half>(),
      k.ptr<half>(),
      v.ptr<half>(),
      num_tokens,
      hidden_size);
}

// ============================================================================
// Vision RoPE Implementation (2D Position Embedding)
// ============================================================================

/**
 * Vision RoPE uses 2D positions (height, width) instead of 1D sequential positions.
 * The rotation is applied to the first half of head_dim (head_dim/2 for cos/sin pairs).
 * 
 * For vision tokens at position (h, w):
 *   - First half of head_dim/2 uses height position
 *   - Second half of head_dim/2 uses width position
 * 
 * This implements: apply_rotary_pos_emb_vision from transformers
 *   q_embed = q * cos + rotate_half(q) * sin
 *   k_embed = k * cos + rotate_half(k) * sin
 */
__global__ void vision_rope_fp16_kernel(
    half* __restrict__ q,        // [num_tokens, num_heads * head_dim]
    half* __restrict__ k,        // [num_tokens, num_heads * head_dim]
    const half* __restrict__ cos_cache,  // [num_tokens, head_dim]
    const half* __restrict__ sin_cache,  // [num_tokens, head_dim]
    int num_tokens,
    int num_heads,
    int head_dim) {
  
  int token_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  
  if (token_idx >= num_tokens || head_idx >= num_heads) return;
  
  int hidden_size = num_heads * head_dim;
  int half_head_dim = head_dim / 2;
  
  half* q_token = q + token_idx * hidden_size + head_idx * head_dim;
  half* k_token = k + token_idx * hidden_size + head_idx * head_dim;
  const half* cos_ptr = cos_cache + token_idx * head_dim;
  const half* sin_ptr = sin_cache + token_idx * head_dim;
  
  // Process each dimension pair in the head
  for (int d = threadIdx.x; d < half_head_dim; d += blockDim.x) {
    // Get original values
    float q1 = __half2float(q_token[d]);
    float q2 = __half2float(q_token[d + half_head_dim]);
    float k1 = __half2float(k_token[d]);
    float k2 = __half2float(k_token[d + half_head_dim]);
    
    // Get rotation angles
    float cos_val = __half2float(cos_ptr[d]);
    float sin_val = __half2float(sin_ptr[d]);
    float cos_val2 = __half2float(cos_ptr[d + half_head_dim]);
    float sin_val2 = __half2float(sin_ptr[d + half_head_dim]);
    
    // Apply rotation (rotate_half swaps and negates)
    // q_embed = q * cos + rotate_half(q) * sin
    // rotate_half: [x1, x2] -> [-x2, x1]
    float q1_rot = q1 * cos_val - q2 * sin_val;
    float q2_rot = q2 * cos_val2 + q1 * sin_val2;
    float k1_rot = k1 * cos_val - k2 * sin_val;
    float k2_rot = k2 * cos_val2 + k1 * sin_val2;
    
    q_token[d] = __float2half(q1_rot);
    q_token[d + half_head_dim] = __float2half(q2_rot);
    k_token[d] = __float2half(k1_rot);
    k_token[d + half_head_dim] = __float2half(k2_rot);
  }
}

void vision_rope_cu(
    tensor::Tensor& q,
    tensor::Tensor& k,
    const tensor::Tensor& cos_cache,
    const tensor::Tensor& sin_cache,
    int num_heads,
    int head_dim,
    cudaStream_t stream) {
  
  int num_tokens = q.get_dim(0);
  
  dim3 block_size(64);  // Process multiple dimensions per thread block
  dim3 grid_size(num_tokens, num_heads);
  
  vision_rope_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
      q.ptr<half>(),
      k.ptr<half>(),
      cos_cache.ptr<half>(),
      sin_cache.ptr<half>(),
      num_tokens,
      num_heads,
      head_dim);
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
    
    // Apply RoPE rotation
    // rotate_half: [x1, x2] -> [-x2, x1]
    float q1_rot = q1 * cos_val - q2 * sin_val;
    float q2_rot = q2 * cos_val2 + q1 * sin_val2;
    float k1_rot = k1 * cos_val - k2 * sin_val;
    float k2_rot = k2 * cos_val2 + k1 * sin_val2;
    
    // Write to transposed output
    q_out[d] = __float2half(q1_rot);
    q_out[d + half_head_dim] = __float2half(q2_rot);
    k_out[d] = __float2half(k1_rot);
    k_out[d + half_head_dim] = __float2half(k2_rot);
  }
  
  // V doesn't need RoPE, just copy with transpose
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    v_out[d] = v_in[d];
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
  
  // Launch with grid [num_heads, num_tokens], block [64]
  // Each block handles one (head, token) pair
  dim3 grid(num_heads, num_tokens);
  dim3 block(64);  // 64 threads to cover head_dim=72
  
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
 * Transpose kernel: [total_tokens, num_heads * head_dim] -> [num_heads, total_tokens, head_dim]
 * Optimized with half2 vectorization (processes 2 halfs at a time)
 * head_dim=72 is divisible by 2
 */
__global__ void transpose_token_head_kernel(
    const half* __restrict__ input,   // [total_tokens, num_heads * head_dim]
    half* __restrict__ output,        // [num_heads, total_tokens, head_dim]
    int total_tokens,
    int num_heads,
    int head_dim) {
  
  // Process 2 elements (1 half2) at a time
  const int vec_size = 2;
  const int head_dim_vec = head_dim / vec_size;  // 72 / 2 = 36
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_vecs = total_tokens * num_heads * head_dim_vec;
  
  if (idx >= total_vecs) return;
  
  // Decode indices for vectorized access
  // output layout: [h, t, d_vec * 2]
  int d_vec = idx % head_dim_vec;
  int temp = idx / head_dim_vec;
  int t = temp % total_tokens;
  int h = temp / total_tokens;
  
  // Input: [t, h * head_dim + d_vec * 2]
  int input_base = t * (num_heads * head_dim) + h * head_dim + d_vec * vec_size;
  int output_base = idx * vec_size;
  
  // Load and store using half2
  half2 data = *reinterpret_cast<const half2*>(&input[input_base]);
  *reinterpret_cast<half2*>(&output[output_base]) = data;
}

/**
 * Transpose kernel: [num_heads, total_tokens, head_dim] -> [total_tokens, num_heads * head_dim]
 * Optimized with half2 vectorization
 */
__global__ void transpose_head_token_kernel(
    const half* __restrict__ input,   // [num_heads, total_tokens, head_dim]
    half* __restrict__ output,        // [total_tokens, num_heads * head_dim]
    int total_tokens,
    int num_heads,
    int head_dim) {
  
  const int vec_size = 2;
  const int head_dim_vec = head_dim / vec_size;
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_vecs = total_tokens * num_heads * head_dim_vec;
  
  if (idx >= total_vecs) return;
  
  // Decode indices from input layout [h, t, d_vec * 2]
  int d_vec = idx % head_dim_vec;
  int temp = idx / head_dim_vec;
  int t = temp % total_tokens;
  int h = temp / total_tokens;
  
  // Output: [t, h * head_dim + d_vec * 2]
  int input_base = idx * vec_size;
  int output_base = t * (num_heads * head_dim) + h * head_dim + d_vec * vec_size;
  
  half2 data = *reinterpret_cast<const half2*>(&input[input_base]);
  *reinterpret_cast<half2*>(&output[output_base]) = data;
}

// ============================================================================
// Tiled Vision FlashAttention Kernel - FlashAttention-3 Style
// ============================================================================

/**
 * High-efficiency FlashAttention for Vision Transformer
 * 
 * Key design:
 * - Each block processes TILE_Q queries (e.g., 32)
 * - All queries in a block SHARE K/V tiles in shared memory
 * - K/V data reuse: each K/V tile is loaded once, used by all TILE_Q queries
 * - Dramatically reduces global memory reads vs single-query-per-block
 * 
 * Memory model:
 * - Grid: [num_heads, ceil(seq_len/TILE_Q)]
 * - Each block: 256 threads organized as 8 warps
 * - Shared memory: K tile [TILE_K, head_dim] + V tile [TILE_K, head_dim]
 * 
 * For ViT with 3876 tokens, 16 heads, head_dim=72:
 * - Old approach: 16 * 3876 = 62016 blocks, each reading all K/V
 * - New approach: 16 * ceil(3876/32) = 16 * 122 = 1952 blocks
 */
constexpr int VIT3_TILE_Q = 32;    // Queries per block
constexpr int VIT3_TILE_K = 32;    // K/V per tile (reduced for shared memory)
constexpr int VIT3_BLOCK_SIZE = 256;

__global__ void vision_flash_attention_v3_kernel(
    const half* __restrict__ Q,        // [num_heads, seq_len, head_dim]
    const half* __restrict__ K,        // [num_heads, seq_len, head_dim]
    const half* __restrict__ V,        // [num_heads, seq_len, head_dim]
    half* __restrict__ O,              // [num_heads, seq_len, head_dim]
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const float scale
) {
    const int head_idx = blockIdx.x;
    const int q_tile_idx = blockIdx.y;
    const int q_start = q_tile_idx * VIT3_TILE_Q;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    if (head_idx >= num_heads) return;
    
    // Calculate head offset
    const long long head_offset = static_cast<long long>(head_idx) * seq_len * head_dim;
    const half* q_base = Q + head_offset + q_start * head_dim;
    const half* k_base = K + head_offset;
    const half* v_base = V + head_offset;
    half* o_base = O + head_offset + q_start * head_dim;
    
    // Determine which query this warp handles
    // 8 warps handle 8 queries in parallel, then we loop for remaining queries
    const int queries_per_iteration = 8;  // 8 warps = 8 queries in parallel
    
    // Shared memory layout:
    // s_k: [VIT3_TILE_K, head_dim] - 32 * 72 * 2 = 4608 bytes
    // s_v: [VIT3_TILE_K, head_dim] - 32 * 72 * 2 = 4608 bytes
    // s_q: [queries_per_iteration, head_dim] - 8 * 72 * 2 = 1152 bytes
    // Total: ~10KB per block
    extern __shared__ char smem[];
    half* s_k = reinterpret_cast<half*>(smem);
    half* s_v = reinterpret_cast<half*>(smem + VIT3_TILE_K * head_dim * sizeof(half));
    half* s_q = reinterpret_cast<half*>(smem + 2 * VIT3_TILE_K * head_dim * sizeof(half));
    
    // Process queries in groups of 8 (one per warp)
    for (int q_offset = 0; q_offset < VIT3_TILE_Q; q_offset += queries_per_iteration) {
        const int my_q_local = warp_id;  // Which query within this iteration
        const int my_q_global = q_start + q_offset + my_q_local;
        const bool valid_q = (my_q_global < seq_len) && (q_offset + my_q_local < VIT3_TILE_Q);
        
        // Load queries to shared memory (all 8 queries for this iteration)
        for (int i = tid; i < queries_per_iteration * head_dim; i += VIT3_BLOCK_SIZE) {
            int q_idx = i / head_dim;
            int d = i % head_dim;
            int global_q = q_start + q_offset + q_idx;
            if (global_q < seq_len && q_idx < queries_per_iteration) {
                s_q[q_idx * head_dim + d] = q_base[(q_offset + q_idx) * head_dim + d];
            }
        }
        __syncthreads();
        
        // Per-query accumulators (stored in registers)
        // Each warp handles one query, lanes handle different output dimensions
        float acc_o[3] = {0.0f, 0.0f, 0.0f};  // ceil(72/32) = 3 dims per lane
        float row_max = -FLT_MAX;
        float row_sum = 0.0f;
        
        // Load my query to registers
        float q_reg[3];
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            int d = lane_id + i * 32;
            if (valid_q && d < head_dim) {
                q_reg[i] = __half2float(s_q[my_q_local * head_dim + d]);
            } else {
                q_reg[i] = 0.0f;
            }
        }
        
        // Process K/V in tiles
        for (int k_start = 0; k_start < seq_len; k_start += VIT3_TILE_K) {
            const int k_len = min(VIT3_TILE_K, seq_len - k_start);
            
            // All threads cooperatively load K and V tiles
            for (int i = tid; i < VIT3_TILE_K * head_dim; i += VIT3_BLOCK_SIZE) {
                int k_idx = i / head_dim;
                int d = i % head_dim;
                if (k_idx < k_len) {
                    int kv_pos = k_start + k_idx;
                    s_k[k_idx * head_dim + d] = k_base[kv_pos * head_dim + d];
                    s_v[k_idx * head_dim + d] = v_base[kv_pos * head_dim + d];
                }
            }
            __syncthreads();
            
            if (!valid_q) {
                __syncthreads();
                continue;
            }
            
            // Each lane computes part of Q·K dot product, then warp reduces
            float tile_max = -FLT_MAX;
            float local_scores[VIT3_TILE_K];
            
            for (int k_idx = 0; k_idx < k_len; k_idx++) {
                // Compute Q·K for this query-key pair
                // Each lane computes partial dot product for its dimensions
                float partial_score = 0.0f;
                #pragma unroll
                for (int i = 0; i < 3; i++) {
                    int d = lane_id + i * 32;
                    if (d < head_dim) {
                        float k_val = __half2float(s_k[k_idx * head_dim + d]);
                        partial_score += q_reg[i] * k_val;
                    }
                }
                
                // Warp reduce to get full dot product
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    partial_score += __shfl_xor_sync(0xffffffff, partial_score, offset);
                }
                
                // Lane 0 has the final score
                float score = partial_score * scale;
                local_scores[k_idx] = score;
                tile_max = fmaxf(tile_max, score);
            }
            
            // Update running max and apply correction
            float m_new = fmaxf(row_max, tile_max);
            float correction = expf(row_max - m_new);
            
            #pragma unroll
            for (int i = 0; i < 3; i++) {
                acc_o[i] *= correction;
            }
            
            // Compute exp(score - m_new), sum, and accumulate V
            float tile_sum = 0.0f;
            for (int k_idx = 0; k_idx < k_len; k_idx++) {
                float score = local_scores[k_idx];
                float exp_score = (score - m_new > -20.0f) ? expf(score - m_new) : 0.0f;
                tile_sum += exp_score;
                
                // Accumulate V weighted by attention
                #pragma unroll
                for (int i = 0; i < 3; i++) {
                    int d = lane_id + i * 32;
                    if (d < head_dim) {
                        float v_val = __half2float(s_v[k_idx * head_dim + d]);
                        acc_o[i] += exp_score * v_val;
                    }
                }
            }
            
            row_max = m_new;
            row_sum = correction * row_sum + tile_sum;
            __syncthreads();
        }
        
        // Final normalization and write output
        if (valid_q) {
            float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
            half* o_ptr = o_base + (q_offset + my_q_local) * head_dim;
            
            #pragma unroll
            for (int i = 0; i < 3; i++) {
                int d = lane_id + i * 32;
                if (d < head_dim) {
                    o_ptr[d] = __float2half(acc_o[i] * inv_sum);
                }
            }
        }
        __syncthreads();
    }
}

// Legacy v2 kernel (kept for reference)
constexpr int VIT_BLOCK_SIZE = 128;
constexpr int VIT_TILE_K = 64;  // Process 64 K/V at a time (fits in shared memory)

__global__ void vision_flash_attention_v2_kernel(
    const half* __restrict__ Q,        // [num_heads, seq_len, head_dim]
    const half* __restrict__ K,        // [num_heads, seq_len, head_dim]
    const half* __restrict__ V,        // [num_heads, seq_len, head_dim]
    half* __restrict__ O,              // [num_heads, seq_len, head_dim]
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const float scale
) {
    const int head_idx = blockIdx.x;
    const int query_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (head_idx >= num_heads || query_idx >= seq_len) return;
    
    // Calculate head offset
    const long long head_offset = static_cast<long long>(head_idx) * seq_len * head_dim;
    const half* q_ptr = Q + head_offset + query_idx * head_dim;
    const half* k_base = K + head_offset;
    const half* v_base = V + head_offset;
    half* o_ptr = O + head_offset + query_idx * head_dim;
    
    // Shared memory for K tile: [VIT_TILE_K, head_dim]
    // head_dim = 72, VIT_TILE_K = 64 -> 64 * 72 * 2 = 9216 bytes
    extern __shared__ char smem[];
    half* s_k = reinterpret_cast<half*>(smem);
    half* s_v = reinterpret_cast<half*>(smem + VIT_TILE_K * head_dim * sizeof(half));
    float* s_scores = reinterpret_cast<float*>(smem + 2 * VIT_TILE_K * head_dim * sizeof(half));
    
    // Load query to registers (72 dims, spread across 128 threads)
    // Each thread handles ceil(72/128) = 1 dimension
    float q_reg = 0.0f;
    if (tid < head_dim) {
        q_reg = __half2float(q_ptr[tid]);
    }
    
    // Per-thread accumulator for output dimension
    float acc_o = 0.0f;
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    
    // Process K/V in tiles
    for (int tile_start = 0; tile_start < seq_len; tile_start += VIT_TILE_K) {
        const int tile_len = min(VIT_TILE_K, seq_len - tile_start);
        
        // Cooperatively load K and V tiles to shared memory
        for (int i = tid; i < tile_len * head_dim; i += VIT_BLOCK_SIZE) {
            int k_idx = i / head_dim;
            int d = i % head_dim;
            int kv_pos = tile_start + k_idx;
            s_k[k_idx * head_dim + d] = k_base[kv_pos * head_dim + d];
            s_v[k_idx * head_dim + d] = v_base[kv_pos * head_dim + d];
        }
        __syncthreads();
        
        // Compute Q·K scores for this tile
        // Each thread computes multiple scores
        float tile_max = -FLT_MAX;
        for (int k_idx = tid; k_idx < tile_len; k_idx += VIT_BLOCK_SIZE) {
            const half* k_ptr = s_k + k_idx * head_dim;
            
            // Dot product using half2 vectorization
            float score = 0.0f;
            const int head_dim_h2 = head_dim / 2;
            const half2* q_h2 = reinterpret_cast<const half2*>(q_ptr);
            const half2* k_h2 = reinterpret_cast<const half2*>(k_ptr);
            
            #pragma unroll
            for (int d = 0; d < head_dim_h2; d++) {
                float2 q_val = __half22float2(q_h2[d]);
                float2 k_val = __half22float2(k_h2[d]);
                score += q_val.x * k_val.x + q_val.y * k_val.y;
            }
            
            score *= scale;
            s_scores[k_idx] = score;
            tile_max = fmaxf(tile_max, score);
        }
        __syncthreads();
        
        // Block reduce for tile max
        __shared__ float s_tile_max;
        {
            // Warp reduce
            for (int offset = 16; offset > 0; offset >>= 1) {
                tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, offset));
            }
            // First lane of each warp writes to shared memory
            __shared__ float warp_max[4];  // 128 threads = 4 warps
            int warp_id = tid / 32;
            int lane_id = tid % 32;
            if (lane_id == 0) warp_max[warp_id] = tile_max;
            __syncthreads();
            if (tid == 0) {
                float m = warp_max[0];
                for (int i = 1; i < 4; i++) m = fmaxf(m, warp_max[i]);
                s_tile_max = m;
            }
            __syncthreads();
        }
        float m_j = s_tile_max;
        
        // Update running max and apply correction
        float m_new = fmaxf(row_max, m_j);
        float correction = expf(row_max - m_new);
        acc_o *= correction;
        
        // Compute exp(score - m_new), sum, and accumulate V
        float tile_sum = 0.0f;
        for (int k_idx = tid; k_idx < tile_len; k_idx += VIT_BLOCK_SIZE) {
            float score = s_scores[k_idx];
            float exp_score = (score - m_new > -20.0f) ? expf(score - m_new) : 0.0f;
            s_scores[k_idx] = exp_score;  // Store for V accumulation
            tile_sum += exp_score;
        }
        __syncthreads();
        
        // Block reduce for tile sum
        __shared__ float s_tile_sum;
        {
            for (int offset = 16; offset > 0; offset >>= 1) {
                tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, offset);
            }
            __shared__ float warp_sum[4];
            int warp_id = tid / 32;
            int lane_id = tid % 32;
            if (lane_id == 0) warp_sum[warp_id] = tile_sum;
            __syncthreads();
            if (tid == 0) {
                float s = 0.0f;
                for (int i = 0; i < 4; i++) s += warp_sum[i];
                s_tile_sum = s;
            }
            __syncthreads();
        }
        float l_j = s_tile_sum;
        
        // Accumulate V weighted by attention scores
        // Each thread handles one output dimension
        if (tid < head_dim) {
            float v_acc = 0.0f;
            for (int k_idx = 0; k_idx < tile_len; k_idx++) {
                float exp_score = s_scores[k_idx];
                float v_val = __half2float(s_v[k_idx * head_dim + tid]);
                v_acc += exp_score * v_val;
            }
            acc_o += v_acc;
        }
        
        row_max = m_new;
        row_sum = correction * row_sum + l_j;
        __syncthreads();
    }
    
    // Final normalization and write output
    if (tid < head_dim) {
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        o_ptr[tid] = __float2half(acc_o * inv_sum);
    }
}

// Legacy tiled kernel (kept for reference)
constexpr int VIT_TILE_K_LEGACY = 128;
constexpr int VIT_TILE_Q = 4;

template <int BLOCK_SIZE = VIT_BLOCK_SIZE, int TILE_K = VIT_TILE_K_LEGACY, int TILE_Q = VIT_TILE_Q>
__global__ void vision_flash_attention_tiled_kernel(
    const half* __restrict__ Q,        // [num_heads, seq_len, head_dim]
    const half* __restrict__ K,        // [num_heads, seq_len, head_dim]
    const half* __restrict__ V,        // [num_heads, seq_len, head_dim]
    half* __restrict__ O,              // [num_heads, seq_len, head_dim]
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const float scale
) {
    const int head_idx = blockIdx.x;
    const int query_block_idx = blockIdx.y;
    const int query_start = query_block_idx * TILE_Q;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    if (head_idx >= num_heads) return;
    
    // Each warp handles one query within the tile
    const int local_query_idx = warp_id;
    const int query_idx = query_start + local_query_idx;
    const bool valid_query = (query_idx < seq_len) && (local_query_idx < TILE_Q);
    
    // Shared memory layout:
    // - s_k: [TILE_K, head_dim] half - shared K tile for all warps
    extern __shared__ char smem[];
    half* s_k = reinterpret_cast<half*>(smem);
    
    // Pointers for this head
    const long long head_offset = static_cast<long long>(head_idx) * seq_len * head_dim;
    const half* q_ptr = valid_query ? (Q + head_offset + query_idx * head_dim) : nullptr;
    const half* k_base = K + head_offset;
    const half* v_base = V + head_offset;
    
    // Per-thread accumulators for output (head_dim / 32 elements per thread)
    const int dims_per_thread = (head_dim + 31) / 32;
    float acc_o[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Max 4 dims per thread for head_dim=72
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    
    // Process K/V in tiles
    for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_K) {
        const int tile_len = min(TILE_K, seq_len - tile_start);
        
        // Load K tile to shared memory (all threads cooperate)
        for (int i = tid; i < tile_len * head_dim; i += BLOCK_SIZE) {
            int k_idx = i / head_dim;
            int d = i % head_dim;
            int kv_pos = tile_start + k_idx;
            s_k[k_idx * head_dim + d] = k_base[kv_pos * head_dim + d];
        }
        __syncthreads();
        
        if (!valid_query) {
            __syncthreads();
            continue;
        }
        
        // Each lane computes partial Q·K scores
        float tile_max = -FLT_MAX;
        float local_scores[TILE_K / 32 + 1];  // Each lane computes TILE_K/32 scores
        
        for (int k_offset = lane_id; k_offset < tile_len; k_offset += 32) {
            const half* k_ptr = s_k + k_offset * head_dim;
            
            // Dot product Q·K with half2 vectorization
            float score = 0.0f;
            const int head_dim_h2 = head_dim / 2;
            const half2* q_h2 = reinterpret_cast<const half2*>(q_ptr);
            const half2* k_h2 = reinterpret_cast<const half2*>(k_ptr);
            
            #pragma unroll 
            for (int d = 0; d < head_dim_h2; d++) {
                float2 q_val = __half22float2(q_h2[d]);
                float2 k_val = __half22float2(k_h2[d]);
                score += q_val.x * k_val.x + q_val.y * k_val.y;
            }
            // Handle odd head_dim
            if (head_dim & 1) {
                score += __half2float(q_ptr[head_dim - 1]) * __half2float(k_ptr[head_dim - 1]);
            }
            
            score *= scale;
            local_scores[k_offset / 32] = score;
            tile_max = fmaxf(tile_max, score);
        }
        
        // Warp reduce for max
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, offset));
        }
        
        // Update running max
        float m_new = fmaxf(row_max, tile_max);
        float correction = expf(row_max - m_new);
        
        // Apply correction to running output
        #pragma unroll
        for (int i = 0; i < dims_per_thread && lane_id + i * 32 < head_dim; i++) {
            acc_o[i] *= correction;
        }
        
        // Compute exp(score - m_new), sum, and accumulate V
        float tile_sum = 0.0f;
        for (int k_offset = lane_id; k_offset < tile_len; k_offset += 32) {
            float score = local_scores[k_offset / 32];
            float exp_score = (score - m_new > -20.0f) ? expf(score - m_new) : 0.0f;
            tile_sum += exp_score;
            
            // Accumulate V weighted by attention
            const int kv_pos = tile_start + k_offset;
            const half* v_ptr = v_base + kv_pos * head_dim;
            
            #pragma unroll
            for (int i = 0; i < dims_per_thread && lane_id + i * 32 < head_dim; i++) {
                int d = lane_id + i * 32;
                acc_o[i] += exp_score * __half2float(v_ptr[d]);
            }
        }
        
        // Warp reduce for sum
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            tile_sum += __shfl_xor_sync(0xffffffff, tile_sum, offset);
        }
        
        row_max = m_new;
        row_sum = correction * row_sum + tile_sum;
        __syncthreads();
    }
    
    // Final normalization and write output
    if (valid_query) {
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        half* o_ptr = O + head_offset + query_idx * head_dim;
        
        #pragma unroll
        for (int i = 0; i < dims_per_thread && lane_id + i * 32 < head_dim; i++) {
            int d = lane_id + i * 32;
            o_ptr[d] = __float2half(acc_o[i] * inv_sum);
        }
    }
}

/**
 * Simple softmax kernel for attention scores
 */
// Helper function for atomic max on floats
__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, 
                        __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

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

void vision_flash_attention_cu(
    const tensor::Tensor& q,
    const tensor::Tensor& k,
    const tensor::Tensor& v,
    tensor::Tensor& output,
    tensor::Tensor& q_transposed,
    tensor::Tensor& k_transposed,
    tensor::Tensor& v_transposed,
    tensor::Tensor& out_transposed,
    tensor::Tensor& scores,
    const tensor::Tensor& cu_seqlens,
    int max_seqlen,
    float softmax_scale,
    const kernel::CudaConfig* config) {
  
  // Optimized vision attention using batched GEMM with pre-allocated workspace
  // This approach uses cuBLAS for maximum GEMM performance
  // Memory usage: scores matrix = [num_heads, seq_len, seq_len] = 481MB for 3876 tokens
  
  int total_tokens = q.get_dim(0);
  int hidden_size = q.get_dim(1);
  int num_heads = 16;  // Qwen3-VL Vision has 16 heads
  int head_dim = hidden_size / num_heads;  // 72
  
  cudaStream_t stream = config->stream;
  cublasHandle_t handle = config->cublas_handle;
  
  const half alpha_h = __float2half(1.0f);
  const half beta_h = __float2half(0.0f);
  
  // Transpose Q, K, V to [num_heads, seq_len, head_dim] layout
  // Using half2 vectorization, process 2 elements at a time
  int total_vecs = total_tokens * num_heads * (head_dim / 2);
  dim3 trans_block(256);
  dim3 trans_grid((total_vecs + 255) / 256);
  
  transpose_token_head_kernel<<<trans_grid, trans_block, 0, stream>>>(
      q.ptr<half>(), q_transposed.ptr<half>(), total_tokens, num_heads, head_dim);
  transpose_token_head_kernel<<<trans_grid, trans_block, 0, stream>>>(
      k.ptr<half>(), k_transposed.ptr<half>(), total_tokens, num_heads, head_dim);
  transpose_token_head_kernel<<<trans_grid, trans_block, 0, stream>>>(
      v.ptr<half>(), v_transposed.ptr<half>(), total_tokens, num_heads, head_dim);
  
  // Batched GEMM for scores: Q @ K^T for all heads
  // Q: [num_heads, total_tokens, head_dim]
  // K: [num_heads, total_tokens, head_dim]
  // scores: [num_heads, total_tokens, total_tokens]
  long long stride_q = total_tokens * head_dim;
  long long stride_k = total_tokens * head_dim;
  long long stride_s = static_cast<long long>(total_tokens) * total_tokens;
  
  cublasHgemmStridedBatched(
      handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      total_tokens, total_tokens, head_dim,
      &alpha_h,
      k_transposed.ptr<half>(), head_dim, stride_k,
      q_transposed.ptr<half>(), head_dim, stride_q,
      &beta_h,
      scores.ptr<half>(), total_tokens, stride_s,
      num_heads);
  
  // Apply softmax with scaling to each row
  dim3 softmax_grid(total_tokens * num_heads);
  dim3 softmax_block(256);
  vision_softmax_fp16_kernel<<<softmax_grid, softmax_block, 0, stream>>>(
      scores.ptr<half>(),
      total_tokens,
      num_heads,
      softmax_scale);
  
  // Batched GEMM for output: scores @ V for all heads
  long long stride_v = total_tokens * head_dim;
  long long stride_o = total_tokens * head_dim;
  
  cublasHgemmStridedBatched(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      head_dim, total_tokens, total_tokens,
      &alpha_h,
      v_transposed.ptr<half>(), head_dim, stride_v,
      scores.ptr<half>(), total_tokens, stride_s,
      &beta_h,
      out_transposed.ptr<half>(), head_dim, stride_o,
      num_heads);
  
  // Transpose output back: [num_heads, total_tokens, head_dim] -> [total_tokens, num_heads * head_dim]
  transpose_head_token_kernel<<<trans_grid, trans_block, 0, stream>>>(
      out_transposed.ptr<half>(), output.ptr<half>(), total_tokens, num_heads, head_dim);
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
  int total_vecs = num_tokens * num_heads * (head_dim / 2);
  dim3 trans_block(256);
  dim3 trans_grid((total_vecs + 255) / 256);
  
  transpose_head_token_kernel<<<trans_grid, trans_block, 0, stream>>>(
      out_transposed.ptr<half>(), output.ptr<half>(), num_tokens, num_heads, head_dim);
}

/**
 * @brief Vision attention using Flash Attention algorithm (no scores matrix storage)
 * 
 * Uses the v3 kernel which implements online softmax with tiling.
 * This avoids storing the full [num_heads, num_tokens, num_tokens] scores matrix,
 * reducing memory bandwidth significantly for large sequence lengths.
 * 
 * Memory savings for 3876 tokens, 16 heads:
 * - cuBLAS approach: 481MB for scores matrix
 * - Flash Attention: ~10KB shared memory per block
 */
void vision_attention_pretransposed_flash_cu(
    const tensor::Tensor& q_trans,      // [num_heads, num_tokens, head_dim]
    const tensor::Tensor& k_trans,      // [num_heads, num_tokens, head_dim]
    const tensor::Tensor& v_trans,      // [num_heads, num_tokens, head_dim]
    tensor::Tensor& output,             // [num_tokens, hidden_size]
    tensor::Tensor& out_transposed,     // workspace [num_heads, num_tokens, head_dim]
    int num_tokens,
    int num_heads,
    int head_dim,
    float softmax_scale,
    const kernel::CudaConfig* config) {
  
  cudaStream_t stream = config->stream;
  
  // Use Flash Attention v3 kernel
  // Grid: [num_heads, ceil(num_tokens / VIT3_TILE_Q)]
  // Each block handles one head and a tile of queries
  const int queries_per_block = VIT3_TILE_Q;  // 32
  const int num_q_tiles = (num_tokens + queries_per_block - 1) / queries_per_block;
  
  dim3 grid(num_heads, num_q_tiles);
  dim3 block(VIT3_BLOCK_SIZE);  // 256
  
  // Shared memory: s_k + s_v + s_q
  // s_k: [TILE_K, head_dim] = 32 * 72 * 2 = 4608 bytes
  // s_v: [TILE_K, head_dim] = 32 * 72 * 2 = 4608 bytes  
  // s_q: [8, head_dim] = 8 * 72 * 2 = 1152 bytes
  const int smem_size = 2 * VIT3_TILE_K * head_dim * sizeof(half) + 8 * head_dim * sizeof(half);
  
  vision_flash_attention_v3_kernel<<<grid, block, smem_size, stream>>>(
      q_trans.ptr<half>(),
      k_trans.ptr<half>(),
      v_trans.ptr<half>(),
      out_transposed.ptr<half>(),
      num_tokens,
      num_heads,
      head_dim,
      softmax_scale);
  
  // Transpose output back: [num_heads, num_tokens, head_dim] -> [num_tokens, num_heads * head_dim]
  int total_vecs = num_tokens * num_heads * (head_dim / 2);
  dim3 trans_block(256);
  dim3 trans_grid((total_vecs + 255) / 256);
  
  transpose_head_token_kernel<<<trans_grid, trans_block, 0, stream>>>(
      out_transposed.ptr<half>(), output.ptr<half>(), num_tokens, num_heads, head_dim);
}

// ============================================================================
// Fused QKV Projection
// ============================================================================

void fused_qkv_projection_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& qkv_weight,
    const tensor::Tensor& qkv_bias,
    tensor::Tensor& q_out,
    tensor::Tensor& k_out,
    tensor::Tensor& v_out,
    const kernel::CudaConfig* config) {
  
  int num_tokens = input.get_dim(0);
  int hidden_size = input.get_dim(1);
  
  // Allocate temporary buffer for full QKV output
  auto alloc = input.get_buffer()->allocator();
  tensor::Tensor qkv_out(base::DataType::kDataTypeFp16, num_tokens, 3 * hidden_size, true, alloc);
  
  const half alpha = __float2half(1.0f);
  const half beta = __float2half(0.0f);
  
  // GEMM: qkv_out = input @ qkv_weight.T
  cublasHgemm(config->cublas_handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              3 * hidden_size, num_tokens, hidden_size,
              &alpha,
              qkv_weight.ptr<half>(), hidden_size,
              input.ptr<half>(), hidden_size,
              &beta,
              qkv_out.ptr<half>(), 3 * hidden_size);
  
  // Add bias
  bias_add_residual_cu(qkv_out, qkv_bias, tensor::Tensor(), qkv_out, config->stream);
  
  // Split into Q, K, V
  cudaMemcpyAsync(q_out.ptr<void>(), qkv_out.ptr<half>(),
                  num_tokens * hidden_size * sizeof(half),
                  cudaMemcpyDeviceToDevice, config->stream);
  cudaMemcpyAsync(k_out.ptr<void>(), qkv_out.ptr<half>() + num_tokens * hidden_size,
                  num_tokens * hidden_size * sizeof(half),
                  cudaMemcpyDeviceToDevice, config->stream);
  cudaMemcpyAsync(v_out.ptr<void>(), qkv_out.ptr<half>() + 2 * num_tokens * hidden_size,
                  num_tokens * hidden_size * sizeof(half),
                  cudaMemcpyDeviceToDevice, config->stream);
}

}  // namespace kernel
