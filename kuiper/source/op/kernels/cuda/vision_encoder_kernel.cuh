/**
 * @file vision_encoder_kernel.cuh
 * @brief CUDA kernels for Qwen3-VL Vision Encoder
 * 
 * This file contains optimized CUDA kernels for the vision encoder operations:
 * - Patch embedding (Conv3D)
 * - Position embedding with bilinear interpolation
 * - LayerNorm with bias
 * - Self-attention (fused QKV projection)
 * - MLP with GELU activation
 * - Spatial merge operation
 */

#ifndef KUIPER_OP_KERNELS_CUDA_VISION_ENCODER_KERNEL_CUH_
#define KUIPER_OP_KERNELS_CUDA_VISION_ENCODER_KERNEL_CUH_

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include "tensor/tensor.h"
#include "base/cuda_config.h"

namespace kernel {

// ============================================================================
// Layer Normalization with Bias (for Vision Transformer)
// ============================================================================

/**
 * @brief LayerNorm kernel with bias (FP16)
 * 
 * Computes: y = (x - mean) / sqrt(var + eps) * weight + bias
 * 
 * @param input Input tensor [num_tokens, hidden_size]
 * @param weight Scale weights [hidden_size]
 * @param bias Bias weights [hidden_size]
 * @param output Output tensor [num_tokens, hidden_size]
 * @param num_tokens Number of tokens
 * @param hidden_size Hidden dimension
 * @param eps Epsilon for numerical stability
 */
__global__ void layernorm_with_bias_fp16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int num_tokens,
    int hidden_size,
    float eps);

void layernorm_with_bias_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& weight,
    const tensor::Tensor& bias,
    tensor::Tensor& output,
    float eps,
    cudaStream_t stream);

// ============================================================================
// GELU Activation
// ============================================================================

/**
 * @brief GELU activation (approximate, tanh version)
 * 
 * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
__global__ void gelu_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int size);

void gelu_cu(
    const tensor::Tensor& input,
    tensor::Tensor& output,
    cudaStream_t stream);

// ============================================================================
// Fused Bias + Add Residual
// ============================================================================

/**
 * @brief Add bias and residual connection
 * 
 * output = residual + (input + bias)
 */
__global__ void bias_add_residual_fp16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ bias,
    const half* __restrict__ residual,
    half* __restrict__ output,
    int size,
    int bias_size);

void bias_add_residual_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& bias,
    const tensor::Tensor& residual,
    tensor::Tensor& output,
    cudaStream_t stream);

// ============================================================================
// Vision Patch Embedding (Conv3D)
// ============================================================================

/**
 * @brief 3D convolution for patch embedding
 * 
 * Converts image patches to embeddings using Conv3D.
 * Input: [num_patches, C * temporal_patch * patch_size * patch_size]
 * Weight: [hidden_size, C, temporal_patch, patch_size, patch_size]
 * Output: [num_patches, hidden_size]
 * 
 * For single image: temporal dimension is padded (same frame repeated)
 */
__global__ void patch_embed_conv3d_fp16_kernel(
    const half* __restrict__ input,     // [num_patches, patch_dim]
    const half* __restrict__ weight,    // [hidden_size, C, T, H, W]
    const half* __restrict__ bias,      // [hidden_size]
    half* __restrict__ output,          // [num_patches, hidden_size]
    int num_patches,
    int hidden_size,
    int patch_dim,
    int in_channels,
    int temporal_patch_size,
    int patch_size);

void patch_embed_conv3d_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& weight,
    const tensor::Tensor& bias,
    tensor::Tensor& output,
    int in_channels,
    int temporal_patch_size,
    int patch_size,
    cudaStream_t stream);

// ============================================================================
// Position Embedding with Bilinear Interpolation
// ============================================================================

/**
 * @brief Add position embeddings with bilinear interpolation
 * 
 * Interpolates from a base position embedding grid (e.g., 48x48)
 * to the actual image grid size, then adds to patch embeddings.
 * 
 * @param patch_embeds Input patch embeddings [num_patches, hidden_size]
 * @param pos_embed Base position embeddings [num_pos_embed, hidden_size]
 * @param output Output with position added [num_patches, hidden_size]
 * @param grid_h Height of the patch grid
 * @param grid_w Width of the patch grid
 * @param grid_t Temporal dimension (frames)
 * @param num_grid_per_side Base grid size (e.g., 48 for 2304 positions)
 * @param spatial_merge_size Merge size for position indexing
 */
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
    int spatial_merge_size);

void pos_embed_interpolate_cu(
    const tensor::Tensor& patch_embeds,
    const tensor::Tensor& pos_embed,
    tensor::Tensor& output,
    int grid_h,
    int grid_w,
    int grid_t,
    int num_grid_per_side,
    int spatial_merge_size,
    cudaStream_t stream);

// ============================================================================
// Vision Self-Attention (Fused QKV)
// ============================================================================

/**
 * @brief Fused QKV projection for vision attention
 * 
 * Computes Q, K, V in a single pass:
 * qkv = input @ qkv_weight.T + qkv_bias
 * 
 * @param input Hidden states [num_tokens, hidden_size]
 * @param qkv_weight Fused weight [3*hidden_size, hidden_size]
 * @param qkv_bias Fused bias [3*hidden_size]
 * @param q_out Query output [num_tokens, hidden_size]
 * @param k_out Key output [num_tokens, hidden_size]
 * @param v_out Value output [num_tokens, hidden_size]
 */
void fused_qkv_projection_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& qkv_weight,
    const tensor::Tensor& qkv_bias,
    tensor::Tensor& q_out,
    tensor::Tensor& k_out,
    tensor::Tensor& v_out,
    const kernel::CudaConfig* config);

/**
 * @brief Vision self-attention with Flash Attention
 * 
 * Runs efficient self-attention for vision tokens using Flash Attention.
 * Supports variable-length sequences via cu_seqlens.
 * 
 * @param q_transposed Pre-allocated buffer [num_heads, total_tokens, head_dim]
 * @param k_transposed Pre-allocated buffer [num_heads, total_tokens, head_dim]
 * @param v_transposed Pre-allocated buffer [num_heads, total_tokens, head_dim]
 * @param out_transposed Pre-allocated buffer [num_heads, total_tokens, head_dim]
 * @param attn_scores Pre-allocated buffer [num_heads, total_tokens, total_tokens]
 */
void vision_flash_attention_cu(
    const tensor::Tensor& q,          // [total_tokens, hidden_size]
    const tensor::Tensor& k,          // [total_tokens, hidden_size]
    const tensor::Tensor& v,          // [total_tokens, hidden_size]
    tensor::Tensor& output,           // [total_tokens, hidden_size]
    tensor::Tensor& q_transposed,     // workspace
    tensor::Tensor& k_transposed,     // workspace
    tensor::Tensor& v_transposed,     // workspace
    tensor::Tensor& out_transposed,   // workspace
    tensor::Tensor& attn_scores,      // workspace
    const tensor::Tensor& cu_seqlens, // [batch_size + 1] cumulative sequence lengths
    int max_seqlen,
    float softmax_scale,
    const kernel::CudaConfig* config);  // Pass config for cuBLAS handle

/**
 * @brief Optimized vision attention with pre-transposed Q, K, V
 * 
 * This version skips the input transpose step because the caller
 * uses fused_split_rope_transpose_cu to produce already-transposed tensors.
 * 
 * @param q_trans [num_heads, num_tokens, head_dim] with RoPE applied
 * @param k_trans [num_heads, num_tokens, head_dim] with RoPE applied
 * @param v_trans [num_heads, num_tokens, head_dim]
 * @param output [num_tokens, hidden_size]
 */
void vision_attention_pretransposed_cu(
    const tensor::Tensor& q_trans,
    const tensor::Tensor& k_trans,
    const tensor::Tensor& v_trans,
    tensor::Tensor& output,
    tensor::Tensor& out_transposed,
    tensor::Tensor& scores,
    int num_tokens,
    int num_heads,
    int head_dim,
    float softmax_scale,
    const kernel::CudaConfig* config);

/**
 * @brief Vision attention using Flash Attention algorithm (no scores matrix storage)
 * 
 * Uses online softmax with tiling to avoid storing the full scores matrix.
 * More memory efficient for large sequence lengths.
 */
void vision_attention_pretransposed_flash_cu(
    const tensor::Tensor& q_trans,
    const tensor::Tensor& k_trans,
    const tensor::Tensor& v_trans,
    tensor::Tensor& output,
    tensor::Tensor& out_transposed,
    int num_tokens,
    int num_heads,
    int head_dim,
    float softmax_scale,
    const kernel::CudaConfig* config);

/**
 * @brief Split interleaved QKV tensor into separate Q, K, V tensors
 * 
 * Input: qkv [num_tokens, 3 * hidden_size] - for each token: [Q, K, V]
 * Output: q, k, v each [num_tokens, hidden_size]
 */
void split_qkv_cu(
    const tensor::Tensor& qkv,
    tensor::Tensor& q,
    tensor::Tensor& k,
    tensor::Tensor& v,
    int num_tokens,
    int hidden_size,
    cudaStream_t stream);

/**
 * @brief Fused Split QKV + Transpose for optimized attention
 * 
 * Input: qkv [num_tokens, 3 * hidden_size] - for each token: [Q, K, V]
 * Output: q, k, v each [num_heads, num_tokens, head_dim]
 * 
 * This fuses split and transpose operations to reduce memory bandwidth
 */
void split_qkv_transpose_cu(
    const tensor::Tensor& qkv,
    tensor::Tensor& q_out,
    tensor::Tensor& k_out,
    tensor::Tensor& v_out,
    int num_tokens,
    int num_heads,
    int head_dim,
    cudaStream_t stream);

/**
 * @brief Apply rotary position embedding for vision attention
 * 
 * Different from LLM RoPE: uses 2D position (height, width) with half rotation
 */
void vision_rope_cu(
    tensor::Tensor& q,
    tensor::Tensor& k,
    const tensor::Tensor& cos_cache,
    const tensor::Tensor& sin_cache,
    int num_heads,
    int head_dim,
    cudaStream_t stream);

/**
 * @brief Fused Split + RoPE + Transpose kernel for vision attention
 * 
 * Combines 3 operations into 1 kernel:
 * 1. Split QKV from [num_tokens, 3*hidden_size] to separate Q, K, V
 * 2. Apply RoPE to Q and K
 * 3. Transpose to [num_heads, num_tokens, head_dim] for batched GEMM
 * 
 * This saves 5 kernel launches and multiple global memory passes.
 */
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
    cudaStream_t stream);

// ============================================================================
// Spatial Merge Operation
// ============================================================================

/**
 * @brief Merge spatial patches for vision-to-LLM projection
 * 
 * Merges (spatial_merge_size x spatial_merge_size) patches into single tokens.
 * E.g., for merge_size=2, 4 adjacent patches become 1 token with 4x features.
 * 
 * Input: [num_patches, hidden_size] where num_patches = T * H * W
 * Output: [num_tokens, hidden_size * merge_size^2] where num_tokens = T * (H/2) * (W/2)
 * 
 * The patches are reordered to group 2x2 spatial neighbors together.
 */
__global__ void spatial_merge_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int grid_t,
    int grid_h,
    int grid_w,
    int hidden_size,
    int spatial_merge_size);

void spatial_merge_cu(
    const tensor::Tensor& input,
    tensor::Tensor& output,
    int grid_t,
    int grid_h,
    int grid_w,
    int hidden_size,
    int spatial_merge_size,
    cudaStream_t stream);

// ============================================================================
// Vision MLP
// ============================================================================

/**
 * @brief Fused vision MLP: Linear + GELU + Linear + Bias + Residual
 * 
 * Computes: output = residual + fc2(gelu(fc1(x) + bias1)) + bias2
 */
void vision_mlp_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& fc1_weight,
    const tensor::Tensor& fc1_bias,
    const tensor::Tensor& fc2_weight,
    const tensor::Tensor& fc2_bias,
    const tensor::Tensor& residual,
    tensor::Tensor& output,
    tensor::Tensor& intermediate,
    const kernel::CudaConfig* config);

// ============================================================================
// Merger MLP (Vision to LLM projection)
// ============================================================================

/**
 * @brief Merger projection: LayerNorm + Linear + GELU + Linear
 * 
 * Projects merged vision features to LLM hidden dimension.
 * Input: [num_tokens, merged_hidden] where merged_hidden = hidden_size * merge_size^2
 * Output: [num_tokens, out_hidden_size]
 */
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
    const kernel::CudaConfig* config);

/**
 * @brief Merger MLP only (no LayerNorm, which is done before spatial merge)
 * 
 * Projects merged vision features to LLM hidden dimension.
 * Input: [num_tokens, merged_hidden] where merged_hidden = hidden_size * merge_size^2
 * Output: [num_tokens, out_hidden_size]
 */
void vision_merger_mlp_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& fc1_weight,
    const tensor::Tensor& fc1_bias,
    const tensor::Tensor& fc2_weight,
    const tensor::Tensor& fc2_bias,
    tensor::Tensor& output,
    tensor::Tensor& intermediate,
    const kernel::CudaConfig* config);

// ============================================================================
// Image Token Replacement
// ============================================================================

/**
 * @brief Replace image placeholder tokens with visual embeddings
 * 
 * Finds <image_pad> tokens and replaces them with visual features.
 * Also handles deepstack features by concatenating them.
 * 
 * @param text_embeds Text token embeddings [seq_len, hidden_size]
 * @param visual_embeds Visual embeddings [num_vision_tokens, visual_hidden]
 * @param output Combined embeddings [new_seq_len, hidden_size]
 * @param token_ids Input token IDs to find image placeholder
 * @param image_token_id Token ID for image placeholder
 * @param num_vision_tokens Number of visual tokens
 * @param hidden_size LLM hidden dimension
 * @param visual_hidden Visual hidden dimension (with deepstack)
 */
void replace_image_tokens_cu(
    const tensor::Tensor& text_embeds,
    const tensor::Tensor& visual_embeds,
    tensor::Tensor& output,
    const int32_t* token_ids,
    int seq_len,
    int image_token_id,
    int num_vision_tokens,
    int hidden_size,
    cudaStream_t stream);

// ============================================================================
// M-RoPE Position Calculation
// ============================================================================

/**
 * @brief Calculate M-RoPE positions for multimodal input
 * 
 * M-RoPE uses 3D position encoding (temporal, height, width) for vision tokens
 * and 1D position for text tokens.
 * 
 * @param token_ids Input token IDs
 * @param positions Output position tensor [3, seq_len] for M-RoPE
 * @param is_multimodal Mask indicating which tokens are visual
 * @param grid_t, grid_h, grid_w Grid dimensions for visual tokens
 * @param vision_start_token_id Token ID for <vision_start>
 * @param vision_end_token_id Token ID for <vision_end>
 */
void calculate_mrope_positions_cu(
    const int32_t* token_ids,
    int32_t* positions,
    bool* is_multimodal,
    int seq_len,
    int grid_t,
    int grid_h,
    int grid_w,
    int vision_start_token_id,
    int vision_end_token_id,
    int spatial_merge_size,
    cudaStream_t stream);

}  // namespace kernel

#endif  // KUIPER_OP_KERNELS_CUDA_VISION_ENCODER_KERNEL_CUH_
