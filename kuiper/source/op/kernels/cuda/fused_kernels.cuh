#ifndef FUSED_KERNELS_CUH
#define FUSED_KERNELS_CUH

#include <base/cuda_config.h>
#include <tensor/tensor.h>

namespace kernel {

/**
 * Fused RMSNorm + GEMV operation
 * Combines normalization and matrix-vector multiplication in a single kernel
 * 
 * output = rmsnorm(input) @ gemv_weight
 * 
 * @param input Input tensor [dim]
 * @param rms_weight RMSNorm weight [dim]
 * @param gemv_weight GEMV weight matrix [out_dim, dim]
 * @param output Output tensor [out_dim]
 * @param eps RMSNorm epsilon
 * @param config CUDA configuration
 */
void fused_rmsnorm_gemv_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& rms_weight,
    const tensor::Tensor& gemv_weight,
    tensor::Tensor& output,
    float eps,
    CudaConfig* config
);

/**
 * Fused SiLU activation + elementwise multiply
 * output = silu(gate) * up
 * 
 * @param gate Gate tensor [dim]
 * @param up Up tensor [dim]
 * @param output Output tensor [dim]
 * @param config CUDA configuration
 */
void fused_silu_multiply_cu(
    const tensor::Tensor& gate,
    const tensor::Tensor& up,
    tensor::Tensor& output,
    CudaConfig* config
);

/**
 * Fused residual add + RMSNorm
 * output = rmsnorm(input + residual)
 * 
 * @param input Input tensor [dim]
 * @param residual Residual tensor [dim]
 * @param weight RMSNorm weight [dim]
 * @param output Output tensor [dim]
 * @param eps RMSNorm epsilon
 * @param config CUDA configuration
 */
void fused_add_rmsnorm_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& residual,
    const tensor::Tensor& weight,
    tensor::Tensor& output,
    float eps,
    CudaConfig* config
);

/**
 * Fused multimodal embedding assembly kernel
 * Combines text embeddings before/after image token with vision embeddings
 * in a single kernel, eliminating 3 separate cudaMemcpyAsync calls
 *
 * output[0:before_len] = text_embeds[0:before_len]
 * output[before_len:before_len+vision_len] = vision_embeds[0:vision_len]
 * output[before_len+vision_len:total] = text_embeds[before_len+1:end]
 *
 * @param text_embeds Text embeddings [text_seq_len, dim]
 * @param vision_embeds Vision embeddings [num_vision_tokens, dim]
 * @param output Output multimodal embeddings [total_seq_len, dim]
 * @param image_token_pos Position of image token in text sequence
 * @param num_vision_tokens Number of vision tokens
 * @param text_seq_len Length of text sequence (including image placeholder)
 * @param dim Hidden dimension
 * @param stream CUDA stream
 */
void fused_multimodal_embed_cu(
    const tensor::Tensor& text_embeds,
    const tensor::Tensor& vision_embeds,
    tensor::Tensor& output,
    int image_token_pos,
    int num_vision_tokens,
    int text_seq_len,
    int dim,
    cudaStream_t stream
);

/**
 * Fused KV cache update kernel
 * Copies both K and V to cache in a single kernel launch
 * 
 * @param key_out Key output from projection [seq_len, kv_dim]
 * @param value_out Value output from projection [seq_len, kv_dim]
 * @param key_cache Key cache [layers, max_seq_len, kv_dim]
 * @param value_cache Value cache [layers, max_seq_len, kv_dim]
 * @param layer_idx Current layer index
 * @param start_pos Starting position in cache
 * @param seq_len Sequence length to copy
 * @param kv_dim KV dimension
 * @param max_seq_len Maximum sequence length
 * @param stream CUDA stream
 */
void fused_kv_cache_update_cu(
    const tensor::Tensor& key_out,
    const tensor::Tensor& value_out,
    tensor::Tensor& key_cache,
    tensor::Tensor& value_cache,
    int layer_idx,
    int start_pos,
    int seq_len,
    int kv_dim,
    int max_seq_len,
    cudaStream_t stream
);

/**
 * GPU-based patch extraction with 2x2 block interleaved order
 * Extracts patches from image tensor directly on GPU, avoiding D2H and H2D copies
 *
 * Input: image [C, H, W] in CHW format (FP16)
 * Output: patches [num_patches, patch_dim] in 2x2 block interleaved order (FP16)
 *
 * The 2x2 block interleaved order matches HuggingFace Qwen3-VL:
 * For spatial_merge_size=2, patches are ordered as:
 *   block(0,0): (0,0), (0,1), (1,0), (1,1)
 *   block(0,1): (0,2), (0,3), (1,2), (1,3)
 *   ...
 *
 * @param image Input image tensor [C, H, W] on GPU (FP16)
 * @param patches Output patch tensor [num_patches, patch_dim] on GPU (FP16)
 * @param channels Number of channels (3)
 * @param height Image height
 * @param width Image width
 * @param patch_size Patch size (16)
 * @param temporal_patch_size Temporal patch size (2)
 * @param stream CUDA stream
 */
void extract_patches_cu(
    const tensor::Tensor& image,
    tensor::Tensor& patches,
    int channels,
    int height,
    int width,
    int patch_size,
    int temporal_patch_size,
    cudaStream_t stream
);

}  // namespace kernel

#endif  // FUSED_KERNELS_CUH
