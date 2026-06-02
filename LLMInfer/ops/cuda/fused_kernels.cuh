#ifndef FUSED_KERNELS_CUH
#define FUSED_KERNELS_CUH

#include <cuda_fp16.h>
#include <base/cuda_config.h>
#include <tensor/tensor.h>

namespace kernel {

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

/**
 * Fused normalize + patches kernel: directly converts uint8 HWC pixels
 * to fp16 normalized patches in 2x2 block interleaved order.
 * 
 * Eliminates CPU normalization, fp32->fp16 conversion, intermediate H2D copy,
 * and separate extract_patches kernel launch.
 *
 * @param pixels_gpu uint8 HWC pixels already on GPU [H, W, 3]
 * @param patches_gpu Output fp16 patches [num_patches, patch_dim]
 * @param height Image height (after smart_resize)
 * @param width Image width (after smart_resize)
 * @param patch_size Patch size (16)
 * @param temporal_patch_size Temporal patch size (2)
 * @param mean_r/g/b Normalization mean per channel
 * @param std_r/g/b Normalization std per channel
 * @param stream CUDA stream
 */
void fused_normalize_patches_cu(
    const unsigned char* pixels_gpu,
    half* patches_gpu,
    int height,
    int width,
    int patch_size,
    int temporal_patch_size,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b,
    cudaStream_t stream
);

/**
 * Fused resize + normalize + patches kernel: bicubic resize (Catmull-Rom)
 * with anti-aliasing, normalize, and 2x2 block interleaved patch extraction
 * in a single kernel.  Eliminates CPU stb resize entirely.
 *
 * @param src_pixels_gpu  uint8 HWC ORIGINAL (un-resized) pixels on GPU [src_H, src_W, 3]
 * @param patches_gpu     Output fp16 patches [num_patches, patch_dim]
 * @param src_h/src_w     Original image dimensions
 * @param dst_h/dst_w     Target (resized) image dimensions
 * @param patch_size      Patch size (16)
 * @param temporal_patch_size  Temporal patch size (2)
 * @param mean_r/g/b      Normalization mean per channel
 * @param std_r/g/b       Normalization std per channel
 * @param stream          CUDA stream
 */
void fused_resize_normalize_patches_cu(
    const unsigned char* src_pixels_gpu,
    half* patches_gpu,
    int src_h, int src_w,
    int dst_h, int dst_w,
    int patch_size,
    int temporal_patch_size,
    float mean_r, float mean_g, float mean_b,
    float std_r,  float std_g,  float std_b,
    cudaStream_t stream
);

}  // namespace kernel

#endif  // FUSED_KERNELS_CUH
