/**
 * Fused CUDA Kernels for LLM Inference Optimization
 * 
 * This file contains fused kernel implementations that combine multiple operations
 * to reduce kernel launch overhead and memory bandwidth requirements.
 * 
 * Optimizations:
 * - RMSNorm + GEMV fusion (save one global memory round trip)
 * - SiLU + elementwise multiply fusion
 * - RoPE + QKV split fusion
 * - half2 vectorization throughout
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include "fused_kernels.cuh"

namespace kernel {

constexpr int FUSED_BLOCK_SIZE = 256;

// ============================================================================
// Fused Multimodal Embedding Assembly Kernel
// ============================================================================

// ============================================================================
// Fused Multimodal Embedding Assembly Kernel
// ============================================================================

/**
 * Fused kernel to assemble multimodal embeddings from text and vision embeddings
 * Replaces 3 separate cudaMemcpyAsync calls with a single kernel launch
 * 
 * Memory layout:
 *   output[0 : image_token_pos] = text_embeds[0 : image_token_pos]
 *   output[image_token_pos : image_token_pos + num_vision] = vision_embeds
 *   output[image_token_pos + num_vision : end] = text_embeds[image_token_pos + 1 : end]
 */
__global__ void fused_multimodal_embed_fp16_kernel(
    const half* __restrict__ text_embeds,    // [text_seq_len, dim]
    const half* __restrict__ vision_embeds,  // [num_vision_tokens, dim]
    half* __restrict__ output,               // [total_seq_len, dim]
    const int image_token_pos,
    const int num_vision_tokens,
    const int text_seq_len,
    const int dim,
    const int total_seq_len
) {
    // Each block handles one row (token), threads handle elements within the row
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (token_idx >= total_seq_len) return;
    
    // Determine which source this token comes from
    const half* src_ptr;
    int src_offset;
    
    if (token_idx < image_token_pos) {
        // Before image token: copy from text_embeds[token_idx]
        src_ptr = text_embeds;
        src_offset = token_idx * dim;
    } else if (token_idx < image_token_pos + num_vision_tokens) {
        // Vision tokens: copy from vision_embeds[token_idx - image_token_pos]
        src_ptr = vision_embeds;
        src_offset = (token_idx - image_token_pos) * dim;
    } else {
        // After vision tokens: copy from text_embeds[token_idx - num_vision_tokens + 1]
        // Because we removed the image placeholder token
        src_ptr = text_embeds;
        src_offset = (token_idx - num_vision_tokens + 1) * dim;
    }
    
    half* dst_ptr = output + token_idx * dim;
    
    // Vectorized copy using half2
    const half2* src_h2 = reinterpret_cast<const half2*>(src_ptr + src_offset);
    half2* dst_h2 = reinterpret_cast<half2*>(dst_ptr);
    const int dim_h2 = dim / 2;
    
    for (int i = tid; i < dim_h2; i += blockDim.x) {
        dst_h2[i] = src_h2[i];
    }
    
    // Handle odd dimension
    if (dim % 2 == 1 && tid == 0) {
        dst_ptr[dim - 1] = src_ptr[src_offset + dim - 1];
    }
}

void fused_multimodal_embed_cu(
    const tensor::Tensor& text_embeds,
    const tensor::Tensor& vision_embeds,
    tensor::Tensor& output,
    int image_token_pos,
    int num_vision_tokens,
    int text_seq_len,
    int dim,
    cudaStream_t stream
) {
    int total_seq_len = text_seq_len - 1 + num_vision_tokens;  // -1 for image placeholder
    
    dim3 grid(total_seq_len);
    dim3 block(min(256, (dim + 1) / 2));  // Each thread handles 2 elements via half2
    
    fused_multimodal_embed_fp16_kernel<<<grid, block, 0, stream>>>(
        text_embeds.ptr<half>(),
        vision_embeds.ptr<half>(),
        output.ptr<half>(),
        image_token_pos,
        num_vision_tokens,
        text_seq_len,
        dim,
        total_seq_len
    );
}

// ============================================================================
// Fused KV Cache Update Kernel
// ============================================================================

/**
 * Fused kernel to update both K and V caches in a single launch
 * Replaces 2 separate cudaMemcpyAsync calls per layer
 */
__global__ void fused_kv_cache_update_fp16_kernel(
    const half* __restrict__ key_out,     // [seq_len, kv_dim]
    const half* __restrict__ value_out,   // [seq_len, kv_dim]
    half* __restrict__ key_cache,         // [layers, max_seq_len, kv_dim]
    half* __restrict__ value_cache,       // [layers, max_seq_len, kv_dim]
    const int layer_offset,               // layer_idx * max_seq_len * kv_dim
    const int start_pos,
    const int seq_len,
    const int kv_dim
) {
    // Grid: (seq_len, 2) - blockIdx.y: 0=key, 1=value
    // Block: handles kv_dim elements
    const int token_idx = blockIdx.x;
    const int is_value = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (token_idx >= seq_len) return;
    
    const half* src;
    half* dst;
    
    if (is_value == 0) {
        src = key_out + token_idx * kv_dim;
        dst = key_cache + layer_offset + (start_pos + token_idx) * kv_dim;
    } else {
        src = value_out + token_idx * kv_dim;
        dst = value_cache + layer_offset + (start_pos + token_idx) * kv_dim;
    }
    
    // Vectorized copy using half2
    const half2* src_h2 = reinterpret_cast<const half2*>(src);
    half2* dst_h2 = reinterpret_cast<half2*>(dst);
    const int kv_dim_h2 = kv_dim / 2;
    
    for (int i = tid; i < kv_dim_h2; i += blockDim.x) {
        dst_h2[i] = src_h2[i];
    }
    
    // Handle odd kv_dim
    if (kv_dim % 2 == 1 && tid == 0) {
        dst[kv_dim - 1] = src[kv_dim - 1];
    }
}

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
) {
    int layer_offset = layer_idx * max_seq_len * kv_dim;
    
    dim3 grid(seq_len, 2);  // 2 for key and value
    dim3 block(min(256, (kv_dim + 1) / 2));
    
    fused_kv_cache_update_fp16_kernel<<<grid, block, 0, stream>>>(
        key_out.ptr<half>(),
        value_out.ptr<half>(),
        key_cache.ptr<half>(),
        value_cache.ptr<half>(),
        layer_offset,
        start_pos,
        seq_len,
        kv_dim
    );
}

// ============================================================================
// GPU Patch Extraction Kernel
// ============================================================================
// Extracts patches from image tensor directly on GPU with 2x2 block interleaved order
// This eliminates the D2H and H2D copies in the original CPU implementation

/**
 * GPU kernel for patch extraction with 2x2 block interleaved order
 * 
 * Each thread block handles one output element in the patch tensor
 * Grid: (num_patches, patch_dim)
 * 
 * Input layout: image[C, H, W] in CHW format
 * Output layout: patches[num_patches, patch_dim] 
 *   where patch_dim = C * temporal_patch_size * patch_size * patch_size
 *
 * Patch ordering (2x2 block interleaved):
 *   grid_h blocks × grid_w blocks, each containing 2×2 patches
 *   Within each block: (0,0), (0,1), (1,0), (1,1)
 */
__global__ void extract_patches_fp16_kernel(
    const half* __restrict__ image,  // [C, H, W]
    half* __restrict__ patches,       // [num_patches, patch_dim]
    const int channels,
    const int height,
    const int width,
    const int patch_size,
    const int temporal_patch_size,
    const int grid_h,
    const int grid_w,
    const int patch_dim
) {
    // Each block handles one patch
    const int patch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const int spatial_merge_size = 2;
    const int w_blocks = grid_w / spatial_merge_size;
    
    // Convert patch_idx to 2x2 block coordinates
    // patch_idx = (bh * w_blocks + bw) * 4 + local_idx
    // where local_idx = local_h * 2 + local_w
    const int block_idx = patch_idx / 4;
    const int local_idx = patch_idx % 4;
    const int bh = block_idx / w_blocks;
    const int bw = block_idx % w_blocks;
    const int local_h = local_idx / 2;
    const int local_w = local_idx % 2;
    
    // Patch position in grid
    const int ph = bh * spatial_merge_size + local_h;
    const int pw = bw * spatial_merge_size + local_w;
    
    // Output base pointer for this patch
    half* patch_out = patches + patch_idx * patch_dim;
    
    // Each thread handles multiple elements in the patch
    // patch_dim = C * T * patch_size * patch_size
    for (int elem = tid; elem < patch_dim; elem += blockDim.x) {
        // Decode element index: patch_offset = ((c * T + t) * patch_size + h) * patch_size + w
        const int w = elem % patch_size;
        const int h = (elem / patch_size) % patch_size;
        // Note: t (temporal) is not used because we repeat the same frame for single images
        // const int t = (elem / (patch_size * patch_size)) % temporal_patch_size;
        const int c = elem / (patch_size * patch_size * temporal_patch_size);
        
        // Image coordinates
        const int img_h = ph * patch_size + h;
        const int img_w = pw * patch_size + w;
        
        // Read from image (CHW format)
        const int img_idx = c * height * width + img_h * width + img_w;
        
        // For temporal dimension, we repeat the same frame (single image)
        // so t doesn't affect the input index
        patch_out[elem] = image[img_idx];
    }
}

void extract_patches_cu(
    const tensor::Tensor& image,
    tensor::Tensor& patches,
    int channels,
    int height,
    int width,
    int patch_size,
    int temporal_patch_size,
    cudaStream_t stream
) {
    const int grid_h = height / patch_size;
    const int grid_w = width / patch_size;
    const int num_patches = grid_h * grid_w;
    const int patch_dim = channels * temporal_patch_size * patch_size * patch_size;
    
    // Each block handles one patch, with 256 threads per block
    dim3 grid(num_patches);
    dim3 block(256);
    
    extract_patches_fp16_kernel<<<grid, block, 0, stream>>>(
        image.ptr<half>(),
        patches.ptr<half>(),
        channels,
        height,
        width,
        patch_size,
        temporal_patch_size,
        grid_h,
        grid_w,
        patch_dim
    );
}

}  // namespace kernel
