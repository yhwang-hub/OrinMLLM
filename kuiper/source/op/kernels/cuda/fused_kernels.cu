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
 * Optimized with float4 (128-bit) vectorized copies — 8 halfs per thread per iteration,
 * giving 4x throughput vs the previous half2 vectorization.
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
    
    // Vectorized copy using float4 (8 halfs = 16 bytes per access)
    const float4* src_f4 = reinterpret_cast<const float4*>(src_ptr + src_offset);
    float4* dst_f4 = reinterpret_cast<float4*>(dst_ptr);
    const int dim_f4 = dim / 8;  // 4096/8 = 512 iterations total
    
    for (int i = tid; i < dim_f4; i += blockDim.x) {
        dst_f4[i] = src_f4[i];
    }
    
    // Handle remainder (dim not divisible by 8)
    const int remainder_start = dim_f4 * 8;
    if (tid == 0) {
        for (int i = remainder_start; i < dim; i++) {
            dst_ptr[i] = src_ptr[src_offset + i];
        }
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
    dim3 block(256);
    
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

// ============================================================================
// Fused Normalize + Patches Kernel (from uint8 HWC to fp16 patches)
// ============================================================================

/**
 * Fused kernel that combines normalize_to_tensor + image_to_patches.
 * 
 * Directly reads uint8 pixels in HWC format, normalizes (pixel/255.0 - mean) / std,
 * converts to fp16, and writes in 2x2 block interleaved patch order.
 * 
 * This eliminates:
 * - CPU normalization loop
 * - CPU fp32->fp16 conversion  
 * - H2D copy of intermediate CHW fp16 tensor
 * - Separate extract_patches_cu kernel launch
 *
 * Input: uint8 pixels [H, W, 3] in HWC format on GPU
 * Output: patches [num_patches, patch_dim] in fp16, 2x2 block interleaved order
 */
__global__ void fused_normalize_patches_kernel(
    const unsigned char* __restrict__ pixels,  // [H, W, 3] HWC uint8 on GPU
    half* __restrict__ patches,                // [num_patches, patch_dim] fp16
    const int height,
    const int width,
    const int patch_size,
    const int temporal_patch_size,
    const int grid_h,
    const int grid_w,
    const int patch_dim,
    const float mean_r, const float mean_g, const float mean_b,
    const float std_r, const float std_g, const float std_b
) {
    // Each block handles one patch
    const int patch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const int spatial_merge_size = 2;
    const int w_blocks = grid_w / spatial_merge_size;
    
    // Convert patch_idx to 2x2 block coordinates
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
    
    const float mean[3] = {mean_r, mean_g, mean_b};
    const float std_val[3] = {std_r, std_g, std_b};
    
    // Each thread handles multiple elements in the patch
    // patch_dim = C * T * patch_size * patch_size
    for (int elem = tid; elem < patch_dim; elem += blockDim.x) {
        // Decode element index: ((c * T + t) * patch_size + h) * patch_size + w
        const int pw_local = elem % patch_size;
        const int ph_local = (elem / patch_size) % patch_size;
        const int c = elem / (patch_size * patch_size * temporal_patch_size);
        
        // Image coordinates
        const int img_h = ph * patch_size + ph_local;
        const int img_w = pw * patch_size + pw_local;
        
        // Read from HWC uint8 source
        const int hwc_idx = (img_h * width + img_w) * 3 + c;
        const float pixel_val = static_cast<float>(pixels[hwc_idx]);
        
        // IMPORTANT: Match CPU computation exactly to avoid numerical divergence.
        // CPU does: float pixel = pixel_val / 255.0f;
        //           normalized = (pixel - mean[c]) / std[c];
        //           fp16 = float_to_half(normalized);  // truncation (round-toward-zero)
        //
        // Use IEEE intrinsics (__fdiv_rn, __fsub_rn) to prevent NVCC from:
        //   - Transforming x/C into x*(1/C) (reciprocal approximation)
        //   - Fusing (a-b)*c or a*(1/c)+d into FMA instructions
        // These intrinsics guarantee exact IEEE 754 round-to-nearest-even semantics,
        // matching the CPU's float arithmetic behavior.
        float pixel_norm = __fdiv_rn(pixel_val, 255.0f);
        float centered   = __fsub_rn(pixel_norm, mean[c]);
        float normalized = __fdiv_rn(centered, std_val[c]);
        
        // Use round-toward-zero to match CPU float_to_half() truncation behavior
        patch_out[elem] = __float2half_rz(normalized);
    }
}

void fused_normalize_patches_cu(
    const unsigned char* pixels_gpu,  // uint8 HWC pixels on GPU
    half* patches_gpu,                // output patches on GPU
    int height,
    int width,
    int patch_size,
    int temporal_patch_size,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b,
    cudaStream_t stream
) {
    const int grid_h = height / patch_size;
    const int grid_w = width / patch_size;
    const int num_patches = grid_h * grid_w;
    const int channels = 3;
    const int patch_dim = channels * temporal_patch_size * patch_size * patch_size;
    
    // Each block handles one patch, with 256 threads per block
    dim3 grid(num_patches);
    dim3 block(256);
    
    fused_normalize_patches_kernel<<<grid, block, 0, stream>>>(
        pixels_gpu,
        patches_gpu,
        height,
        width,
        patch_size,
        temporal_patch_size,
        grid_h,
        grid_w,
        patch_dim,
        mean_r, mean_g, mean_b,
        std_r, std_g, std_b
    );
}

// ============================================================================
// Fused Resize + Normalize + Patches Kernel (Bicubic with anti-aliasing)
// ============================================================================

/**
 * Catmull-Rom bicubic interpolation weight (Keys cubic, a = -0.5)
 * Used for UPSCALING, matches stb_image_resize2 STBIR_FILTER_CATMULLROM.
 */
__device__ __forceinline__ float catmull_rom_weight(float x) {
    float ax = fabsf(x);
    if (ax < 1.0f) {
        return (1.5f * ax - 2.5f) * ax * ax + 1.0f;
    }
    if (ax < 2.0f) {
        return ((-0.5f * ax + 2.5f) * ax - 4.0f) * ax + 2.0f;
    }
    return 0.0f;
}

/**
 * Mitchell-Netravali filter (B=1/3, C=1/3)
 * Used for DOWNSCALING, matches stb_image_resize2 STBIR_FILTER_MITCHELL.
 * Computation mirrors stb's stbir__filter_mitchell_netravali() exactly.
 */
__device__ __forceinline__ float mitchell_weight(float x) {
    const float B = 1.0f / 3.0f;
    const float C = 1.0f / 3.0f;
    float ax = fabsf(x);
    if (ax < 1.0f) {
        return ((12.0f - 9.0f*B - 6.0f*C) * ax*ax*ax +
                (-18.0f + 12.0f*B + 6.0f*C) * ax*ax +
                (6.0f - 2.0f*B)) / 6.0f;
    }
    if (ax < 2.0f) {
        return ((-B - 6.0f*C) * ax*ax*ax +
                (6.0f*B + 30.0f*C) * ax*ax +
                (-12.0f*B - 48.0f*C) * ax +
                (8.0f*B + 24.0f*C)) / 6.0f;
    }
    return 0.0f;
}

/**
 * Bicubic weight selector: Mitchell for downscale, Catmull-Rom for upscale.
 * Matches stb_image_resize2 default filter behaviour (STBIR_FILTER_DEFAULT).
 */
__device__ __forceinline__ float bicubic_weight(float x, bool downscale) {
    return downscale ? mitchell_weight(x) : catmull_rom_weight(x);
}

/**
 * Fused kernel: bicubic resize + normalize + patch extraction.
 *
 * Three operations in a single kernel:
 * 1. Bicubic resize with stb-compatible filters (Mitchell down / Catmull-Rom up)
 * 2. Normalize: round to uint8, then (pixel/255 - mean) / std
 * 3. Patch extraction with 2x2 spatial merge block interleaved ordering
 *
 * Coordinate mapping matches stb_image_resize2:
 *   center = (out + 0.5) * (src_size / dst_size)
 *
 * Input:  original uint8 pixels [src_H, src_W, 3] HWC on GPU
 * Output: patches [num_patches, patch_dim] fp16
 */
__global__ void fused_resize_normalize_patches_kernel(
    const unsigned char* __restrict__ src_pixels,  // [src_H, src_W, 3]
    half* __restrict__ patches,                     // [num_patches, patch_dim]
    const int src_h, const int src_w,
    const int dst_h, const int dst_w,
    const int patch_size,
    const int temporal_patch_size,
    const int grid_h, const int grid_w,
    const int patch_dim,
    const float mean_r, const float mean_g, const float mean_b,
    const float std_r,  const float std_g,  const float std_b
) {
    const int patch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const int spatial_merge_size = 2;
    const int w_blocks = grid_w / spatial_merge_size;

    // 2x2 block interleaved patch ordering
    const int block_idx = patch_idx / 4;
    const int local_idx = patch_idx % 4;
    const int bh = block_idx / w_blocks;
    const int bw = block_idx % w_blocks;
    const int local_h = local_idx / 2;
    const int local_w = local_idx % 2;

    const int ph = bh * spatial_merge_size + local_h;
    const int pw = bw * spatial_merge_size + local_w;

    half* patch_out = patches + patch_idx * patch_dim;
    const float mean[3] = {mean_r, mean_g, mean_b};
    const float std_val[3] = {std_r, std_g, std_b};

    // Resize parameters — matches stb_image_resize2 default behaviour
    const float scale_y = (float)src_h / (float)dst_h;
    const float scale_x = (float)src_w / (float)dst_w;
    const bool  down_y  = (scale_y > 1.0f);
    const bool  down_x  = (scale_x > 1.0f);
    const float filter_scale_y = fmaxf(1.0f, scale_y);
    const float filter_scale_x = fmaxf(1.0f, scale_x);
    const float inv_fsy = 1.0f / filter_scale_y;
    const float inv_fsx = 1.0f / filter_scale_x;
    const float support_y = 2.0f * filter_scale_y;
    const float support_x = 2.0f * filter_scale_x;

    for (int elem = tid; elem < patch_dim; elem += blockDim.x) {
        const int pw_local = elem % patch_size;
        const int ph_local = (elem / patch_size) % patch_size;
        const int c = elem / (patch_size * patch_size * temporal_patch_size);

        // Destination pixel position in resized image
        const int dst_y = ph * patch_size + ph_local;
        const int dst_x = pw * patch_size + pw_local;

        // Map to source coordinates — stb_image_resize2 coordinate mapping:
        //   center = (out + 0.5) * (src_size / dst_size)
        const float center_y = (dst_y + 0.5f) * scale_y;
        const float center_x = (dst_x + 0.5f) * scale_x;

        // Source bounds — matching stb's STBIR_CALC_FLOOR
        const int y_start = max(0, (int)floorf(center_y - support_y));
        const int y_end   = min(src_h - 1, (int)floorf(center_y + support_y));
        const int x_start = max(0, (int)floorf(center_x - support_x));
        const int x_end   = min(src_w - 1, (int)floorf(center_x + support_x));

        float sum = 0.0f;
        float weight_sum = 0.0f;

        // 2D bicubic interpolation with anti-aliasing
        // Uses Mitchell for downscale, Catmull-Rom for upscale (per dimension)
        for (int sy = y_start; sy <= y_end; ++sy) {
            float dy = (sy - center_y) * inv_fsy;
            float wy = bicubic_weight(dy, down_y);
            if (wy == 0.0f) continue;

            const int row_offset = sy * src_w;
            for (int sx = x_start; sx <= x_end; ++sx) {
                float dx = (sx - center_x) * inv_fsx;
                float wx = bicubic_weight(dx, down_x);

                float w = wy * wx;
                sum += w * (float)__ldg(&src_pixels[(row_offset + sx) * 3 + c]);
                weight_sum += w;
            }
        }

        // Normalize weights and get pixel value
        float pixel_val = (weight_sum > 0.0f) ? (sum / weight_sum) : 0.0f;

        // Clamp to [0,255] and round to uint8 (matching PIL/stb output pipeline)
        pixel_val = fminf(fmaxf(roundf(pixel_val), 0.0f), 255.0f);

        // Normalize and convert to fp16.  Use IEEE intrinsics to prevent
        // FMA fusion and __float2half_rz to match CPU float_to_half truncation.
        float pixel_norm = __fdiv_rn(pixel_val, 255.0f);
        float centered   = __fsub_rn(pixel_norm, mean[c]);
        float normalized = __fdiv_rn(centered, std_val[c]);

        patch_out[elem] = __float2half_rz(normalized);
    }
}

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
) {
    const int grid_h = dst_h / patch_size;
    const int grid_w = dst_w / patch_size;
    const int num_patches = grid_h * grid_w;
    const int channels = 3;
    const int patch_dim = channels * temporal_patch_size * patch_size * patch_size;

    dim3 grid(num_patches);
    dim3 block(256);

    fused_resize_normalize_patches_kernel<<<grid, block, 0, stream>>>(
        src_pixels_gpu,
        patches_gpu,
        src_h, src_w,
        dst_h, dst_w,
        patch_size,
        temporal_patch_size,
        grid_h, grid_w,
        patch_dim,
        mean_r, mean_g, mean_b,
        std_r,  std_g,  std_b
    );
}

}  // namespace kernel
