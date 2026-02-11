#ifndef ROPE_KERNEL_CU_CUH
#define ROPE_KERNEL_CU_CUH
#include "tensor/tensor.h"
namespace kernel {
void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                    const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache, void* stream);

// RoPE kernel with position read from GPU memory (for CUDA Graph optimization)
void rope_kernel_cu_gpu_pos(int32_t dim, int32_t kv_dim, int32_t head_size, 
                            const tensor::Tensor& input_q, const tensor::Tensor& input_k, 
                            const int32_t* pos_ptr, const tensor::Tensor& sin_cache, 
                            const tensor::Tensor& cos_cache, void* stream);

// FP16 RoPE kernel with position read from GPU memory (for CUDA Graph + FP16 optimization)
void rope_kernel_cu_fp16_gpu_pos(int32_t dim, int32_t kv_dim, int32_t head_size, 
                                  const tensor::Tensor& input_q, const tensor::Tensor& input_k, 
                                  const int32_t* pos_ptr, const tensor::Tensor& sin_cache, 
                                  const tensor::Tensor& cos_cache, void* stream);

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, cudaStream_t stream);

// Batched RoPE for prefill phase
void batched_rope_kernel_cu(int32_t start_pos, int32_t seq_len, int32_t dim, int32_t kv_dim,
                            int32_t head_size, const tensor::Tensor& input_q,
                            const tensor::Tensor& input_k, const tensor::Tensor& sin_cache,
                            const tensor::Tensor& cos_cache, void* stream);

// Pure FP16 RoPE kernel
void rope_kernel_cu_pure_fp16(int32_t dim, int32_t kv_dim, int32_t head_size, 
                               const tensor::Tensor& input_q, const tensor::Tensor& input_k, 
                               int32_t pos, const tensor::Tensor& sin_cache, 
                               const tensor::Tensor& cos_cache, void* stream);

// Batched pure FP16 RoPE for prefill phase
void batched_rope_kernel_cu_pure_fp16(int32_t start_pos, int32_t seq_len, int32_t dim, int32_t kv_dim,
                                       int32_t head_size, const tensor::Tensor& input_q,
                                       const tensor::Tensor& input_k, const tensor::Tensor& sin_cache,
                                       const tensor::Tensor& cos_cache, void* stream);

// ==================== M-RoPE (Multimodal RoPE) ====================
// M-RoPE uses 3D position encoding: (temporal, height, width)
// mrope_section = [24, 20, 20] for Qwen3-VL (head_size=128)
// - Dimensions [0, 48): use temporal position (t)
// - Dimensions [48, 88): use height position (h)  
// - Dimensions [88, 128): use width position (w)

/**
 * @brief M-RoPE kernel for single token with 3D position encoding (FP16)
 * 
 * @param pos_t Temporal position
 * @param pos_h Height position
 * @param pos_w Width position
 * @param dim Total Q dimension
 * @param kv_dim Total K dimension
 * @param head_size Size per head (128)
 * @param section0 First section pairs (24 for temporal)
 * @param section1 Second section pairs (20 for height)
 * @param section2 Third section pairs (20 for width)
 */
void mrope_kernel_cu_fp16(
    int32_t pos_t, int32_t pos_h, int32_t pos_w,
    int32_t dim, int32_t kv_dim, int32_t head_size,
    int32_t section0, int32_t section1, int32_t section2,
    const tensor::Tensor& input_q, const tensor::Tensor& input_k,
    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
    void* stream);

/**
 * @brief Batched M-RoPE kernel for multiple tokens with 3D position encoding (FP16)
 * Each token has its own (pos_t, pos_h, pos_w) positions.
 * 
 * @param seq_len Number of tokens
 * @param dim Total Q dimension
 * @param kv_dim Total K dimension
 * @param head_size Size per head (128)
 * @param section0 First section pairs (24 for temporal)
 * @param section1 Second section pairs (20 for height)
 * @param section2 Third section pairs (20 for width)
 * @param pos_t_arr Array of temporal positions [seq_len] on GPU
 * @param pos_h_arr Array of height positions [seq_len] on GPU
 * @param pos_w_arr Array of width positions [seq_len] on GPU
 */
void batched_mrope_kernel_cu_fp16(
    int32_t seq_len, int32_t dim, int32_t kv_dim, int32_t head_size,
    int32_t section0, int32_t section1, int32_t section2,
    const int32_t* pos_t_arr, const int32_t* pos_h_arr, const int32_t* pos_w_arr,
    const tensor::Tensor& input_q, const tensor::Tensor& input_k,
    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
    void* stream);

/**
 * @brief M-RoPE kernel with GPU-resident position for CUDA Graph compatibility (FP16)
 * For decode phase where all 3 positions are identical (text_pos)
 * 
 * @param pos_gpu GPU pointer to position value
 * @param dim Total Q dimension
 * @param kv_dim Total K dimension
 * @param head_size Size per head (128)
 * @param section0 First section pairs (24 for temporal)
 * @param section1 Second section pairs (20 for height)
 * @param section2 Third section pairs (20 for width)
 */
void mrope_kernel_cu_fp16_gpu_pos(
    const int32_t* pos_gpu,
    int32_t dim, int32_t kv_dim, int32_t head_size,
    int32_t section0, int32_t section1, int32_t section2,
    const tensor::Tensor& input_q, const tensor::Tensor& input_k,
    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
    void* stream);

}  // namespace kernel
#endif  // ROPE_KERNEL_CU_CUH
