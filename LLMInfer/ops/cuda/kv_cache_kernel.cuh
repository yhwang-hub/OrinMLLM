#ifndef KUIPER_SOURCE_OP_KERNELS_CUDA_KV_CACHE_KERNEL_CUH_
#define KUIPER_SOURCE_OP_KERNELS_CUDA_KV_CACHE_KERNEL_CUH_
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace kernel {

/**
 * @brief Copy key/value from temporary buffer to KV cache at specified position (FP32)
 * 
 * This kernel is designed for CUDA Graph compatibility:
 * - Source address (temp buffer) is fixed
 * - Destination base address (KV cache) is fixed  
 * - Position is read from GPU memory (pos_tensor), allowing value updates without graph recapture
 * 
 * @param kv_cache     Base pointer of KV cache [layer_num, seq_len, kv_dim]
 * @param src          Source temporary buffer [kv_dim]
 * @param pos          Pointer to position value in GPU memory
 * @param kv_dim       Dimension of key/value
 * @param layer_idx    Current layer index
 * @param seq_len      Maximum sequence length
 * @param stream       CUDA stream
 */
void copy_to_kv_cache_kernel(float* kv_cache, const float* src, const int32_t* pos,
                             int32_t kv_dim, int32_t layer_idx, int32_t seq_len,
                             cudaStream_t stream);

/**
 * @brief Copy key/value from temporary buffer to KV cache at specified position (FP16)
 */
void copy_to_kv_cache_kernel_fp16(half* kv_cache, const half* src, const int32_t* pos,
                                   int32_t kv_dim, int32_t layer_idx, int32_t seq_len,
                                   cudaStream_t stream);

}  // namespace kernel

#endif  // KUIPER_SOURCE_OP_KERNELS_CUDA_KV_CACHE_KERNEL_CUH_
