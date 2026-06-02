#ifndef MHA_KERNEL_H
#define MHA_KERNEL_H
namespace kernel {
void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                   base::DeviceType device_type, CudaConfig* config);

// MHA kernel with position read from GPU memory (for CUDA Graph optimization)
// Instead of passing pos as a value, it reads from pos_ptr in device memory
// This allows the kernel to be captured in CUDA Graph while pos changes
void mha_kernel_cu_gpu_pos(const int32_t* pos_ptr, int32_t head_num, int32_t layer_index, 
                           int32_t seq_len, int32_t kv_dim, int32_t kv_mul, int32_t head_size, 
                           const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor, 
                           const tensor::Tensor& score_tensor, const tensor::Tensor& key_cache_tensor, 
                           const tensor::Tensor& value_cache_tensor, base::DeviceType device_type, 
                           CudaConfig* config);

// Batched MHA for prefill phase
void batched_mha_kernel_cu(int32_t start_pos, int32_t seq_len, int32_t head_num, int32_t layer_index,
                           int32_t max_seq_len, int32_t dim, int32_t kv_dim, int32_t kv_mul, 
                           int32_t head_size, const tensor::Tensor& mha_out,
                           const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                           const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                           base::DeviceType device_type, CudaConfig* config);
}
#endif  // MHA_KERNEL_H
