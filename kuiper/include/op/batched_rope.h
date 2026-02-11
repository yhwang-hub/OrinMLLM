#ifndef KUIPER_INCLUDE_OP_BATCHED_ROPE_H_
#define KUIPER_INCLUDE_OP_BATCHED_ROPE_H_
#include "layer.h"
namespace op {

/**
 * @brief Extended RoPE Layer with GPU position support
 * 
 * Supports both CPU and GPU position tensors for CUDA Graph compatibility.
 * 
 * Input:
 *   - input[0]: query tensor [dim] or [batch, dim]
 *   - input[1]: key tensor [kv_dim] or [batch, kv_dim]
 *   - input[2]: position tensor [1] or [batch] (can be GPU memory)
 *   - input[3]: sin_cache tensor [seq_len, head_size/2]
 *   - input[4]: cos_cache tensor [seq_len, head_size/2]
 * Output:
 *   - Modifies input[0] and input[1] in-place
 */
class RoPEGpuPosLayer : public Layer {
 public:
  explicit RoPEGpuPosLayer(base::DeviceType device_type);
  RoPEGpuPosLayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, 
                  int32_t head_size, bool use_fp16 = false);

  base::Status check() const override;
  base::Status forward() override;

  void set_use_gpu_pos(bool use_gpu_pos) { use_gpu_pos_ = use_gpu_pos; }
  void set_dims(int32_t dim, int32_t kv_dim, int32_t head_size) {
    dim_ = dim;
    kv_dim_ = kv_dim;
    head_size_ = head_size;
  }
  void set_use_fp16(bool use_fp16) { use_fp16_ = use_fp16; }

 private:
  int32_t dim_ = 0;
  int32_t kv_dim_ = 0;
  int32_t head_size_ = 0;
  bool use_fp16_ = false;
  bool use_gpu_pos_ = false;
};

/**
 * @brief Batched RoPE Layer for prefill phase
 * 
 * Applies rotary position embedding to batched Q/K tensors.
 * 
 * Input:
 *   - input[0]: query tensor [batch, dim]
 *   - input[1]: key tensor [batch, kv_dim]
 *   - input[2]: sin_cache tensor [seq_len, head_size/2]
 *   - input[3]: cos_cache tensor [seq_len, head_size/2]
 * Output:
 *   - Modifies input[0] and input[1] in-place
 */
class BatchedRoPELayer : public Layer {
 public:
  explicit BatchedRoPELayer(base::DeviceType device_type);
  BatchedRoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, 
                   int32_t head_size, int32_t head_num, int32_t kv_head_num);

  base::Status check() const override;
  base::Status forward() override;

  void set_seq_len(int32_t seq_len) { seq_len_ = seq_len; }
  void set_start_pos(int32_t start_pos) { start_pos_ = start_pos; }
  void set_dims(int32_t dim, int32_t kv_dim, int32_t head_size, 
                int32_t head_num, int32_t kv_head_num) {
    dim_ = dim;
    kv_dim_ = kv_dim;
    head_size_ = head_size;
    head_num_ = head_num;
    kv_head_num_ = kv_head_num;
  }

 private:
  int32_t dim_ = 0;
  int32_t kv_dim_ = 0;
  int32_t head_size_ = 0;
  int32_t head_num_ = 0;
  int32_t kv_head_num_ = 0;
  int32_t seq_len_ = 0;
  int32_t start_pos_ = 0;
};

}  // namespace op
#endif  // KUIPER_INCLUDE_OP_BATCHED_ROPE_H_
