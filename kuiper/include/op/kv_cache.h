#ifndef KUIPER_INCLUDE_OP_KV_CACHE_H_
#define KUIPER_INCLUDE_OP_KV_CACHE_H_
#include "layer.h"
namespace op {

/**
 * @brief KV Cache Update Layer
 * 
 * Copies key/value tensors to KV cache at specified position.
 * Supports both CPU and GPU position tensors for CUDA Graph compatibility.
 * 
 * Input:
 *   - input[0]: key/value tensor to copy [kv_dim]
 *   - input[1]: kv_cache tensor [layer_num, seq_len, kv_dim]
 *   - input[2]: position tensor [1] (can be GPU or CPU memory)
 * 
 * Output: None (modifies kv_cache in-place)
 */
class KVCacheLayer : public Layer {
 public:
  explicit KVCacheLayer(base::DeviceType device_type);
  KVCacheLayer(base::DeviceType device_type, int32_t kv_dim, int32_t seq_len, bool use_fp16 = false);

  base::Status check() const override;
  base::Status forward() override;

  void set_layer_index(int32_t layer_index) { layer_index_ = layer_index; }
  void set_use_gpu_pos(bool use_gpu_pos) { use_gpu_pos_ = use_gpu_pos; }
  void set_dims(int32_t kv_dim, int32_t seq_len) {
    kv_dim_ = kv_dim;
    seq_len_ = seq_len;
  }
  void set_use_fp16(bool use_fp16) { use_fp16_ = use_fp16; }

 private:
  int32_t kv_dim_ = 0;
  int32_t seq_len_ = 0;
  int32_t layer_index_ = 0;
  bool use_fp16_ = false;
  bool use_gpu_pos_ = false;
};

}  // namespace op
#endif  // KUIPER_INCLUDE_OP_KV_CACHE_H_
