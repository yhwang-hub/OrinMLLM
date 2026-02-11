#ifndef KUIPER_INCLUDE_OP_FLASH_ATTENTION_H_
#define KUIPER_INCLUDE_OP_FLASH_ATTENTION_H_
#include "layer.h"
namespace op {

/**
 * @brief Flash Attention Layer for decode phase (single token)
 * 
 * Input:
 *   - input[0]: query tensor [dim]
 *   - input[1]: mha_output tensor [dim]
 *   - input[2]: key_cache tensor [layer_num, seq_len, kv_dim]
 *   - input[3]: value_cache tensor [layer_num, seq_len, kv_dim]
 *   - input[4]: pos tensor (GPU or CPU) [1] - only needed when use_gpu_pos=true
 * Output:
 *   - Modifies input[1] (mha_output) in-place
 */
class FlashAttentionDecodeLayer : public Layer {
 public:
  explicit FlashAttentionDecodeLayer(base::DeviceType device_type);
  FlashAttentionDecodeLayer(base::DeviceType device_type, int32_t head_num, int32_t kv_head_num,
                             int32_t head_size, int32_t kv_mul, int32_t seq_len, int32_t kv_dim,
                             bool use_fp16 = false);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward for Flash Attention decode
  base::Status forward(int32_t pos, int32_t head_num, int32_t kv_head_num,
                       int32_t head_size, int32_t kv_mul, int32_t layer_idx,
                       int32_t seq_len, int32_t kv_dim,
                       const tensor::Tensor& query, const tensor::Tensor& mha_output,
                       const tensor::Tensor& key_cache, const tensor::Tensor& val_cache);

  void set_layer_index(int32_t layer_index) { layer_index_ = layer_index; }
  void set_pos(int32_t pos) { pos_ = pos; }
  void set_use_gpu_pos(bool use_gpu_pos) { use_gpu_pos_ = use_gpu_pos; }
  void set_dims(int32_t head_num, int32_t kv_head_num, int32_t head_size, 
                int32_t kv_mul, int32_t seq_len, int32_t kv_dim) {
    head_num_ = head_num;
    kv_head_num_ = kv_head_num;
    head_size_ = head_size;
    kv_mul_ = kv_mul;
    seq_len_ = seq_len;
    kv_dim_ = kv_dim;
  }
  void set_use_fp16(bool use_fp16) { use_fp16_ = use_fp16; }

 private:
  int32_t head_num_ = 0;
  int32_t kv_head_num_ = 0;
  int32_t head_size_ = 0;
  int32_t kv_mul_ = 0;
  int32_t seq_len_ = 0;
  int32_t kv_dim_ = 0;
  int32_t layer_index_ = 0;
  int32_t pos_ = 0;
  bool use_fp16_ = false;
  bool use_gpu_pos_ = false;
};

/**
 * @brief Flash Attention Layer for prefill phase (multiple tokens)
 * 
 * Input:
 *   - input[0]: query tensor [seq_len, dim]
 *   - input[1]: key tensor [seq_len, kv_dim]
 *   - input[2]: value tensor [seq_len, kv_dim]
 * Output:
 *   - output[0]: attention output [seq_len, dim]
 */
class FlashAttentionPrefillLayer : public Layer {
 public:
  explicit FlashAttentionPrefillLayer(base::DeviceType device_type);
  FlashAttentionPrefillLayer(base::DeviceType device_type, int32_t head_num, int32_t kv_head_num,
                              int32_t head_size, int32_t seq_len, bool use_fp16 = false);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward for Flash Attention prefill
  base::Status forward(int32_t start_pos, int32_t seq_len, int32_t head_num, int32_t kv_head_num,
                       int32_t head_size, int32_t kv_mul, int32_t layer_idx,
                       int32_t max_seq_len, int32_t kv_dim,
                       const tensor::Tensor& query, const tensor::Tensor& output,
                       const tensor::Tensor& key_cache, const tensor::Tensor& val_cache);

  void set_cur_seq_len(int32_t seq_len) { cur_seq_len_ = seq_len; }
  void set_start_pos(int32_t start_pos) { start_pos_ = start_pos; }
  void set_layer_index(int32_t layer_idx) { layer_idx_ = layer_idx; }
  void set_dims(int32_t head_num, int32_t kv_head_num, int32_t head_size, int32_t max_seq_len) {
    head_num_ = head_num;
    kv_head_num_ = kv_head_num;
    head_size_ = head_size;
    max_seq_len_ = max_seq_len;
  }
  void set_use_fp16(bool use_fp16) { use_fp16_ = use_fp16; }

 private:
  int32_t head_num_ = 0;
  int32_t kv_head_num_ = 0;
  int32_t head_size_ = 0;
  int32_t max_seq_len_ = 0;
  int32_t cur_seq_len_ = 0;
  int32_t start_pos_ = 0;
  int32_t layer_idx_ = 0;
  bool use_fp16_ = false;
};

}  // namespace op
#endif  // KUIPER_INCLUDE_OP_FLASH_ATTENTION_H_
