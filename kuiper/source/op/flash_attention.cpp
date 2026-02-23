#include "op/flash_attention.h"
#include <cuda_runtime_api.h>
#include "kernels/cuda/flash_attention_kernel.cuh"
#include "kernels/cuda/paged_attention_kernel.cuh"
#include "kernels/cuda/mha_kernel.cuh"
#include "kernels/kernels_interface.h"

namespace op {

// ==================== FlashAttentionDecodeLayer ====================

FlashAttentionDecodeLayer::FlashAttentionDecodeLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerMHA, "FlashAttentionDecode") {
  reset_input_size(5);  // query, mha_output, key_cache, value_cache, pos
  reset_output_size(0); // output is in-place via input[1]
}

FlashAttentionDecodeLayer::FlashAttentionDecodeLayer(
    base::DeviceType device_type, int32_t head_num, int32_t kv_head_num,
    int32_t head_size, int32_t kv_mul, int32_t seq_len, int32_t kv_dim, bool use_fp16)
    : Layer(device_type, LayerType::kLayerMHA, "FlashAttentionDecode"),
      head_num_(head_num),
      kv_head_num_(kv_head_num),
      head_size_(head_size),
      kv_mul_(kv_mul),
      seq_len_(seq_len),
      kv_dim_(kv_dim),
      use_fp16_(use_fp16) {
  reset_input_size(5);  // query, mha_output, key_cache, value_cache, pos
  reset_output_size(0); // output is in-place via input[1]
}

base::Status FlashAttentionDecodeLayer::check() const {
  if (head_num_ <= 0 || kv_head_num_ <= 0 || head_size_ <= 0) {
    return base::error::InvalidArgument("Invalid attention parameters");
  }
  return base::error::Success();
}

base::Status FlashAttentionDecodeLayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }

  const tensor::Tensor& query = get_input(0);
  const tensor::Tensor& mha_output = get_input(1);
  const tensor::Tensor& key_cache = get_input(2);
  const tensor::Tensor& value_cache = get_input(3);

  // ===================== Paged Attention Path =====================
  if (paged_mode_) {
    if (use_fp16_) {
      if (use_gpu_pos_) {
        const tensor::Tensor& pos_tensor = get_input(4);
        kernel::paged_flash_attention_decode_fp16_gpu_pos_cu(
            pos_tensor.ptr<int32_t>(),
            head_num_, kv_head_num_, head_size_, kv_mul_,
            layer_index_, kv_dim_, page_size_, max_blocks_per_seq_,
            query, mha_output, key_pool_, value_pool_, block_table_,
            cuda_config_.get());
      } else {
        kernel::paged_flash_attention_decode_fp16_cu(
            pos_, head_num_, kv_head_num_, head_size_, kv_mul_,
            layer_index_, kv_dim_, page_size_, max_blocks_per_seq_,
            query, mha_output, key_pool_, value_pool_, block_table_,
            cuda_config_.get());
      }
    } else {
      kernel::paged_flash_attention_decode_cu(
          pos_, head_num_, kv_head_num_, head_size_, kv_mul_,
          layer_index_, kv_dim_, page_size_, max_blocks_per_seq_,
          query, mha_output, key_pool_, value_pool_, block_table_,
          cuda_config_.get());
    }
    return base::error::Success();
  }

  // ===================== Contiguous Attention Path =====================
  if (use_fp16_) {
    if (use_gpu_pos_) {
      const tensor::Tensor& pos_tensor = get_input(4);
      if (attention_type_ == base::AttentionType::kAttentionFlash2) {
        kernel::flash_attention2_decode_fp16_gpu_pos_cu(
            pos_tensor.ptr<int32_t>(),
            head_num_, kv_head_num_, head_size_, kv_mul_,
            layer_index_, seq_len_, kv_dim_,
            query, mha_output, key_cache, value_cache,
            cuda_config_.get());
      } else {
        kernel::flash_attention_decode_fp16_gpu_pos_cu(
            pos_tensor.ptr<int32_t>(),  // GPU memory pointer
            head_num_, kv_head_num_, head_size_, kv_mul_,
            layer_index_, seq_len_, kv_dim_,
            query, mha_output, key_cache, value_cache,
            cuda_config_.get());
      }
    } else {
      if (attention_type_ == base::AttentionType::kAttentionFlash2) {
        kernel::flash_attention2_decode_fp16_cu(
            pos_, head_num_, kv_head_num_, head_size_, kv_mul_,
            layer_index_, seq_len_, kv_dim_,
            query, mha_output, key_cache, value_cache,
            cuda_config_.get());
      } else {
        kernel::flash_attention_decode_fp16_cu(
            pos_, head_num_, kv_head_num_, head_size_, kv_mul_,
            layer_index_, seq_len_, kv_dim_,
            query, mha_output, key_cache, value_cache,
            cuda_config_.get());
      }
    }
  } else {
    // FP32 path: dispatch based on attention_type_
    if (attention_type_ == base::AttentionType::kAttentionFlash2) {
      kernel::flash_attention2_decode_cu(
          pos_, head_num_, kv_head_num_, head_size_, kv_mul_,
          layer_index_, seq_len_, kv_dim_,
          query, mha_output, key_cache, value_cache,
          cuda_config_.get());
    } else if (attention_type_ == base::AttentionType::kAttentionFlash1) {
      kernel::flash_attention_decode_cu(
          pos_, head_num_, kv_head_num_, head_size_, kv_mul_,
          layer_index_, seq_len_, kv_dim_,
          query, mha_output, key_cache, value_cache,
          cuda_config_.get());
    } else {
      // MHA fallback for FP32
      kernel::get_mha_kernel(device_type_)(
          pos_, head_num_, layer_index_, seq_len_,
          kv_dim_, kv_mul_, head_size_,
          mha_output, query, tensor::Tensor(), key_cache, value_cache,
          device_type_, cuda_config_.get());
    }
  }

  return base::error::Success();
}

base::Status FlashAttentionDecodeLayer::forward(int32_t pos, int32_t head_num, int32_t kv_head_num,
                                                 int32_t head_size, int32_t kv_mul, int32_t layer_idx,
                                                 int32_t seq_len, int32_t kv_dim,
                                                 const tensor::Tensor& query, const tensor::Tensor& mha_output,
                                                 const tensor::Tensor& key_cache, const tensor::Tensor& val_cache) {
  // Paged attention path (direct overload)
  if (paged_mode_) {
    if (query.data_type() == base::DataType::kDataTypeFp16) {
      kernel::paged_flash_attention_decode_fp16_cu(
          pos, head_num, kv_head_num, head_size, kv_mul, layer_idx,
          kv_dim, page_size_, max_blocks_per_seq_,
          query, mha_output, key_pool_, value_pool_, block_table_,
          cuda_config_.get());
    } else {
      kernel::paged_flash_attention_decode_cu(
          pos, head_num, kv_head_num, head_size, kv_mul, layer_idx,
          kv_dim, page_size_, max_blocks_per_seq_,
          query, mha_output, key_pool_, value_pool_, block_table_,
          cuda_config_.get());
    }
    return base::error::Success();
  }
  if (attention_type_ == base::AttentionType::kAttentionFlash2) {
    kernel::flash_attention2_decode_fp16_cu(
        pos, head_num, kv_head_num, head_size, kv_mul,
        layer_idx, seq_len, kv_dim,
        query, mha_output, key_cache, val_cache,
        cuda_config_.get());
  } else {
    kernel::flash_attention_decode_fp16_cu(
        pos, head_num, kv_head_num, head_size, kv_mul,
        layer_idx, seq_len, kv_dim,
        query, mha_output, key_cache, val_cache,
        cuda_config_.get());
  }
  return base::error::Success();
}

// ==================== FlashAttentionPrefillLayer ====================

FlashAttentionPrefillLayer::FlashAttentionPrefillLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerMHA, "FlashAttentionPrefill") {
  reset_input_size(4);  // query, output, key_cache, value_cache
  reset_output_size(0); // output is in-place via input[1]
}

FlashAttentionPrefillLayer::FlashAttentionPrefillLayer(
    base::DeviceType device_type, int32_t head_num, int32_t kv_head_num,
    int32_t head_size, int32_t seq_len, bool use_fp16)
    : Layer(device_type, LayerType::kLayerMHA, "FlashAttentionPrefill"),
      head_num_(head_num),
      kv_head_num_(kv_head_num),
      head_size_(head_size),
      max_seq_len_(seq_len),
      use_fp16_(use_fp16) {
  reset_input_size(4);  // query, output, key_cache, value_cache
  reset_output_size(0); // output is in-place via input[1]
}

base::Status FlashAttentionPrefillLayer::check() const {
  if (head_num_ <= 0 || kv_head_num_ <= 0 || head_size_ <= 0) {
    return base::error::InvalidArgument("Invalid attention parameters");
  }
  return base::error::Success();
}

base::Status FlashAttentionPrefillLayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }

  const tensor::Tensor& query = get_input(0);
  const tensor::Tensor& output = get_input(1);
  const tensor::Tensor& key_cache = get_input(2);
  const tensor::Tensor& value_cache = get_input(3);
  
  int32_t kv_mul = head_num_ / kv_head_num_;
  int32_t kv_dim = kv_head_num_ * head_size_;

  // ===================== Paged Attention Path =====================
  if (paged_mode_) {
    if (use_fp16_) {
      kernel::paged_flash_attention_prefill_fp16_cu(
          start_pos_, cur_seq_len_, head_num_, kv_head_num_, head_size_,
          kv_mul, layer_idx_, kv_dim, page_size_, max_blocks_per_seq_,
          query, output, key_pool_, value_pool_, block_table_,
          cuda_config_.get());
    } else {
      kernel::paged_flash_attention_prefill_cu(
          start_pos_, cur_seq_len_, head_num_, kv_head_num_, head_size_,
          kv_mul, layer_idx_, kv_dim, page_size_, max_blocks_per_seq_,
          query, output, key_pool_, value_pool_, block_table_,
          cuda_config_.get());
    }
    return base::error::Success();
  }

  // ===================== Contiguous Attention Path =====================
  if (use_fp16_) {
    if (attention_type_ == base::AttentionType::kAttentionFlash2) {
      kernel::flash_attention2_prefill_fp16_cu(
          start_pos_, cur_seq_len_, head_num_, kv_head_num_, head_size_,
          kv_mul, layer_idx_, max_seq_len_, kv_dim,
          query, output, key_cache, value_cache,
          cuda_config_.get());
    } else {
      kernel::flash_attention_prefill_fp16_cu(
          start_pos_, cur_seq_len_, head_num_, kv_head_num_, head_size_,
          kv_mul, layer_idx_, max_seq_len_, kv_dim,
          query, output, key_cache, value_cache,
          cuda_config_.get());
    }
  } else {
    if (attention_type_ == base::AttentionType::kAttentionFlash2) {
      kernel::flash_attention2_prefill_cu(
          start_pos_, cur_seq_len_, head_num_, kv_head_num_, head_size_,
          kv_mul, layer_idx_, max_seq_len_, kv_dim,
          query, output, key_cache, value_cache,
          cuda_config_.get());
    } else {
      kernel::flash_attention_prefill_cu(
          start_pos_, cur_seq_len_, head_num_, kv_head_num_, head_size_,
          kv_mul, layer_idx_, max_seq_len_, kv_dim,
          query, output, key_cache, value_cache,
          cuda_config_.get());
    }
  }

  return base::error::Success();
}

base::Status FlashAttentionPrefillLayer::forward(int32_t start_pos, int32_t seq_len, 
                                                  int32_t head_num, int32_t kv_head_num,
                                                  int32_t head_size, int32_t kv_mul, int32_t layer_idx,
                                                  int32_t max_seq_len, int32_t kv_dim,
                                                  const tensor::Tensor& query, const tensor::Tensor& output,
                                                  const tensor::Tensor& key_cache, const tensor::Tensor& val_cache) {
  // Paged attention path (direct overload)
  if (paged_mode_) {
    if (query.data_type() == base::DataType::kDataTypeFp16) {
      kernel::paged_flash_attention_prefill_fp16_cu(
          start_pos, seq_len, head_num, kv_head_num, head_size,
          kv_mul, layer_idx, kv_dim, page_size_, max_blocks_per_seq_,
          query, output, key_pool_, value_pool_, block_table_,
          cuda_config_.get());
    } else {
      kernel::paged_flash_attention_prefill_cu(
          start_pos, seq_len, head_num, kv_head_num, head_size,
          kv_mul, layer_idx, kv_dim, page_size_, max_blocks_per_seq_,
          query, output, key_pool_, value_pool_, block_table_,
          cuda_config_.get());
    }
    return base::error::Success();
  }
  if (attention_type_ == base::AttentionType::kAttentionFlash2) {
    kernel::flash_attention2_prefill_fp16_cu(
        start_pos, seq_len, head_num, kv_head_num, head_size,
        kv_mul, layer_idx, max_seq_len, kv_dim,
        query, const_cast<tensor::Tensor&>(output), key_cache, val_cache,
        cuda_config_.get());
  } else {
    kernel::flash_attention_prefill_fp16_cu(
        start_pos, seq_len, head_num, kv_head_num, head_size,
        kv_mul, layer_idx, max_seq_len, kv_dim,
        query, const_cast<tensor::Tensor&>(output), key_cache, val_cache,
        cuda_config_.get());
  }
  return base::error::Success();
}

}  // namespace op
