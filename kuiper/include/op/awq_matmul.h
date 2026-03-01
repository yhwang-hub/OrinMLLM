//
// AWQ INT4 Quantized Matrix Multiplication Layer
//

#ifndef KUIPER_INCLUDE_OP_AWQ_MATMUL_H_
#define KUIPER_INCLUDE_OP_AWQ_MATMUL_H_

#include <base/cuda_config.h>
#include "layer.h"

namespace op {

/**
 * AWQ INT4 Quantized MatMul Layer
 * 
 * Stores weights in AWQ format:
 * - qweight: [in_features, out_features/8] INT32 (8 INT4 per INT32)
 * - qzeros: [in_features/group_size, out_features/8] INT32
 * - scales: [in_features/group_size, out_features] FP16
 * 
 * Input: [batch, in_features] FP16
 * Output: [batch, out_features] FP16
 */
class AWQMatmulLayer : public Layer {
 public:
  explicit AWQMatmulLayer(base::DeviceType device_type, 
                          int32_t in_features, 
                          int32_t out_features,
                          int32_t group_size = 128);

  base::Status check() const override;
  base::Status forward() override;
  
  // Direct forward with tensors (avoids set_input/set_output overhead)
  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& output);
  
  // Set AWQ weights from raw data
  void set_awq_weights(const void* qweight_ptr, 
                       const void* qzeros_ptr,
                       const void* scales_ptr,
                       base::DeviceType src_device);
  
  void to_cuda() override;
  
  // Getters
  int32_t in_features() const { return in_features_; }
  int32_t out_features() const { return out_features_; }
  int32_t group_size() const { return group_size_; }
  
  const tensor::Tensor& get_qweight() const { return qweight_; }
  const tensor::Tensor& get_qzeros() const { return qzeros_; }
  const tensor::Tensor& get_scales() const { return scales_; }

 private:
  int32_t in_features_ = 0;
  int32_t out_features_ = 0;
  int32_t group_size_ = 128;
  
  // AWQ quantized weights
  tensor::Tensor qweight_;  // [in_features, out_features/8] INT32
  tensor::Tensor qzeros_;   // [in_features/group_size, out_features/8] INT32
  tensor::Tensor scales_;   // [in_features/group_size, out_features] FP16
};

}  // namespace op
#endif  // KUIPER_INCLUDE_OP_AWQ_MATMUL_H_
