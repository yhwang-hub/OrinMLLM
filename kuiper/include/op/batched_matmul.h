#ifndef KUIPER_INCLUDE_OP_BATCHED_MATMUL_H_
#define KUIPER_INCLUDE_OP_BATCHED_MATMUL_H_
#include "layer.h"
namespace op {

/**
 * @brief Batched Matrix Multiplication Layer
 * 
 * Performs batched matrix multiplication for prefill phase:
 *   output[batch, dim0] = input[batch, dim1] @ weight[dim0, dim1]^T
 * 
 * Supports FP16, FP32, and mixed precision modes.
 * 
 * Input:
 *   - input[0]: input tensor [batch, dim1]
 *   - input[1]: weight tensor [dim0, dim1]
 * Output:
 *   - output[0]: output tensor [batch, dim0]
 */
class BatchedMatmulLayer : public LayerParam {
 public:
  BatchedMatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1);

  base::Status check() const override;
  base::Status forward() override;

  void set_batch_size(int32_t batch_size) { batch_size_ = batch_size; }
  void set_scale(float scale) { scale_ = scale; }

 private:
  int32_t dim0_ = 0;
  int32_t dim1_ = 0;
  int32_t batch_size_ = 1;
  float scale_ = 1.0f;
};

}  // namespace op
#endif  // KUIPER_INCLUDE_OP_BATCHED_MATMUL_H_
