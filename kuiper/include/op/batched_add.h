#ifndef KUIPER_INCLUDE_OP_BATCHED_ADD_H
#define KUIPER_INCLUDE_OP_BATCHED_ADD_H
#include "base/base.h"
#include "layer.h"
#include <cuda_fp16.h>

namespace op {

// BatchedAddLayer: Element-wise add for any-dim tensors 
// (unlike VecAddLayer which checks for specific 1D dim)
class BatchedAddLayer : public Layer {
 public:
  explicit BatchedAddLayer(base::DeviceType device_type);

  base::Status check() const override;

  base::Status forward() override;
  
  // Forward with tensors directly (avoids set_input/set_output overhead)
  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output1) override;
  
  // Forward with raw pointers (for tensor slice operations)
  base::Status forward_raw(half* a, const half* b, half* output, int n);
};

// BatchedSwiGLULayer: SwiGLU activation for any-dim tensors
class BatchedSwiGLULayer : public Layer {
 public:
  explicit BatchedSwiGLULayer(base::DeviceType device_type);

  base::Status check() const override;

  base::Status forward() override;
  
  // Forward with tensors directly
  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output1) override;
};

// BiasAddLayer: Broadcast bias addition (adds 1D bias to each row of 2D tensor)
class BiasAddLayer : public Layer {
 public:
  explicit BiasAddLayer(base::DeviceType device_type);

  void set_dims(int32_t rows, int32_t cols);
  
  base::Status check() const override;

  base::Status forward() override;
  
 private:
  int32_t rows_ = 0;
  int32_t cols_ = 0;
};

}  // namespace op
#endif  // KUIPER_INCLUDE_OP_BATCHED_ADD_H
