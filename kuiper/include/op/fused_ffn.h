#ifndef KUIPER_INCLUDE_OP_FUSED_FFN_H_
#define KUIPER_INCLUDE_OP_FUSED_FFN_H_
#include "layer.h"
namespace op {

/**
 * @brief Fused FFN Layer (Gate + Up + SwiGLU)
 * 
 * Performs fused gate_up projection with SwiGLU activation:
 *   output = silu(input @ W1^T) * (input @ W3^T)
 * 
 * This is more efficient than separate matmul + swiglu operations.
 * 
 * Input:
 *   - input[0]: input tensor [batch, dim] or [dim]
 *   - input[1]: W1 (gate) weight tensor [hidden_dim, dim]
 *   - input[2]: W3 (up) weight tensor [hidden_dim, dim]
 * Output:
 *   - output[0]: activated output [batch, hidden_dim] or [hidden_dim]
 */
class FusedFFNLayer : public Layer {
 public:
  explicit FusedFFNLayer(base::DeviceType device_type);
  FusedFFNLayer(base::DeviceType device_type, int32_t dim, int32_t hidden_dim, 
                bool use_fp16 = false, bool use_mixed = false);

  base::Status check() const override;
  base::Status forward() override;

  void set_dims(int32_t dim, int32_t hidden_dim) {
    dim_ = dim;
    hidden_dim_ = hidden_dim;
  }
  void set_use_fp16(bool use_fp16) { use_fp16_ = use_fp16; }
  void set_use_mixed(bool use_mixed) { use_mixed_ = use_mixed; }

 private:
  int32_t dim_ = 0;
  int32_t hidden_dim_ = 0;
  bool use_fp16_ = false;
  bool use_mixed_ = false;  // FP16 weights with FP32 activations
};

}  // namespace op
#endif  // KUIPER_INCLUDE_OP_FUSED_FFN_H_
