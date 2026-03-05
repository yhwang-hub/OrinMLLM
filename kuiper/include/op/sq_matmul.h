//
// SmoothQuant INT8 Quantized Matrix Multiplication Layer
//

#ifndef KUIPER_INCLUDE_OP_SQ_MATMUL_H_
#define KUIPER_INCLUDE_OP_SQ_MATMUL_H_

#include <base/cuda_config.h>
#include "layer.h"

namespace op {

/**
 * SmoothQuant INT8 Per-Tensor Quantized MatMul Layer
 *
 * True INT8 inference path using CUTLASS Tensor Core GEMM:
 *   1. Weights stored as INT8 on disk, uploaded directly to GPU (fast loading)
 *   2. Activation dynamically quantized per-tensor: FP16 → INT8 on GPU
 *   3. CUTLASS INT8 Tensor Core GEMM: INT8 × INT8 → INT32
 *   4. Epilogue fused dequantization: FP16 = alpha * INT32
 *      where alpha = dynamic_input_scale * weight_scale (computed on device)
 *
 * Benefits vs FP16-dequant approach:
 * - ~2× faster GEMM (INT8 Tensor Core throughput)
 * - ~4× less GPU memory for weights (INT8 vs FP16)
 * - ~50% smaller .bin file on disk
 * - Fast model loading (no CPU dequantization)
 *
 * Input:  [batch, in_features] FP16
 * Output: [batch, out_features] FP16
 */
class SQMatmulLayer : public Layer {
 public:
  explicit SQMatmulLayer(base::DeviceType device_type,
                         int32_t in_features,
                         int32_t out_features);

  base::Status check() const override;
  base::Status forward() override;

  // Direct forward with tensors (polymorphic dispatch from base Layer)
  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& output) override;

  // Set SQ weights from raw data pointers (binary file format)
  // qweight_ptr: [out_features, in_features] INT8
  // weight_scale_ptr: FP16 scalar (2 bytes)
  // input_scale_ptr: FP32 scalar (4 bytes)
  void set_sq_weights(const void* qweight_ptr,
                      const void* weight_scale_ptr,
                      const void* input_scale_ptr,
                      base::DeviceType src_device);

  void to_cuda() override;

  // Getters
  int32_t in_features() const { return in_features_; }
  int32_t out_features() const { return out_features_; }
  float weight_scale() const { return weight_scale_; }
  float input_scale() const { return input_scale_; }

  // Access GPU INT8 weight data (for fused kernels that bypass the layer forward)
  const int8_t* qweight_ptr() const { return qweight_.ptr<int8_t>(); }
  const tensor::Tensor& qweight_tensor() const { return qweight_; }

  /**
   * Fused SQ FFN: Gate(W1) + Up(W3) + SwiGLU in a single fused operation.
   * For decode (M=1): quantize input once, then fused W1+W3 GEMV + SwiGLU.
   * Saves 6 kernel launches vs. separate w1, w3 SQ GEMM calls.
   */
  static base::Status fused_ffn_forward(const tensor::Tensor& input,
                                         const tensor::Tensor& output,
                                         const SQMatmulLayer& w1_layer,
                                         const SQMatmulLayer& w3_layer,
                                         cudaStream_t stream);

  /**
   * Quantize input for shared use across multiple GEMV calls (e.g., Q, K, V).
   * Stores quantized INT8 input and input_scale in kernel workspace.
   * Call once, then use forward_preq() for each layer.
   *
   * Saves 6 kernel launches per layer for QKV (216 per decode step for 36 layers).
   */
  static void quantize_input(const tensor::Tensor& input, cudaStream_t stream);

  /**
   * GEMV with pre-quantized input (from quantize_input()).
   * Uses kernel workspace's quantized input and input_scale.
   * Multiply input_scale by per-layer weight_scale inside the kernel.
   */
  static base::Status forward_preq(const tensor::Tensor& output,
                                    const SQMatmulLayer& layer,
                                    cudaStream_t stream);

 private:
  int32_t in_features_ = 0;
  int32_t out_features_ = 0;

  // INT8 quantized weight [out_features, in_features] – lives on GPU after to_cuda()
  tensor::Tensor qweight_;

  // Per-tensor scales
  float weight_scale_ = 0.0f;   // weight quantization scale (from model file)
  float input_scale_ = 0.0f;    // calibration input scale (stored for reference, not used at runtime)
};

}  // namespace op
#endif  // KUIPER_INCLUDE_OP_SQ_MATMUL_H_
