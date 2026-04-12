//
// FP8 E4M3 Block-Quantized Matrix Multiplication Layer
//

#ifndef KUIPER_INCLUDE_OP_FP8_MATMUL_H_
#define KUIPER_INCLUDE_OP_FP8_MATMUL_H_

#include <base/cuda_config.h>
#include "layer.h"

namespace op {

/**
 * FP8 E4M3 Block-Quantized MatMul Layer
 *
 * Stores weights in FP8 E4M3 with per-block FP16 scale factors:
 * - weight: [out_features, in_features] FP8 E4M3 (1 byte per element)
 * - scale_inv: [ceil(out/block_size), ceil(in/block_size)] FP16
 *
 * Dequantization: fp16_val = fp8_val * scale_inv[row/block_size, col/block_size]
 *
 * Dispatch:
 *   M=1 (decode): FP8 GEMV with on-the-fly block dequant (~2x bandwidth savings)
 *   M>1 (prefill): FP8→FP16 dequant + cuBLAS FP16 Tensor Core GEMM
 *
 * Input:  [batch, in_features] FP16
 * Output: [batch, out_features] FP16
 */
class FP8MatmulLayer : public Layer {
 public:
  explicit FP8MatmulLayer(base::DeviceType device_type,
                          int32_t in_features,
                          int32_t out_features,
                          int32_t block_size = 128);

  base::Status check() const override;
  base::Status forward() override;

  // Direct forward with tensors
  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& output) override;

  // Set FP8 weights from raw data pointers (binary file format)
  // fp8_weight_ptr: [out_features, in_features] FP8 E4M3 (1 byte each)
  // scale_inv_ptr: [scale_rows, scale_cols] FP16 (2 bytes each)
  void set_fp8_weights(const void* fp8_weight_ptr,
                       const void* scale_inv_ptr,
                       int32_t scale_rows,
                       int32_t scale_cols,
                       base::DeviceType src_device);

  void to_cuda() override;

  // Getters
  int32_t in_features() const { return in_features_; }
  int32_t out_features() const { return out_features_; }
  int32_t block_size() const { return block_size_; }
  int32_t scale_rows() const { return scale_rows_; }
  int32_t scale_cols() const { return scale_cols_; }

  const uint8_t* fp8_weight_ptr() const { return fp8_weight_.ptr<uint8_t>(); }
  const tensor::Tensor& fp8_weight_tensor() const { return fp8_weight_; }
  const tensor::Tensor& scale_inv_tensor() const { return scale_inv_; }

 private:
  int32_t in_features_ = 0;
  int32_t out_features_ = 0;
  int32_t block_size_ = 128;
  int32_t scale_rows_ = 0;
  int32_t scale_cols_ = 0;

  // FP8 E4M3 quantized weight [out_features, in_features] – on GPU after to_cuda()
  tensor::Tensor fp8_weight_;

  // Per-block scale_inv [scale_rows, scale_cols] FP16 – on GPU after to_cuda()
  tensor::Tensor scale_inv_;
};

}  // namespace op
#endif  // KUIPER_INCLUDE_OP_FP8_MATMUL_H_
