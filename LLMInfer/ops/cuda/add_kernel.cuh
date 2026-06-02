#ifndef ADD_CU_H
#define ADD_CU_H
#include "tensor/tensor.h"
#include <cuda_fp16.h>

namespace kernel {
// FP32 add kernel
void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream = nullptr);

// FP16 broadcast add bias: adds bias vector to each row of matrix
// matrix: [rows, cols], bias: [cols], output: [rows, cols]
void broadcast_add_bias_fp16_cu(
    const tensor::Tensor& matrix,
    const tensor::Tensor& bias,
    const tensor::Tensor& output,
    int32_t rows,
    int32_t cols,
    void* stream = nullptr);

// Simple vector add in-place: output = a + b (element-wise)
// For FP16 vectors of length n
void add_cu(half* a, const half* b, half* output, int n, void* stream = nullptr);

}  // namespace kernel
#endif  // ADD_CU_H
