/**
 * Batched GEMV Kernel for Decode Phase Optimization
 * 
 * Inspired by llama.cpp's mmvf.cu implementation with ncols_dst template parameter.
 * This kernel allows processing multiple tokens in a single kernel call, sharing
 * the weight read across all tokens.
 * 
 * Key Optimization: For decode phase, weights (~28GB for Qwen2.5-7B) dominate memory
 * bandwidth. By processing N tokens together, the effective weight read is reduced
 * from 28GB * N to 28GB, improving throughput by up to N times.
 * 
 * Author: Auto-generated based on llama.cpp analysis
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <tensor/tensor.h>

namespace kernel {

/**
 * Launch batched GEMV kernel
 * 
 * Computes: output[batch, k] = sum_m(weight[k, m] * input[batch, m])
 * Weight: [K, M], Input: [batch_size, M], Output: [batch_size, K]
 * 
 * @param input       Input tensor [batch_size, M] (GPU)
 * @param weight      Weight tensor [K, M] (GPU)  
 * @param output      Output tensor [batch_size, K] (GPU)
 * @param batch_size  Number of tokens to process in parallel
 * @param config      CUDA configuration
 */
void batched_gemv_kernel_cu(
    const tensor::Tensor& input,      // [batch_size, M]
    const tensor::Tensor& weight,     // [K, M]
    tensor::Tensor& output,           // [batch_size, K]
    int batch_size,
    const CudaConfig* config);

/**
 * Fused Batched Gate-Up-SwiGLU GEMV Kernel
 * 
 * Combines W1 @ input, W3 @ input, and SwiGLU activation for multiple tokens.
 * output[batch, k] = silu(W1[k,:] @ input[batch,:]) * (W3[k,:] @ input[batch,:])
 * 
 * Memory savings per batch:
 * - Without fusion: 3 kernel calls, weights read 2x
 * - With fusion: 1 kernel call, weights read 1x
 * 
 * @param input       Input tensor [batch_size, M] (GPU)
 * @param w1          Gate projection [K, M] (GPU)
 * @param w3          Up projection [K, M] (GPU)
 * @param output      Output tensor [batch_size, K] (GPU)
 * @param batch_size  Number of tokens to process
 * @param config      CUDA configuration
 */
void batched_fused_gate_up_swiglu_kernel_cu(
    const tensor::Tensor& input,      // [batch_size, M]
    const tensor::Tensor& w1,         // [K, M] gate
    const tensor::Tensor& w3,         // [K, M] up
    tensor::Tensor& output,           // [batch_size, K]
    int batch_size,
    const CudaConfig* config);

} // namespace kernel
