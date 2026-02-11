#ifndef FUSED_FFN_KERNEL_CU_CUH
#define FUSED_FFN_KERNEL_CU_CUH
#include "../kernels_interface.h"
#include "tensor/tensor.h"

namespace kernel {

/**
 * Fused Gate-Up-SwiGLU GEMV Kernel
 * 
 * This kernel fuses three operations into one:
 * 1. gate = W1 @ input (gate projection)
 * 2. up = W3 @ input (up projection)  
 * 3. output = silu(gate) * up (SwiGLU activation)
 *
 * Benefits:
 * - Single kernel launch instead of 3 (W1 GEMV + W3 GEMV + SwiGLU)
 * - Input vector is loaded only once from memory
 * - No intermediate memory writes for gate and up results
 * - Estimated ~20-30% speedup for FFN block
 *
 * @param input: Input tensor [M], the hidden state after FFN RMSNorm
 * @param w1: Gate projection weight [K, M]
 * @param w3: Up projection weight [K, M]
 * @param output: Output tensor [K]
 * @param config: CUDA configuration (stream, etc.)
 */
void fused_gate_up_swiglu_kernel_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& w1,      // gate projection
    const tensor::Tensor& w3,      // up projection
    const tensor::Tensor& output,
    const CudaConfig* config = nullptr);

/**
 * FP16 version of Fused Gate-Up-SwiGLU for pure FP16 compute path
 */
void fused_gate_up_swiglu_kernel_cu_fp16(
    const tensor::Tensor& input,
    const tensor::Tensor& w1,
    const tensor::Tensor& w3,
    const tensor::Tensor& output,
    const CudaConfig* config = nullptr);

/**
 * Mixed precision version: FP16 weights with FP32 input/output
 * 
 * This is optimal for models with FP16 weights but FP32 compute path.
 * The kernel reads FP16 weights, converts to FP32 for computation,
 * and writes FP32 output.
 *
 * @param input: FP32 input tensor [M]
 * @param w1: FP16 gate projection weight [K, M]
 * @param w3: FP16 up projection weight [K, M]
 * @param output: FP32 output tensor [K]
 * @param config: CUDA configuration
 */
void fused_gate_up_swiglu_kernel_cu_mixed(
    const tensor::Tensor& input,
    const tensor::Tensor& w1,
    const tensor::Tensor& w3,
    const tensor::Tensor& output,
    const CudaConfig* config = nullptr);

/**
 * Batched version of Fused Gate-Up-SwiGLU for prefill
 * 
 * @param input: Input tensor [batch_size, M]
 * @param w1: Gate projection weight [K, M]
 * @param w3: Up projection weight [K, M]
 * @param output: Output tensor [batch_size, K]
 * @param batch_size: Number of tokens to process
 * @param config: CUDA configuration
 */
void batched_fused_gate_up_swiglu_kernel_cu(
    const tensor::Tensor& input,
    const tensor::Tensor& w1,
    const tensor::Tensor& w3,
    const tensor::Tensor& output,
    int32_t batch_size,
    const CudaConfig* config = nullptr);

}  // namespace kernel

#endif  // FUSED_FFN_KERNEL_CU_CUH
