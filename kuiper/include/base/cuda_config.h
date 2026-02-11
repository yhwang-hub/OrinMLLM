#ifndef BLAS_HELPER_H
#define BLAS_HELPER_H
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <memory>
#include "base/cuda_graph.h"

namespace kernel {

/**
 * @brief CUDA configuration for model inference
 * 
 * Contains CUDA stream, cuBLAS handle, and optional CUDA Graph context for decode optimization.
 * Also includes pre-allocated FP16 workspace buffers for efficient HGEMM operations.
 */
struct CudaConfig {
  cudaStream_t stream = nullptr;
  
  // cuBLAS handle for optimized GEMM operations
  cublasHandle_t cublas_handle = nullptr;
  
  // CUDA Graph context for decode phase optimization
  std::shared_ptr<base::CudaGraphContext> graph_context = nullptr;
  
  // Whether to use CUDA Graph for decode (can be disabled for debugging)
  bool use_cuda_graph = false;
  
  // Pre-allocated FP16 workspace buffers for HGEMM to avoid per-call cudaMalloc
  // These are sized for Qwen2.5-7B's maximum batch_size * max_dim operations
  __half* fp16_input_workspace = nullptr;   // For FP32->FP16 input conversion
  __half* fp16_output_workspace = nullptr;  // For FP16->FP32 output conversion
  size_t fp16_workspace_size = 0;           // Current allocated size (in elements)
  
  CudaConfig() = default;
  
  ~CudaConfig() {
    // Free FP16 workspace buffers
    if (fp16_input_workspace) {
      cudaFree(fp16_input_workspace);
      fp16_input_workspace = nullptr;
    }
    if (fp16_output_workspace) {
      cudaFree(fp16_output_workspace);
      fp16_output_workspace = nullptr;
    }
    if (cublas_handle) {
      cublasDestroy(cublas_handle);
    }
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }
  
  // Non-copyable due to stream ownership
  CudaConfig(const CudaConfig&) = delete;
  CudaConfig& operator=(const CudaConfig&) = delete;
  
  // Movable
  CudaConfig(CudaConfig&& other) noexcept 
    : stream(other.stream), 
      cublas_handle(other.cublas_handle),
      graph_context(std::move(other.graph_context)),
      use_cuda_graph(other.use_cuda_graph),
      fp16_input_workspace(other.fp16_input_workspace),
      fp16_output_workspace(other.fp16_output_workspace),
      fp16_workspace_size(other.fp16_workspace_size) {
    other.stream = nullptr;
    other.cublas_handle = nullptr;
    other.fp16_input_workspace = nullptr;
    other.fp16_output_workspace = nullptr;
    other.fp16_workspace_size = 0;
  }
  
  CudaConfig& operator=(CudaConfig&& other) noexcept {
    if (this != &other) {
      // Clean up current resources
      if (fp16_input_workspace) cudaFree(fp16_input_workspace);
      if (fp16_output_workspace) cudaFree(fp16_output_workspace);
      if (cublas_handle) cublasDestroy(cublas_handle);
      if (stream) cudaStreamDestroy(stream);
      
      // Move resources
      stream = other.stream;
      cublas_handle = other.cublas_handle;
      graph_context = std::move(other.graph_context);
      use_cuda_graph = other.use_cuda_graph;
      fp16_input_workspace = other.fp16_input_workspace;
      fp16_output_workspace = other.fp16_output_workspace;
      fp16_workspace_size = other.fp16_workspace_size;
      
      // Invalidate other
      other.stream = nullptr;
      other.cublas_handle = nullptr;
      other.fp16_input_workspace = nullptr;
      other.fp16_output_workspace = nullptr;
      other.fp16_workspace_size = 0;
    }
    return *this;
  }
  
  /**
   * @brief Ensure FP16 workspace buffers are allocated and large enough
   * @param input_size Required input buffer size (in elements)
   * @param output_size Required output buffer size (in elements)
   * @return true if workspace is ready, false on allocation failure
   */
  bool ensure_fp16_workspace(size_t input_size, size_t output_size) {
    size_t required_size = std::max(input_size, output_size);
    
    if (required_size <= fp16_workspace_size) {
      return true;  // Already have enough space
    }
    
    // Allocate with some extra margin (1.5x) to reduce future reallocations
    size_t new_size = static_cast<size_t>(required_size * 1.5);
    
    // Free old buffers
    if (fp16_input_workspace) {
      cudaFree(fp16_input_workspace);
      fp16_input_workspace = nullptr;
    }
    if (fp16_output_workspace) {
      cudaFree(fp16_output_workspace);
      fp16_output_workspace = nullptr;
    }
    
    // Allocate new buffers
    cudaError_t err1 = cudaMalloc(&fp16_input_workspace, new_size * sizeof(__half));
    cudaError_t err2 = cudaMalloc(&fp16_output_workspace, new_size * sizeof(__half));
    
    if (err1 != cudaSuccess || err2 != cudaSuccess) {
      // Cleanup on failure
      if (fp16_input_workspace) cudaFree(fp16_input_workspace);
      if (fp16_output_workspace) cudaFree(fp16_output_workspace);
      fp16_input_workspace = nullptr;
      fp16_output_workspace = nullptr;
      fp16_workspace_size = 0;
      return false;
    }
    
    fp16_workspace_size = new_size;
    return true;
  }
  
  /**
   * @brief Initialize the CUDA configuration
   * @param enable_graph Whether to enable CUDA Graph for decode
   */
  void init(bool enable_graph = false) {
    if (!stream) {
      cudaStreamCreate(&stream);
    }
    if (!cublas_handle) {
      cublasCreate(&cublas_handle);
      cublasSetStream(cublas_handle, stream);
      // Use Tensor Core when available (for FP16/TF32)
      cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    }
    use_cuda_graph = enable_graph;
    if (enable_graph && !graph_context) {
      graph_context = std::make_shared<base::CudaGraphContext>();
    }
  }
  
  /**
   * @brief Check if CUDA Graph should be used for the current operation
   */
  bool should_use_graph() const {
    return use_cuda_graph && graph_context && 
           !graph_context->decode_graph->is_disabled();
  }
  
  /**
   * @brief Invalidate the CUDA Graph (force recapture on next decode)
   */
  void invalidate_graph() {
    if (graph_context) {
      graph_context->invalidate();
    }
  }
};

}  // namespace kernel
#endif  // BLAS_HELPER_H
