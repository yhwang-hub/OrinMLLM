/**
 * @file cuda_graph.h
 * @brief CUDA Graph support for optimizing decode phase
 * 
 * This implementation is inspired by llama.cpp's CUDA Graph approach.
 * CUDA Graphs can significantly reduce kernel launch overhead in the decode phase
 * where the same sequence of kernels is executed repeatedly with only minor parameter changes.
 * 
 * Key benefits:
 * - Reduced CPU overhead from kernel launches
 * - Better GPU utilization for small batch sizes (batch_size=1 in decode)
 * - Particularly effective for memory-bound operations
 * 
 * Limitations:
 * - Graph needs to be recaptured when input shapes change
 * - Not suitable for prefill (varying sequence lengths)
 * - Some operations may not be compatible with graph capture
 */

#ifndef KUIPER_INCLUDE_BASE_CUDA_GRAPH_H_
#define KUIPER_INCLUDE_BASE_CUDA_GRAPH_H_

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <cstdint>

namespace base {

/**
 * @brief Properties of a compute node for change detection
 */
struct GraphNodeProperties {
  int32_t pos;              // Current position in sequence
  int32_t layer_idx;        // Layer index (for debugging)
  void* input_ptr;          // Input data pointer
  void* output_ptr;         // Output data pointer
  
  bool operator==(const GraphNodeProperties& other) const {
    // Position changes are expected and handled via parameter updates
    // Only check structural properties
    return input_ptr == other.input_ptr && output_ptr == other.output_ptr;
  }
  
  bool operator!=(const GraphNodeProperties& other) const {
    return !(*this == other);
  }
};

/**
 * @brief CUDA Graph manager for decode phase optimization
 * 
 * Manages the lifecycle of CUDA graphs:
 * 1. Capture: Record kernel launches into a graph
 * 2. Instantiate: Create an executable graph
 * 3. Launch: Execute the graph
 * 4. Update: Modify parameters without recapturing (when possible)
 */
class CudaGraph {
 public:
  CudaGraph() = default;
  
  ~CudaGraph() {
    destroy();
  }
  
  // Non-copyable
  CudaGraph(const CudaGraph&) = delete;
  CudaGraph& operator=(const CudaGraph&) = delete;
  
  // Movable
  CudaGraph(CudaGraph&& other) noexcept {
    graph_ = other.graph_;
    instance_ = other.instance_;
    is_capturing_ = other.is_capturing_;
    is_valid_ = other.is_valid_;
    capture_count_ = other.capture_count_;
    consecutive_update_failures_ = other.consecutive_update_failures_;
    disabled_ = other.disabled_;
    
    other.graph_ = nullptr;
    other.instance_ = nullptr;
    other.is_capturing_ = false;
    other.is_valid_ = false;
  }
  
  CudaGraph& operator=(CudaGraph&& other) noexcept {
    if (this != &other) {
      destroy();
      graph_ = other.graph_;
      instance_ = other.instance_;
      is_capturing_ = other.is_capturing_;
      is_valid_ = other.is_valid_;
      capture_count_ = other.capture_count_;
      consecutive_update_failures_ = other.consecutive_update_failures_;
      disabled_ = other.disabled_;
      
      other.graph_ = nullptr;
      other.instance_ = nullptr;
      other.is_capturing_ = false;
      other.is_valid_ = false;
    }
    return *this;
  }
  
  /**
   * @brief Begin capturing CUDA operations into a graph
   * @param stream The CUDA stream to capture from
   * @return true if capture started successfully
   */
  bool begin_capture(cudaStream_t stream) {
    if (disabled_) {
      return false;
    }
    
    // Destroy any existing graph
    if (graph_ != nullptr) {
      cudaGraphDestroy(graph_);
      graph_ = nullptr;
    }
    if (instance_ != nullptr) {
      cudaGraphExecDestroy(instance_);
      instance_ = nullptr;
    }
    
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) {
      return false;
    }
    
    is_capturing_ = true;
    is_valid_ = false;
    return true;
  }
  
  /**
   * @brief End capturing and instantiate the graph
   * @param stream The CUDA stream being captured
   * @return true if capture ended and graph was instantiated successfully
   */
  bool end_capture(cudaStream_t stream) {
    if (!is_capturing_) {
      return false;
    }
    
    cudaError_t err = cudaStreamEndCapture(stream, &graph_);
    is_capturing_ = false;
    
    if (err != cudaSuccess || graph_ == nullptr) {
      consecutive_update_failures_++;
      if (consecutive_update_failures_ >= kMaxConsecutiveFailures) {
        disabled_ = true;
      }
      return false;
    }
    
    // Instantiate the graph
    err = cudaGraphInstantiate(&instance_, graph_, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
      cudaGraphDestroy(graph_);
      graph_ = nullptr;
      consecutive_update_failures_++;
      if (consecutive_update_failures_ >= kMaxConsecutiveFailures) {
        disabled_ = true;
      }
      return false;
    }
    
    is_valid_ = true;
    capture_count_++;
    consecutive_update_failures_ = 0;
    return true;
  }
  
  /**
   * @brief Launch the captured graph
   * @param stream The CUDA stream to launch on
   * @return true if launch was successful
   */
  bool launch(cudaStream_t stream) {
    if (!is_valid_ || instance_ == nullptr) {
      return false;
    }
    
    cudaError_t err = cudaGraphLaunch(instance_, stream);
    return err == cudaSuccess;
  }
  
  /**
   * @brief Try to update the graph executable with a new graph
   * @param stream The CUDA stream (not currently used, for future API compatibility)
   * @return true if update was successful, false if recapture is needed
   */
  bool try_update(cudaStream_t stream) {
    if (!is_valid_ || instance_ == nullptr || graph_ == nullptr) {
      return false;
    }
    
    // Try to update the executable graph
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t err = cudaGraphExecUpdate(instance_, graph_, &result_info);
    
    if (err == cudaSuccess) {
      return true;
    }
    
    // Update failed, need to recapture
    if (err == cudaErrorGraphExecUpdateFailure) {
      // Clear the error
      cudaGetLastError();
      
      // Re-instantiate
      cudaGraphExecDestroy(instance_);
      instance_ = nullptr;
      
      err = cudaGraphInstantiate(&instance_, graph_, nullptr, nullptr, 0);
      if (err == cudaSuccess) {
        return true;
      }
    }
    
    consecutive_update_failures_++;
    if (consecutive_update_failures_ >= kMaxConsecutiveFailures) {
      disabled_ = true;
    }
    
    return false;
  }
  
  /**
   * @brief Check if the graph is currently valid and can be launched
   */
  bool is_valid() const { return is_valid_ && !disabled_; }
  
  /**
   * @brief Check if graph capture is in progress
   */
  bool is_capturing() const { return is_capturing_; }
  
  /**
   * @brief Check if CUDA graphs have been disabled (too many failures)
   */
  bool is_disabled() const { return disabled_; }
  
  /**
   * @brief Get the number of times the graph has been captured
   */
  int capture_count() const { return capture_count_; }
  
  /**
   * @brief Reset the graph state
   */
  void reset() {
    destroy();
    is_capturing_ = false;
    is_valid_ = false;
    capture_count_ = 0;
    consecutive_update_failures_ = 0;
    // Don't reset disabled_ - if it was disabled due to failures, keep it disabled
  }
  
  /**
   * @brief Force enable CUDA graphs (reset disabled state)
   */
  void force_enable() {
    disabled_ = false;
    consecutive_update_failures_ = 0;
  }

 private:
  void destroy() {
    if (instance_ != nullptr) {
      cudaGraphExecDestroy(instance_);
      instance_ = nullptr;
    }
    if (graph_ != nullptr) {
      cudaGraphDestroy(graph_);
      graph_ = nullptr;
    }
    is_valid_ = false;
  }
  
  static constexpr int kMaxConsecutiveFailures = 3;
  
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t instance_ = nullptr;
  bool is_capturing_ = false;
  bool is_valid_ = false;
  bool disabled_ = false;
  int capture_count_ = 0;
  int consecutive_update_failures_ = 0;
};

/**
 * @brief Context for managing CUDA Graph in decode phase
 */
struct CudaGraphContext {
  std::unique_ptr<CudaGraph> decode_graph;
  
  // Properties for detecting when recapture is needed
  GraphNodeProperties last_properties;
  bool needs_recapture = true;
  
  // Statistics
  int graph_launches = 0;
  int graph_recaptures = 0;
  
  CudaGraphContext() : decode_graph(std::make_unique<CudaGraph>()) {}
  
  /**
   * @brief Check if the graph needs to be recaptured based on property changes
   */
  bool check_recapture_needed(const GraphNodeProperties& current) {
    if (needs_recapture) {
      return true;
    }
    
    // Check if structural properties changed
    if (current != last_properties) {
      needs_recapture = true;
      return true;
    }
    
    return false;
  }
  
  /**
   * @brief Update properties after successful capture
   */
  void update_properties(const GraphNodeProperties& props) {
    last_properties = props;
    needs_recapture = false;
  }
  
  /**
   * @brief Mark that recapture is needed and invalidate the current graph
   */
  void invalidate() {
    needs_recapture = true;
    if (decode_graph) {
      decode_graph->reset();  // 真正使 graph 无效，下次 is_valid() 会返回 false
    }
  }
};

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_CUDA_GRAPH_H_
