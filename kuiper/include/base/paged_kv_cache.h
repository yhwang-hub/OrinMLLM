#ifndef KUIPER_INCLUDE_BASE_PAGED_KV_CACHE_H_
#define KUIPER_INCLUDE_BASE_PAGED_KV_CACHE_H_

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <base/base.h>

namespace base {

/**
 * @brief PagedKVCacheManager - Block-based KV cache management for PagedAttention
 *
 * Instead of allocating a contiguous [layer_num, seq_len, kv_dim] buffer,
 * the KV cache is organized as fixed-size pages (blocks).
 *
 * Memory layout:
 *   - Key pool:   [num_blocks, page_size, kv_dim] contiguous GPU buffer
 *   - Value pool:  [num_blocks, page_size, kv_dim] contiguous GPU buffer
 *   - Block table: [num_layers, max_blocks_per_seq] GPU buffer of int32
 *
 * For a token at position `pos` in layer `layer_idx`:
 *   logical_block  = pos / page_size
 *   block_offset   = pos % page_size
 *   physical_block = block_table[layer_idx * max_blocks_per_seq + logical_block]
 *   kv_address     = pool + (physical_block * page_size + block_offset) * kv_dim
 */
class PagedKVCacheManager {
 public:
  static constexpr int32_t kDefaultPageSize = 16;

  /**
   * @param num_layers       Number of transformer layers
   * @param page_size        Tokens per page (must be power of 2)
   * @param kv_dim           KV dimension (kv_head_num * head_size)
   * @param max_seq_len      Maximum sequence length
   * @param dtype            Data type (FP16 or FP32)
   * @param stream           CUDA stream for async operations
   */
  PagedKVCacheManager(int32_t num_layers, int32_t page_size, int32_t kv_dim,
                      int32_t max_seq_len, DataType dtype, cudaStream_t stream = nullptr);

  ~PagedKVCacheManager();

  // Non-copyable
  PagedKVCacheManager(const PagedKVCacheManager&) = delete;
  PagedKVCacheManager& operator=(const PagedKVCacheManager&) = delete;

  /**
   * @brief Ensure all pages are allocated for positions [0, pos] across all layers
   *
   * This allocates new pages from the free list as needed.
   * Must be called before writing data at position `pos`.
   */
  void ensure_allocated_to(int32_t pos);

  /**
   * @brief Sync block table from CPU to GPU
   *
   * Must be called after ensure_allocated_to() and before launching
   * paged attention kernels.
   */
  void sync_block_table();

  /**
   * @brief Clear all page allocations, returning blocks to free list
   *
   * Also zeros the GPU pools for safety.
   */
  void clear();

  /**
   * @brief Get the physical address for writing KV at (layer_idx, pos)
   *
   * Returns byte offsets into key_pool / value_pool.
   * Used by host code for cudaMemcpy operations (e.g., batched_attention_qkv).
   */
  size_t get_kv_byte_offset(int32_t layer_idx, int32_t pos) const;

  /**
   * @brief Get typed pointer into key pool at (layer_idx, pos)
   */
  template<typename T>
  T* key_ptr_at(int32_t layer_idx, int32_t pos) const {
    return reinterpret_cast<T*>(static_cast<char*>(key_pool_gpu_) + get_kv_byte_offset(layer_idx, pos));
  }

  template<typename T>
  T* value_ptr_at(int32_t layer_idx, int32_t pos) const {
    return reinterpret_cast<T*>(static_cast<char*>(value_pool_gpu_) + get_kv_byte_offset(layer_idx, pos));
  }

  // GPU pointers for kernel arguments
  int32_t* block_table_gpu() const { return block_table_gpu_; }
  void* key_pool_gpu() const { return key_pool_gpu_; }
  void* value_pool_gpu() const { return value_pool_gpu_; }

  // Configuration getters
  int32_t page_size() const { return page_size_; }
  int32_t max_blocks_per_seq() const { return max_blocks_per_seq_; }
  int32_t num_layers() const { return num_layers_; }
  int32_t kv_dim() const { return kv_dim_; }
  int32_t num_blocks() const { return num_blocks_; }
  int32_t dtype_size() const { return dtype_size_; }
  DataType dtype() const { return dtype_; }

  // Current allocation state
  int32_t allocated_pos() const { return allocated_pos_; }

 private:
  // Allocate a single physical block from free list
  int32_t allocate_block();

  // Return a block to the free list
  void free_block(int32_t block_idx);

  int32_t num_layers_;
  int32_t page_size_;
  int32_t kv_dim_;
  int32_t max_seq_len_;
  int32_t max_blocks_per_seq_;
  int32_t num_blocks_;
  int32_t dtype_size_;
  DataType dtype_;

  // Tracks the highest position we've allocated pages for
  int32_t allocated_pos_ = -1;

  // CPU-side block table: [num_layers * max_blocks_per_seq]
  // block_table_cpu_[layer * max_blocks_per_seq + logical_block] = physical_block_idx
  // -1 means unallocated
  std::vector<int32_t> block_table_cpu_;

  // Free block list (stack-based allocator)
  std::vector<int32_t> free_list_;

  // GPU-side block table
  int32_t* block_table_gpu_ = nullptr;

  // GPU KV pools
  void* key_pool_gpu_ = nullptr;
  void* value_pool_gpu_ = nullptr;

  cudaStream_t stream_;
};

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_PAGED_KV_CACHE_H_
