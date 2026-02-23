#include "base/paged_kv_cache.h"
#include <glog/logging.h>
#include <algorithm>
#include <numeric>

namespace base {

PagedKVCacheManager::PagedKVCacheManager(
    int32_t num_layers, int32_t page_size, int32_t kv_dim,
    int32_t max_seq_len, DataType dtype, cudaStream_t stream)
    : num_layers_(num_layers),
      page_size_(page_size),
      kv_dim_(kv_dim),
      max_seq_len_(max_seq_len),
      dtype_(dtype),
      stream_(stream) {
  CHECK_GT(page_size, 0) << "Page size must be positive";
  CHECK_EQ(page_size & (page_size - 1), 0) << "Page size must be power of 2, got " << page_size;
  CHECK_GT(kv_dim, 0) << "KV dim must be positive";
  CHECK_GT(max_seq_len, 0) << "Max seq len must be positive";
  CHECK_GT(num_layers, 0) << "Num layers must be positive";

  dtype_size_ = (dtype == DataType::kDataTypeFp16) ? 2 : 4;
  max_blocks_per_seq_ = (max_seq_len + page_size - 1) / page_size;
  // Total blocks for all layers: each layer can use up to max_blocks_per_seq blocks
  num_blocks_ = num_layers * max_blocks_per_seq_;

  LOG(INFO) << "PagedKVCache: page_size=" << page_size
            << ", max_blocks_per_seq=" << max_blocks_per_seq_
            << ", total_blocks=" << num_blocks_
            << ", dtype=" << (dtype == DataType::kDataTypeFp16 ? "FP16" : "FP32");

  // Initialize CPU block table (all unallocated = -1)
  block_table_cpu_.resize(num_layers_ * max_blocks_per_seq_, -1);

  // Initialize free list (all blocks available, allocated in order)
  free_list_.resize(num_blocks_);
  std::iota(free_list_.rbegin(), free_list_.rend(), 0);  // [N-1, N-2, ..., 1, 0] so pop_back gives 0 first

  // Allocate GPU block table
  size_t block_table_bytes = num_layers_ * max_blocks_per_seq_ * sizeof(int32_t);
  cudaMalloc(&block_table_gpu_, block_table_bytes);
  cudaMemset(block_table_gpu_, 0xFF, block_table_bytes);  // -1 in int32

  // Allocate GPU KV pools
  size_t pool_bytes = (size_t)num_blocks_ * page_size * kv_dim * dtype_size_;
  cudaMalloc(&key_pool_gpu_, pool_bytes);
  cudaMalloc(&value_pool_gpu_, pool_bytes);
  cudaMemset(key_pool_gpu_, 0, pool_bytes);
  cudaMemset(value_pool_gpu_, 0, pool_bytes);

  size_t total_mb = (2 * pool_bytes + block_table_bytes) / (1024 * 1024);
  LOG(INFO) << "PagedKVCache: allocated " << total_mb << " MB GPU memory "
            << "(pools: 2x" << pool_bytes / (1024 * 1024) << " MB, "
            << "block_table: " << block_table_bytes / 1024 << " KB)";
}

PagedKVCacheManager::~PagedKVCacheManager() {
  if (block_table_gpu_) cudaFree(block_table_gpu_);
  if (key_pool_gpu_) cudaFree(key_pool_gpu_);
  if (value_pool_gpu_) cudaFree(value_pool_gpu_);
}

int32_t PagedKVCacheManager::allocate_block() {
  CHECK(!free_list_.empty()) << "PagedKVCache: out of free blocks!";
  int32_t block_idx = free_list_.back();
  free_list_.pop_back();
  return block_idx;
}

void PagedKVCacheManager::free_block(int32_t block_idx) {
  free_list_.push_back(block_idx);
}

void PagedKVCacheManager::ensure_allocated_to(int32_t pos) {
  if (pos <= allocated_pos_) return;

  // For each new position that needs a page, allocate across all layers
  for (int32_t p = allocated_pos_ + 1; p <= pos; ++p) {
    int32_t logical_block = p / page_size_;
    int32_t block_offset = p % page_size_;

    // Only need to allocate a new physical block when we enter a new logical block
    // (i.e., block_offset == 0, or this logical block hasn't been allocated yet)
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
      int32_t table_idx = layer * max_blocks_per_seq_ + logical_block;
      if (block_table_cpu_[table_idx] == -1) {
        block_table_cpu_[table_idx] = allocate_block();
      }
    }
  }

  allocated_pos_ = pos;
}

void PagedKVCacheManager::sync_block_table() {
  size_t bytes = num_layers_ * max_blocks_per_seq_ * sizeof(int32_t);
  if (stream_) {
    cudaMemcpyAsync(block_table_gpu_, block_table_cpu_.data(), bytes,
                    cudaMemcpyHostToDevice, stream_);
  } else {
    cudaMemcpy(block_table_gpu_, block_table_cpu_.data(), bytes,
               cudaMemcpyHostToDevice);
  }
}

void PagedKVCacheManager::clear() {
  // Reset block table
  std::fill(block_table_cpu_.begin(), block_table_cpu_.end(), -1);

  // Reset free list
  free_list_.resize(num_blocks_);
  std::iota(free_list_.rbegin(), free_list_.rend(), 0);

  allocated_pos_ = -1;

  // Zero GPU pools
  size_t pool_bytes = (size_t)num_blocks_ * page_size_ * kv_dim_ * dtype_size_;
  if (stream_) {
    cudaMemsetAsync(key_pool_gpu_, 0, pool_bytes, stream_);
    cudaMemsetAsync(value_pool_gpu_, 0, pool_bytes, stream_);
    // Also sync block table
    size_t bt_bytes = num_layers_ * max_blocks_per_seq_ * sizeof(int32_t);
    cudaMemsetAsync(block_table_gpu_, 0xFF, bt_bytes, stream_);
    cudaStreamSynchronize(stream_);
  } else {
    cudaMemset(key_pool_gpu_, 0, pool_bytes);
    cudaMemset(value_pool_gpu_, 0, pool_bytes);
    size_t bt_bytes = num_layers_ * max_blocks_per_seq_ * sizeof(int32_t);
    cudaMemset(block_table_gpu_, 0xFF, bt_bytes);
  }
}

size_t PagedKVCacheManager::get_kv_byte_offset(int32_t layer_idx, int32_t pos) const {
  int32_t logical_block = pos / page_size_;
  int32_t block_offset = pos % page_size_;
  int32_t table_idx = layer_idx * max_blocks_per_seq_ + logical_block;
  int32_t physical_block = block_table_cpu_[table_idx];
  CHECK_GE(physical_block, 0) << "Accessing unallocated page at layer=" << layer_idx
                               << " pos=" << pos << " logical_block=" << logical_block;
  size_t offset = ((size_t)physical_block * page_size_ + block_offset) * kv_dim_ * dtype_size_;
  return offset;
}

}  // namespace base
