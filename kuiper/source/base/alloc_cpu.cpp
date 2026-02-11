#include <glog/logging.h>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include "base/alloc.h"

#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define KUIPER_HAVE_POSIX_MEMALIGN
#endif

namespace base {
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {
}

void* CPUDeviceAllocator::allocate(size_t byte_size) const {
  if (!byte_size) {
    return nullptr;
  }
#ifdef KUIPER_HAVE_POSIX_MEMALIGN
  void* data = nullptr;
  const size_t alignment = (byte_size >= size_t(1024)) ? size_t(32) : size_t(16);
  int status = posix_memalign((void**)&data,
                              ((alignment >= sizeof(void*)) ? alignment : sizeof(void*)),
                              byte_size);
  if (status != 0) {
    return nullptr;
  }
  return data;
#else
  void* data = malloc(byte_size);
  return data;
#endif
}

void CPUDeviceAllocator::release(void* ptr) const {
  if (ptr) {
    free(ptr);
  }
}

// Pinned (page-locked) memory allocator implementation
CPUPinnedAllocator::CPUPinnedAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {
}

void* CPUPinnedAllocator::allocate(size_t byte_size) const {
  if (!byte_size) {
    return nullptr;
  }
  void* data = nullptr;
  cudaError_t err = cudaMallocHost(&data, byte_size);
  if (err != cudaSuccess) {
    LOG(ERROR) << "Failed to allocate pinned memory: " << cudaGetErrorString(err);
    return nullptr;
  }
  return data;
}

void CPUPinnedAllocator::release(void* ptr) const {
  if (ptr) {
    cudaFreeHost(ptr);
  }
}

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;
std::shared_ptr<CPUPinnedAllocator> CPUPinnedAllocatorFactory::instance = nullptr;
}  // namespace base