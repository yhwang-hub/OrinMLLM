#include "alloc.h"

namespace memory {

CPUDeviceAllocator::CPUDeviceAllocator()
    : DeviceAllocator(common::DeviceType::kDeviceCPU, common::MemoryType::kMemoryCPU) {}

void* CPUDeviceAllocator::allocate(size_t byte_size) const {
    if (!byte_size) {
        return nullptr;
    }

    void* data = malloc(byte_size);
    return data;
}

void CPUDeviceAllocator::release(void* ptr) const {
    if (ptr) {
        free(ptr);
    }
}

CPUPinnedAllocator::CPUPinnedAllocator()
    : DeviceAllocator(common::DeviceType::kDeviceCPU, common::MemoryType::kMemoryCPUPinned) {}

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

}