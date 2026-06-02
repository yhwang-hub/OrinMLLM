#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include "common.h"

namespace memory {

class DeviceAllocator {
public:
    explicit DeviceAllocator(common::DeviceType device_type,
                             common::MemoryType memory_type = common::MemoryType::kMemoryUnknown)
        : device_type_(device_type), memory_type_(memory_type) {}
    virtual ~DeviceAllocator() = default;
    virtual common::DeviceType device_type() const { return device_type_; }
    virtual common::MemoryType memory_type() const { return memory_type_; }
    virtual void release(void* ptr) const = 0;
    virtual void* allocate(size_t byte_size) const = 0;
    virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                        common::MemcpyKind memcpy_kind = common::MemcpyKind::kMemcpyCPU2CPU,
                        void* stream = nullptr, bool need_sync = false) const;
    virtual void memset_zero(void* ptr, size_t byte_size, void* stream = nullptr, bool need_sync = false) const;

private:
    common::DeviceType device_type_ = common::DeviceType::kDeviceUnknown;
    common::MemoryType memory_type_ = common::MemoryType::kMemoryUnknown;
};

struct CudaMemoryBuffer {
    void* data;
    size_t byte_size;
    bool busy;

    CudaMemoryBuffer() = default;
    CudaMemoryBuffer(void* data, size_t byte_size, bool busy)
        : data(data), byte_size(byte_size), busy(busy) {}
};

class CUDADeviceAllocator : public DeviceAllocator {
public:
    explicit CUDADeviceAllocator();
    void* allocate(size_t byte_size) const override;
    void release(void* ptr) const override;

private:
    mutable std::mutex mutex_;
    mutable std::map<int, size_t> no_busy_cnt_;
    mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;
    mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
};

class CPUDeviceAllocator : public DeviceAllocator {
public:
    explicit CPUDeviceAllocator();
    void* allocate(size_t byte_size) const override;
    void release(void* ptr) const override;
};

class CPUPinnedAllocator : public DeviceAllocator {
public:
    explicit CPUPinnedAllocator();
    void* allocate(size_t byte_size) const override;
    void release(void* ptr) const override;
};

class CUDAZeroCopyAllocator : public DeviceAllocator {
public:
    explicit CUDAZeroCopyAllocator();
    void* allocate(size_t byte_size) const override;
    void release(void* ptr) const override;
};

// Process-wide singleton factories. The allocators keep a memory pool, so they must
// be shared across every Buffer/Tensor that talks to the same device.
class CPUDeviceAllocatorFactory {
public:
    static std::shared_ptr<CPUDeviceAllocator> get_instance() {
        if (instance == nullptr) {
            instance = std::make_shared<CPUDeviceAllocator>();
        }
        return instance;
    }

private:
    static std::shared_ptr<CPUDeviceAllocator> instance;
};

class CPUPinnedAllocatorFactory {
public:
    static std::shared_ptr<CPUPinnedAllocator> get_instance() {
        if (instance == nullptr) {
            instance = std::make_shared<CPUPinnedAllocator>();
        }
        return instance;
    }

private:
    static std::shared_ptr<CPUPinnedAllocator> instance;
};

class CUDADeviceAllocatorFactory {
public:
    static std::shared_ptr<CUDADeviceAllocator> get_instance() {
        if (instance == nullptr) {
            instance = std::make_shared<CUDADeviceAllocator>();
        }
        return instance;
    }

private:
    static std::shared_ptr<CUDADeviceAllocator> instance;
};

class CUDAZeroCopyAllocatorFactory {
public:
    static std::shared_ptr<CUDAZeroCopyAllocator> get_instance() {
        if (instance == nullptr) {
            instance = std::make_shared<CUDAZeroCopyAllocator>();
        }
        return instance;
    }

private:
    static std::shared_ptr<CUDAZeroCopyAllocator> instance;
};

}