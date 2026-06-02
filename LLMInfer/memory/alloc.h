#pragma once

#include <map>
#include <memory>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include "common.h"

namespace memory {

class DeviceAllocator {
public:
    explicit DeviceAllocator(common::DeviceType device_type) : device_type_(device_type) {}
    virtual common::DeviceType device_type() const { return device_type_; }
    virtual void release(void* ptr) const = 0;
    virtual void* allocate(size_t byte_size) const = 0;
    virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                        common::MemcpyKind memcpy_kind = common::MemcpyKind::kMemcpyCPU2CPU,
                        void* stream = nullptr, bool need_sync = false) const;
    virtual void memset_zero(void* ptr, size_t byte_size, void* stream = nullptr, bool need_sync = false) const;

private:
    common::DeviceType device_type_ = common::DeviceType::kDeviceUnknown;
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
    mutable std::map<int, size_t> no_busy_cnt_;
    mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;
    mutable std::map<int, std::vector<CudaMemoryBuffer>> small_buffers_map_;
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

class CPUDeviceAllocatorFactory {
public:
    static std::shared_ptr<CPUDeviceAllocator> create() {
        if (instance_ == nullptr) {
            instance_ = std::make_shared<CPUDeviceAllocator>();
        }
        return instance_;
    }

private:
    static std::shared_ptr<CPUDeviceAllocator> instance_;
};

class CUDADeviceAllocatorFactory {
public:
    static std::shared_ptr<CUDADeviceAllocator> create() {
        if (instance_ == nullptr) {
            instance_ = std::make_shared<CUDADeviceAllocator>();
        }
        return instance_;
    }

private:
    static std::shared_ptr<CUDADeviceAllocator> instance_;
};

}