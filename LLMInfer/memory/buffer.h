#pragma once

#include <memory>
#include "alloc.h"

namespace memory {

// Buffer owns (or externally references) a contiguous memory region living on a
// specific device / memory type.
//
// Differences vs. the original kuiper base::Buffer:
//   * Tracks the concrete MemoryType (CPU / Pinned / CUDA / ZeroCopy) in addition
//     to DeviceType, so upper layers (Tensor / KV-cache / scheduler) can pick the
//     right transfer path and placement without guessing.
//   * Inherits common::NoCopyable explicitly and exposes enable_shared_from_this
//     publicly so the buffer can hand out shared views of itself.
class Buffer : public common::NoCopyable, public std::enable_shared_from_this<Buffer> {
public:
    explicit Buffer() = default;

    explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                    void* ptr = nullptr, bool use_external = false);

    virtual ~Buffer();

    // Allocate byte_size_ bytes through the bound allocator. Returns false if there
    // is no allocator or the size is zero.
    bool allocate();

    void copy_from(const Buffer& buffer) const;
    void copy_from(const Buffer* buffer) const;

    void* ptr();
    const void* ptr() const;
    size_t byte_size() const;

    std::shared_ptr<DeviceAllocator> allocator() const;

    common::DeviceType device_type() const;
    void set_device_type(common::DeviceType device_type);

    common::MemoryType memory_type() const;
    void set_memory_type(common::MemoryType memory_type);

    std::shared_ptr<Buffer> get_shared_from_this();
    bool is_external() const;

private:
    size_t byte_size_ = 0;
    void* ptr_ = nullptr;
    bool use_external_ = false;
    common::DeviceType device_type_ = common::DeviceType::kDeviceUnknown;
    common::MemoryType memory_type_ = common::MemoryType::kMemoryUnknown;
    std::shared_ptr<DeviceAllocator> allocator_;
};

}  // namespace memory

