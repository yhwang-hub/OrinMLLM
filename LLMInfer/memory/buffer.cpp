#include "buffer.h"
#include <glog/logging.h>

namespace memory {

Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator, void* ptr,
               bool use_external)
    : byte_size_(byte_size), ptr_(ptr), use_external_(use_external), allocator_(allocator) {
    if (!ptr_ && allocator_) {
        device_type_ = allocator_->device_type();
        memory_type_ = allocator_->memory_type();
        use_external_ = false;
        ptr_ = allocator_->allocate(byte_size);
    }
    if (allocator_) {
        device_type_ = allocator_->device_type();
        memory_type_ = allocator_->memory_type();
    }
}

Buffer::~Buffer() {
    if (!use_external_ && ptr_ && allocator_) {
        allocator_->release(ptr_);
        ptr_ = nullptr;
    }
}

bool Buffer::allocate() {
    if (allocator_ && byte_size_ != 0) {
        use_external_ = false;
        ptr_ = allocator_->allocate(byte_size_);
        return ptr_ != nullptr;
    }
    return false;
}

void* Buffer::ptr() { return ptr_; }
const void* Buffer::ptr() const { return ptr_; }
size_t Buffer::byte_size() const { return byte_size_; }
std::shared_ptr<DeviceAllocator> Buffer::allocator() const { return allocator_; }
common::DeviceType Buffer::device_type() const { return device_type_; }
void Buffer::set_device_type(common::DeviceType device_type) { device_type_ = device_type; }
common::MemoryType Buffer::memory_type() const { return memory_type_; }
void Buffer::set_memory_type(common::MemoryType memory_type) { memory_type_ = memory_type; }
std::shared_ptr<Buffer> Buffer::get_shared_from_this() { return shared_from_this(); }
bool Buffer::is_external() const { return use_external_; }

static common::MemcpyKind PickMemcpyKind(common::DeviceType src, common::DeviceType dst) {
    if (src == common::DeviceType::kDeviceCPU && dst == common::DeviceType::kDeviceCPU) {
        return common::MemcpyKind::kMemcpyCPU2CPU;
    } else if (src == common::DeviceType::kDeviceCUDA && dst == common::DeviceType::kDeviceCPU) {
        return common::MemcpyKind::kMemcpyCUDA2CPU;
    } else if (src == common::DeviceType::kDeviceCPU && dst == common::DeviceType::kDeviceCUDA) {
        return common::MemcpyKind::kMemcpyCPU2CUDA;
    } else {
        return common::MemcpyKind::kMemcpyCUDA2CUDA;
    }
}

void Buffer::copy_from(const Buffer& buffer) const { copy_from(&buffer); }

void Buffer::copy_from(const Buffer* buffer) const {
    CHECK(allocator_ != nullptr);
    CHECK(buffer != nullptr && buffer->ptr_ != nullptr);

    size_t byte_size = std::min(byte_size_, buffer->byte_size_);
    const common::DeviceType src_device = buffer->device_type();
    const common::DeviceType dst_device = this->device_type();
    CHECK(src_device != common::DeviceType::kDeviceUnknown &&
          dst_device != common::DeviceType::kDeviceUnknown);

    allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size,
                       PickMemcpyKind(src_device, dst_device));
}

}  // namespace memory
