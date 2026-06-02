#include "tensor.h"
#include <algorithm>

namespace tensor {

static size_t ReduceDimension(const std::vector<int32_t>& dims) {
    if (dims.empty()) return 0;
    size_t size = 1;
    for (int32_t d : dims) size *= static_cast<size_t>(d);
    return size;
}

Tensor::Tensor(common::DataType data_type, int32_t dim0, bool need_alloc,
               std::shared_ptr<memory::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
    dims_.push_back(dim0);
    size_ = dim0;
    if (need_alloc && alloc) {
        allocate(alloc);
    } else if (ptr != nullptr) {
        init_buffer(alloc, data_type_, need_alloc, ptr);
    }
}

Tensor::Tensor(common::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
               std::shared_ptr<memory::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
    dims_ = {dim0, dim1};
    size_ = ReduceDimension(dims_);
    if (need_alloc && alloc) {
        allocate(alloc);
    } else if (ptr != nullptr) {
        init_buffer(alloc, data_type_, need_alloc, ptr);
    }
}

Tensor::Tensor(common::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
               bool need_alloc, std::shared_ptr<memory::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
    dims_ = {dim0, dim1, dim2};
    size_ = ReduceDimension(dims_);
    if (need_alloc && alloc) {
        allocate(alloc);
    } else if (ptr != nullptr) {
        init_buffer(alloc, data_type_, need_alloc, ptr);
    }
}

Tensor::Tensor(common::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
               bool need_alloc, std::shared_ptr<memory::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
    dims_ = {dim0, dim1, dim2, dim3};
    size_ = ReduceDimension(dims_);
    if (need_alloc && alloc) {
        allocate(alloc);
    } else if (ptr != nullptr) {
        init_buffer(alloc, data_type_, need_alloc, ptr);
    }
}

Tensor::Tensor(common::DataType data_type, std::vector<int32_t> dims, bool need_alloc,
               std::shared_ptr<memory::DeviceAllocator> alloc, void* ptr)
    : dims_(std::move(dims)), data_type_(data_type) {
    size_ = ReduceDimension(dims_);
    if (need_alloc && alloc) {
        allocate(alloc);
    } else if (ptr != nullptr) {
        init_buffer(alloc, data_type_, need_alloc, ptr);
    }
}

bool Tensor::allocate(std::shared_ptr<memory::DeviceAllocator> allocator, bool need_realloc) {
    if (!allocator) {
        LOG(ERROR) << "The allocator is null when allocating tensor memory.";
        return false;
    }
    size_t byte = this->byte_size();
    if (!byte) {
        LOG(ERROR) << "The byte size is 0 when allocating tensor memory.";
        return false;
    }
    if (buffer_ && byte <= buffer_->byte_size() && !need_realloc) {
        return true;
    }
    buffer_ = std::make_shared<memory::Buffer>(byte, allocator, nullptr);
    if (!buffer_->ptr()) {
        LOG(ERROR) << "Failed to allocate " << byte << " bytes for tensor.";
        return false;
    }
    return true;
}

void Tensor::init_buffer(std::shared_ptr<memory::DeviceAllocator> alloc,
                         common::DataType data_type, bool need_alloc, void* ptr) {
    if (!alloc && !need_alloc) {
        // 纯外部指针, 设备类型未知时默认按 CPU(上层可 set_device_type 覆盖)
        buffer_ = std::make_shared<memory::Buffer>(byte_size(), nullptr, ptr, true);
    } else {
        allocate(alloc, true);
    }
}

bool Tensor::assign(std::shared_ptr<memory::Buffer> buffer) {
    if (!buffer) {
        LOG(ERROR) << "assign: buffer is null.";
        return false;
    }
    if (buffer_ && buffer_->byte_size() < byte_size()) {
        LOG(ERROR) << "assign: buffer too small.";
        return false;
    }
    buffer_ = buffer;
    return true;
}

void Tensor::reset(common::DataType data_type, const std::vector<int32_t>& dims) {
    data_type_ = data_type;
    dims_ = dims;
    size_ = ReduceDimension(dims_);
    buffer_ = nullptr;
}

void Tensor::reshape(const std::vector<int32_t>& dims) {
    size_t new_size = ReduceDimension(dims);
    if (!buffer_) {
        dims_ = dims;
        size_ = new_size;
        return;
    }
    CHECK_LE(new_size * common::DataTypeSize(data_type_), buffer_->byte_size())
        << "reshape exceeds buffer capacity.";
    dims_ = dims;
    size_ = new_size;
}

void Tensor::to_cpu() {
    CHECK_NE(buffer_, nullptr);
    if (device_type() == common::DeviceType::kDeviceCPU) return;
    size_t byte = this->byte_size();
    auto cpu_alloc = memory::CPUDeviceAllocatorFactory::get_instance();
    auto cpu_buffer = std::make_shared<memory::Buffer>(byte, cpu_alloc);
    cpu_alloc->memcpy(buffer_->ptr(), cpu_buffer->ptr(), byte,
                      common::MemcpyKind::kMemcpyCUDA2CPU);
    buffer_ = cpu_buffer;
}

void Tensor::to_cuda(cudaStream_t stream) {
    CHECK_NE(buffer_, nullptr);
    if (device_type() == common::DeviceType::kDeviceCUDA) return;
    size_t byte = this->byte_size();
    auto cuda_alloc = memory::CUDADeviceAllocatorFactory::get_instance();
    auto cuda_buffer = std::make_shared<memory::Buffer>(byte, cuda_alloc);
    cuda_alloc->memcpy(buffer_->ptr(), cuda_buffer->ptr(), byte,
                       common::MemcpyKind::kMemcpyCPU2CUDA, stream);
    buffer_ = cuda_buffer;
}

tensor::Tensor Tensor::clone() const {
    Tensor t = *this;
    size_t byte = this->byte_size();
    CHECK(buffer_ != nullptr);
    t.buffer_ = std::make_shared<memory::Buffer>(byte, buffer_->allocator());
    t.buffer_->copy_from(buffer_.get());
    return t;
}

bool Tensor::is_empty() const {
    return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
}

std::shared_ptr<memory::Buffer> Tensor::get_buffer() const { return buffer_; }
size_t Tensor::size() const { return size_; }
size_t Tensor::byte_size() const { return size_ * common::DataTypeSize(data_type_); }
int32_t Tensor::dims_size() const { return static_cast<int32_t>(dims_.size()); }
common::DataType Tensor::data_type() const { return data_type_; }

int32_t Tensor::get_dim(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, static_cast<int32_t>(dims_.size()));
    return dims_[idx];
}

const std::vector<int32_t>& Tensor::dims() const { return dims_; }

std::vector<size_t> Tensor::strides() const {
    std::vector<size_t> s;
    if (dims_.empty()) return s;
    s.resize(dims_.size());
    s.back() = 1;
    for (int i = static_cast<int>(dims_.size()) - 2; i >= 0; --i) {
        s[i] = s[i + 1] * static_cast<size_t>(dims_[i + 1]);
    }
    return s;
}

void Tensor::set_device_type(common::DeviceType device_type) const {
    if (buffer_) buffer_->set_device_type(device_type);
}

common::DeviceType Tensor::device_type() const {
    if (!buffer_) return common::DeviceType::kDeviceUnknown;
    return buffer_->device_type();
}

}  // namespace tensor
