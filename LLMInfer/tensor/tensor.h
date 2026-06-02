#pragma once

#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <memory>
#include <numeric>
#include <vector>
#include "buffer.h"
#include "common.h"

namespace tensor {

// ============================================================================
// Tensor: 多维张量(数据 + 形状 + 数据类型 + 设备)
// ----------------------------------------------------------------------------
// 端侧推理设计要点:
//  1. 底层数据托管给 memory::Buffer, Tensor 仅描述逻辑视图(dims/dtype);
//  2. need_alloc=false + 外部 ptr 支持权重零拷贝 / CUDA Graph 固定地址;
//  3. to_cpu()/to_cuda() 实现透明的设备迁移(Jetson 上可退化为零拷贝);
//  4. clone() 深拷贝; assign() 共享底层 Buffer(view), 支持 KV cache 切片。
// ============================================================================
class Tensor {
public:
    explicit Tensor() = default;

    explicit Tensor(common::DataType data_type, int32_t dim0, bool need_alloc = false,
                    std::shared_ptr<memory::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

    explicit Tensor(common::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc = false,
                    std::shared_ptr<memory::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

    explicit Tensor(common::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                    bool need_alloc = false,
                    std::shared_ptr<memory::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

    explicit Tensor(common::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                    int32_t dim3, bool need_alloc = false,
                    std::shared_ptr<memory::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

    explicit Tensor(common::DataType data_type, std::vector<int32_t> dims, bool need_alloc = false,
                    std::shared_ptr<memory::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

    void to_cpu();
    void to_cuda(cudaStream_t stream = nullptr);

    bool is_empty() const;

    void init_buffer(std::shared_ptr<memory::DeviceAllocator> alloc, common::DataType data_type,
                     bool need_alloc, void* ptr);

    template <typename T> T* ptr();
    template <typename T> const T* ptr() const;
    template <typename T> T* ptr(int64_t index);
    template <typename T> const T* ptr(int64_t index) const;
    template <typename T> T& index(int64_t offset);
    template <typename T> const T& index(int64_t offset) const;

    void reshape(const std::vector<int32_t>& dims);

    std::shared_ptr<memory::Buffer> get_buffer() const;

    size_t size() const;
    size_t byte_size() const;
    int32_t dims_size() const;
    common::DataType data_type() const;
    int32_t get_dim(int32_t idx) const;
    const std::vector<int32_t>& dims() const;
    std::vector<size_t> strides() const;

    bool assign(std::shared_ptr<memory::Buffer> buffer);
    void reset(common::DataType data_type, const std::vector<int32_t>& dims);

    void set_device_type(common::DeviceType device_type) const;
    common::DeviceType device_type() const;

    bool allocate(std::shared_ptr<memory::DeviceAllocator> allocator, bool need_realloc = false);

    tensor::Tensor clone() const;

private:
    size_t size_ = 0;
    std::vector<int32_t> dims_;
    std::shared_ptr<memory::Buffer> buffer_;
    common::DataType data_type_ = common::DataType::kDataTypeUnknown;
};

// ------------------------- 模板成员实现 -------------------------
template <typename T> T& Tensor::index(int64_t offset) {
    CHECK_GE(offset, 0);
    CHECK_LT(offset, this->size());
    return *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
}

template <typename T> const T& Tensor::index(int64_t offset) const {
    CHECK_GE(offset, 0);
    CHECK_LT(offset, this->size());
    return *(reinterpret_cast<const T*>(buffer_->ptr()) + offset);
}

template <typename T> const T* Tensor::ptr() const {
    if (!buffer_) return nullptr;
    return reinterpret_cast<const T*>(buffer_->ptr());
}

template <typename T> T* Tensor::ptr() {
    if (!buffer_) return nullptr;
    return reinterpret_cast<T*>(buffer_->ptr());
}

template <typename T> T* Tensor::ptr(int64_t index) {
    CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
        << "The data area buffer of this tensor is empty or it points to a null pointer.";
    return reinterpret_cast<T*>(buffer_->ptr()) + index;
}

template <typename T> const T* Tensor::ptr(int64_t index) const {
    CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
        << "The data area buffer of this tensor is empty or it points to a null pointer.";
    return reinterpret_cast<const T*>(buffer_->ptr()) + index;
}

}  // namespace tensor
