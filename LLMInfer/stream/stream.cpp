#include "stream.h"
#include <glog/logging.h>

namespace stream {

// ============================ Event ============================
Event::Event(bool enable_timing) {
    unsigned flags = enable_timing ? cudaEventDefault : cudaEventDisableTiming;
    cudaError_t st = cudaEventCreateWithFlags(&event_, flags);
    CHECK(st == cudaSuccess) << "cudaEventCreate failed: " << cudaGetErrorString(st);
}

Event::~Event() {
    if (event_) cudaEventDestroy(event_);
}

void Event::record(cudaStream_t stream) {
    cudaError_t st = cudaEventRecord(event_, stream);
    CHECK(st == cudaSuccess) << "cudaEventRecord failed: " << cudaGetErrorString(st);
}

void Event::synchronize() {
    cudaError_t st = cudaEventSynchronize(event_);
    CHECK(st == cudaSuccess) << "cudaEventSynchronize failed: " << cudaGetErrorString(st);
}

float Event::elapsed_ms(const Event& start) const {
    float ms = 0.0f;
    cudaError_t st = cudaEventElapsedTime(&ms, start.event_, event_);
    CHECK(st == cudaSuccess) << "cudaEventElapsedTime failed: " << cudaGetErrorString(st);
    return ms;
}

// ============================ Stream ============================
Stream::Stream(StreamRole role, bool high_priority) : role_(role), owns_(true) {
    if (high_priority) {
        int low = 0, high = 0;
        cudaDeviceGetStreamPriorityRange(&low, &high);
        cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, high);
    } else {
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    }
    CHECK(stream_ != nullptr) << "Failed to create CUDA stream.";
}

Stream::Stream(cudaStream_t external, StreamRole role)
    : stream_(external), role_(role), owns_(false) {}

Stream::~Stream() {
    if (owns_ && stream_) cudaStreamDestroy(stream_);
}

void Stream::synchronize() {
    cudaError_t st = cudaStreamSynchronize(stream_);
    CHECK(st == cudaSuccess) << "cudaStreamSynchronize failed: " << cudaGetErrorString(st);
}

bool Stream::query() { return cudaStreamQuery(stream_) == cudaSuccess; }

void Stream::record(Event& event) { event.record(stream_); }

void Stream::wait(const Event& event) {
    // 让本流在 GPU 侧等待 event, 实现跨流依赖而不阻塞主机
    cudaError_t st = cudaStreamWaitEvent(stream_, event.handle(), 0);
    CHECK(st == cudaSuccess) << "cudaStreamWaitEvent failed: " << cudaGetErrorString(st);
}

// ============================ StreamManager ============================
StreamManager& StreamManager::instance() {
    static StreamManager mgr;
    return mgr;
}

StreamManager::StreamManager() {
    // 计算流取高优先级, 保证 decode 热路径的 GPU 计算优先于拷贝。
    compute_ = std::make_shared<Stream>(StreamRole::kCompute, /*high_priority=*/true);
    h2d_ = std::make_shared<Stream>(StreamRole::kCopyH2D);
    d2h_ = std::make_shared<Stream>(StreamRole::kCopyD2H);
    set_num_workers(2);  // 端侧默认 2 个 worker 流, 可由 pipeline 覆盖
}

void StreamManager::set_num_workers(size_t n) {
    workers_.clear();
    workers_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        workers_.push_back(std::make_shared<Stream>(StreamRole::kWorker));
    }
}

StreamPtr StreamManager::next_worker() {
    if (workers_.empty()) return compute_;
    size_t idx = rr_.fetch_add(1) % workers_.size();
    return workers_[idx];
}

void StreamManager::synchronize_all() {
    compute_->synchronize();
    h2d_->synchronize();
    d2h_->synchronize();
    for (auto& w : workers_) w->synchronize();
}

}  // namespace stream
