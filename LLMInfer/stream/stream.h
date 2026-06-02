#pragma once

#include <cuda_runtime_api.h>
#include <atomic>
#include <memory>
#include <vector>
#include "common.h"

// ============================================================================
// stream 模块 / stream module
// ----------------------------------------------------------------------------
// 目标 / Goal:
//   屏蔽底层平台的异步执行队列差异(当前以 CUDA Stream 为后端), 向上层
//   (scheduler / ops / models) 提供统一的 Stream / Event 抽象。
//   Abstract away platform-specific async execution queues (currently backed by
//   CUDA streams) and expose a unified Stream / Event abstraction to the upper
//   layers (scheduler / ops / models).
//
// 端侧大模型动机 / Edge-LLM motivation:
//   * 计算流 与 H2D / D2H 拷贝流 分离 => decode 阶段 pos 的异步 H2D、argmax
//     结果的异步 D2H 可与 GPU 计算重叠 (copy/compute overlap)。
//   * 多个 worker 流支持算子级并行 (独立 DAG 分支并发执行)。
//   * 句柄可直接喂给 ops/kernel 的 void* stream 形参, 与 CUDA Graph 捕获兼容。
// ============================================================================
namespace stream {

// ---------------------------------------------------------------------------
// Event: 跨流同步 / 计时原语 (CUDA event 封装)
// ---------------------------------------------------------------------------
class Event : public common::NoCopyable {
public:
    // enable_timing=false => 关闭计时位, 同步开销更低 (decode 热路径默认)
    explicit Event(bool enable_timing = false);
    ~Event();

    cudaEvent_t handle() const { return event_; }

    // 在指定流上记录该事件
    void record(cudaStream_t stream);
    // 阻塞主机直到事件完成
    void synchronize();
    // 距离 start 事件的耗时(毫秒); 需两端均 enable_timing=true
    float elapsed_ms(const Event& start) const;

private:
    cudaEvent_t event_ = nullptr;
};

// 流的逻辑角色, 便于 Scheduler 按任务类型派发
enum class StreamRole : uint8_t {
    kCompute = 0,  // GPU 计算
    kCopyH2D = 1,  // Host -> Device 拷贝
    kCopyD2H = 2,  // Device -> Host 拷贝
    kWorker  = 3,  // 算子级并行 worker
};

// ---------------------------------------------------------------------------
// Stream: 平台无关的异步执行队列 (当前 CUDA 后端)
// ---------------------------------------------------------------------------
class Stream : public common::NoCopyable {
public:
    explicit Stream(StreamRole role = StreamRole::kCompute, bool high_priority = false);
    // 包装外部已存在的 cudaStream_t (不持有所有权)
    explicit Stream(cudaStream_t external, StreamRole role = StreamRole::kCompute);
    ~Stream();

    cudaStream_t handle() const { return stream_; }
    void* raw() const { return static_cast<void*>(stream_); }
    StreamRole role() const { return role_; }

    void synchronize();              // 阻塞直到流内全部完成
    bool query();                    // 非阻塞查询是否完成
    void record(Event& event);       // 在本流记录事件
    void wait(const Event& event);   // 让本流等待某事件(跨流依赖)

private:
    cudaStream_t stream_ = nullptr;
    StreamRole role_ = StreamRole::kCompute;
    bool owns_ = true;
};

using StreamPtr = std::shared_ptr<Stream>;

// ---------------------------------------------------------------------------
// StreamManager: 进程级流池 (compute / h2d / d2h / workers)
// ---------------------------------------------------------------------------
class StreamManager : public common::NoCopyable {
public:
    static StreamManager& instance();

    StreamPtr compute_stream() const { return compute_; }
    StreamPtr h2d_stream() const { return h2d_; }
    StreamPtr d2h_stream() const { return d2h_; }

    // 轮询取一个 worker 流, 供算子级并行使用
    StreamPtr next_worker();
    void set_num_workers(size_t n);
    size_t num_workers() const { return workers_.size(); }

    void synchronize_all();

private:
    StreamManager();

    StreamPtr compute_;
    StreamPtr h2d_;
    StreamPtr d2h_;
    std::vector<StreamPtr> workers_;
    std::atomic<size_t> rr_{0};
};

}  // namespace stream
