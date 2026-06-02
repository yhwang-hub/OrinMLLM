#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>
#include "common.h"
#include "tensor.h"

namespace scheduler {

class Edge;
class Task;
using TaskPtr = std::shared_ptr<Task>;
using EdgePtr = std::shared_ptr<Edge>;

// ============================================================================
// Edge: 任务之间的数据流通道(借鉴 nndeploy Edge / CGraph GParam)
// ----------------------------------------------------------------------------
// 端侧大模型设计:
//  - 承载 tensor::Tensor(隐藏态 / logits / KV 增量);
//  - kStateful 边用于 KV Cache 等"跨 step 复用"的有状态资源;
//  - 支持队列模式(流水线 decode 时缓冲多 step), 由 Scheduler 决定深度。
// ============================================================================
enum class EdgeKind : uint8_t {
    kDataFlow = 0,   // 普通前向数据(逐 step 覆盖)
    kStateful = 1,   // 有状态资源(KV cache, 持久持有)
};

class Edge {
public:
    explicit Edge(std::string name, EdgeKind kind = EdgeKind::kDataFlow)
        : name_(std::move(name)), kind_(kind) {}

    const std::string& name() const { return name_; }
    EdgeKind kind() const { return kind_; }

    void set(const tensor::Tensor& t) {
        std::lock_guard<std::mutex> lock(mutex_);
        tensor_ = t;
        ready_ = true;
    }
    tensor::Tensor& get() {
        std::lock_guard<std::mutex> lock(mutex_);
        return tensor_;
    }
    bool ready() const { return ready_.load(); }
    void reset() { ready_ = false; }

private:
    std::string name_;
    EdgeKind kind_;
    tensor::Tensor tensor_;
    std::atomic<bool> ready_{false};
    std::mutex mutex_;
};

// ============================================================================
// Task: DAG 中的最小执行单元(借鉴 CGraph GNode / nndeploy Node)
// ----------------------------------------------------------------------------
// 生命周期: init() -> [run() x N] -> deinit()
// 端侧大模型典型 Task: EmbeddingTask / TransformerLayerTask / LMHeadTask /
//                      SamplerTask / VisionEncoderTask。
// ============================================================================
enum class TaskType : uint8_t {
    kCompute = 0,   // GPU/CPU 计算(默认)
    kCopyH2D = 1,   // 主机->设备拷贝(可与计算重叠)
    kCopyD2H = 2,   // 设备->主机拷贝
    kIO = 3,        // 权重加载等 IO
};

class Task : public common::NoCopyable {
public:
    explicit Task(std::string name, TaskType type = TaskType::kCompute)
        : name_(std::move(name)), type_(type) {}
    virtual ~Task() = default;

    // ---- 用户需实现的核心逻辑 ----
    virtual common::Status init() { return common::Success(); }
    virtual common::Status run() = 0;
    virtual common::Status deinit() { return common::Success(); }

    // ---- 内存预算(供 Scheduler 做静态规划, 借鉴 nndeploy getMemorySize) ----
    virtual int64_t workspace_bytes() const { return 0; }

    // ---- 拓扑连接 ----
    void add_input(const EdgePtr& e) { inputs_.push_back(e); }
    void add_output(const EdgePtr& e) { outputs_.push_back(e); }
    void add_depend(const TaskPtr& t) { depends_.insert(t.get()); }

    const std::vector<EdgePtr>& inputs() const { return inputs_; }
    const std::vector<EdgePtr>& outputs() const { return outputs_; }
    const std::unordered_set<Task*>& depends() const { return depends_; }

    const std::string& name() const { return name_; }
    TaskType type() const { return type_; }

    // ---- 设备 / 流绑定(多设备调度) ----
    void set_device(common::DeviceType d) { device_ = d; }
    common::DeviceType device() const { return device_; }
    void set_stream(void* s) { stream_ = s; }
    void* stream() const { return stream_; }

    // ---- 调度运行时状态(由 Scheduler 使用) ----
    std::atomic<int>& pending_deps() { return pending_deps_; }

protected:
    std::string name_;
    TaskType type_ = TaskType::kCompute;
    common::DeviceType device_ = common::DeviceType::kDeviceCUDA;
    void* stream_ = nullptr;

    std::vector<EdgePtr> inputs_;
    std::vector<EdgePtr> outputs_;
    std::unordered_set<Task*> depends_;

    std::atomic<int> pending_deps_{0};
};

// 用 lambda 快速构造 Task(适合算子级小任务)
class LambdaTask : public Task {
public:
    LambdaTask(std::string name, std::function<common::Status()> fn,
               TaskType type = TaskType::kCompute)
        : Task(std::move(name), type), fn_(std::move(fn)) {}
    common::Status run() override { return fn_ ? fn_() : common::Success(); }

private:
    std::function<common::Status()> fn_;
};

}  // namespace scheduler
