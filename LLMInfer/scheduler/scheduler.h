#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include "common.h"
#include "task.h"

namespace scheduler {

// ============================================================================
// ThreadPool: 轻量工作线程池(借鉴 CGraph UThreadPool / nndeploy ThreadPool)
//  - 端侧 CPU 核数有限(Orin 12 核), 池大小默认取 min(硬件并发, 上限);
//  - 用于算子级并行(独立分支)与 H2D/D2H 拷贝任务的重叠。
// ============================================================================
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = 0);
    ~ThreadPool();

    std::future<void> submit(std::function<void()> fn);
    size_t size() const { return workers_.size(); }

private:
    void worker_loop();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};

// 调度策略
enum class ScheduleMode : uint8_t {
    kSequential = 0,   // 拓扑序串行(prefill / 单请求 decode 最稳)
    kParallel = 1,     // 依赖驱动并行(算子级并行, 利用多流/多线程)
    kPipeline = 2,     // 流水线(多 step decode / 多请求批处理重叠)
};

// ============================================================================
// Graph: 端侧大模型推理 DAG
// ----------------------------------------------------------------------------
// 典型构图:
//   embed -> layer_0 -> layer_1 -> ... -> layer_N -> norm -> lm_head -> sampler
//   KV cache 作为 kStateful Edge 在相邻 layer 间共享。
// 支持子图(prefill 子图 / decode 子图)与重复执行(decode 循环)。
// ============================================================================
class Graph {
public:
    explicit Graph(std::string name) : name_(std::move(name)) {}

    // 注册节点 / 边
    TaskPtr add_task(const TaskPtr& task);
    EdgePtr add_edge(const EdgePtr& edge);

    // 便捷连边: producer 的某输出流向 consumer 的某输入, 并建立依赖
    void connect(const TaskPtr& producer, const EdgePtr& edge, const TaskPtr& consumer);

    const std::string& name() const { return name_; }
    const std::vector<TaskPtr>& tasks() const { return tasks_; }

    // 拓扑排序(返回 false 表示存在环)
    bool topo_sort(std::vector<Task*>& ordered) const;

    // 统计静态 workspace 峰值(供内存预规划)
    int64_t total_workspace_bytes() const;

private:
    std::string name_;
    std::vector<TaskPtr> tasks_;
    std::vector<EdgePtr> edges_;
};

// ============================================================================
// Scheduler: DAG 执行引擎
// ----------------------------------------------------------------------------
//  - kSequential: 按拓扑序逐个 run(), 适合 prefill 与确定性 decode;
//  - kParallel  : 入度为 0 的任务并行提交线程池, 完成后递减后继入度;
//  - kPipeline  : 跨多次 run() 重叠不同 step 的层执行(decode 吞吐优化)。
// 多设备: Scheduler 持有每设备/每流的句柄, 按 Task.device()/stream() 派发。
// ============================================================================
class Scheduler {
public:
    explicit Scheduler(ScheduleMode mode = ScheduleMode::kSequential, size_t num_threads = 0);

    // 绑定要执行的图(完成拓扑排序与一次性校验)
    common::Status compile(std::shared_ptr<Graph> graph);

    // 执行一次完整前向(prefill 或单 step decode)
    common::Status run();

    // 重复执行(decode 循环), step_cb 在每步后回调(取 logits / 采样 / 决定是否停止)
    common::Status run_loop(int max_steps, const std::function<bool(int)>& step_cb);

    void set_mode(ScheduleMode mode) { mode_ = mode; }
    ScheduleMode mode() const { return mode_; }

private:
    common::Status run_sequential();
    common::Status run_parallel();

    ScheduleMode mode_;
    std::shared_ptr<Graph> graph_;
    std::vector<Task*> ordered_;       // 拓扑序缓存
    std::unique_ptr<ThreadPool> pool_;
    bool compiled_ = false;
};

}  // namespace scheduler
