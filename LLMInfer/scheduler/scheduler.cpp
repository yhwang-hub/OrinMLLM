#include "scheduler.h"
#include <glog/logging.h>
#include <algorithm>
#include <unordered_map>

namespace scheduler {

// ============================ ThreadPool ============================
ThreadPool::ThreadPool(size_t num_threads) {
    if (num_threads == 0) {
        unsigned hc = std::thread::hardware_concurrency();
        num_threads = hc ? std::min<unsigned>(hc, 12u) : 4u;  // 端侧上限保守取 12
    }
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
}

void ThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
}

std::future<void> ThreadPool::submit(std::function<void()> fn) {
    auto promise = std::make_shared<std::promise<void>>();
    std::future<void> fut = promise->get_future();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.emplace([fn = std::move(fn), promise]() mutable {
            fn();
            promise->set_value();
        });
    }
    cv_.notify_one();
    return fut;
}

// ============================ Graph ============================
TaskPtr Graph::add_task(const TaskPtr& task) {
    tasks_.push_back(task);
    return task;
}

EdgePtr Graph::add_edge(const EdgePtr& edge) {
    edges_.push_back(edge);
    return edge;
}

void Graph::connect(const TaskPtr& producer, const EdgePtr& edge, const TaskPtr& consumer) {
    producer->add_output(edge);
    consumer->add_input(edge);
    // kStateful 边(KV cache)不引入执行顺序依赖, 由资源所有权保证安全。
    if (edge->kind() == EdgeKind::kDataFlow) {
        consumer->add_depend(producer);
    }
}

bool Graph::topo_sort(std::vector<Task*>& ordered) const {
    ordered.clear();
    std::unordered_map<Task*, int> indeg;
    std::unordered_map<Task*, std::vector<Task*>> succ;
    for (const auto& t : tasks_) indeg[t.get()] = 0;

    for (const auto& t : tasks_) {
        for (Task* dep : t->depends()) {
            succ[dep].push_back(t.get());
            indeg[t.get()]++;
        }
    }

    std::queue<Task*> q;
    for (auto& kv : indeg) {
        if (kv.second == 0) q.push(kv.first);
    }
    while (!q.empty()) {
        Task* cur = q.front();
        q.pop();
        ordered.push_back(cur);
        for (Task* s : succ[cur]) {
            if (--indeg[s] == 0) q.push(s);
        }
    }
    return ordered.size() == tasks_.size();
}

int64_t Graph::total_workspace_bytes() const {
    int64_t total = 0;
    for (const auto& t : tasks_) total += t->workspace_bytes();
    return total;
}

// ============================ Scheduler ============================
Scheduler::Scheduler(ScheduleMode mode, size_t num_threads)
    : mode_(mode), pool_(std::make_unique<ThreadPool>(num_threads)) {}

common::Status Scheduler::compile(std::shared_ptr<Graph> graph) {
    graph_ = std::move(graph);
    if (!graph_->topo_sort(ordered_)) {
        return common::InternalError("Graph has a cycle; topo sort failed.");
    }
    for (Task* t : ordered_) {
        common::Status st = t->init();
        if (!st) return st;
    }
    compiled_ = true;
    return common::Success();
}

common::Status Scheduler::run() {
    if (!compiled_) return common::InternalError("Scheduler not compiled.");
    switch (mode_) {
        case ScheduleMode::kSequential:
            return run_sequential();
        case ScheduleMode::kParallel:
        case ScheduleMode::kPipeline:
            return run_parallel();
        default:
            return run_sequential();
    }
}

common::Status Scheduler::run_sequential() {
    for (Task* t : ordered_) {
        common::Status st = t->run();
        if (!st) {
            LOG(ERROR) << "Task '" << t->name() << "' failed: " << st.get_err_msg();
            return st;
        }
    }
    return common::Success();
}

// 依赖驱动并行: 入度为 0 的任务并行提交, 完成后递减后继入度。
// 同一设备多流可天然并行; CPU 拷贝任务与 GPU 计算任务重叠。
common::Status Scheduler::run_parallel() {
    std::unordered_map<Task*, std::atomic<int>> indeg;
    std::unordered_map<Task*, std::vector<Task*>> succ;
    for (Task* t : ordered_) indeg[t].store(static_cast<int>(t->depends().size()));
    for (Task* t : ordered_) {
        for (Task* dep : t->depends()) succ[dep].push_back(t);
    }

    std::mutex done_mutex;
    std::condition_variable done_cv;
    int remaining = static_cast<int>(ordered_.size());
    std::atomic<bool> failed{false};

    std::function<void(Task*)> launch = [&](Task* t) {
        pool_->submit([&, t] {
            if (!failed.load()) {
                common::Status st = t->run();
                if (!st) {
                    LOG(ERROR) << "Task '" << t->name() << "' failed: " << st.get_err_msg();
                    failed.store(true);
                }
            }
            std::vector<Task*> ready;
            for (Task* s : succ[t]) {
                if (indeg[s].fetch_sub(1) == 1) ready.push_back(s);
            }
            for (Task* s : ready) launch(s);
            {
                std::lock_guard<std::mutex> lock(done_mutex);
                --remaining;
            }
            done_cv.notify_all();
        });
    };

    for (Task* t : ordered_) {
        if (indeg[t].load() == 0) launch(t);
    }

    std::unique_lock<std::mutex> lock(done_mutex);
    done_cv.wait(lock, [&] { return remaining == 0; });
    return failed.load() ? common::InternalError("Parallel DAG execution failed.")
                         : common::Success();
}

common::Status Scheduler::run_loop(int max_steps, const std::function<bool(int)>& step_cb) {
    for (int step = 0; step < max_steps; ++step) {
        common::Status st = run();
        if (!st) return st;
        if (step_cb && !step_cb(step)) break;  // 回调返回 false 表示停止生成(EOS)
    }
    return common::Success();
}

}  // namespace scheduler
