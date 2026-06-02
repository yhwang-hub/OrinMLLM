#include "task.h"

// Task / Edge 的核心逻辑均为头文件内联实现。
// 本文件保留为编译单元, 供未来添加非内联的、跨翻译单元共享的实现
// (例如 Task 的性能埋点、序列化、跨设备资源迁移等)。

namespace scheduler {

// 预留: 全局任务计数(便于调试与 profiling)
namespace {
std::atomic<uint64_t> g_task_seq{0};
}

uint64_t NextTaskSeq() { return g_task_seq.fetch_add(1); }

}  // namespace scheduler
