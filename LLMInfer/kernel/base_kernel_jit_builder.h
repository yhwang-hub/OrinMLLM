#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "kernel_traits.h"

namespace kernel {

// ============================================================================
// JitSpec: 一次 JIT 编译任务的完整描述(借鉴 flashinfer 的 JitSpec)
//   name        : 唯一名(决定缓存目录/产物名)
//   sources     : 生成/已有的 .cu/.cpp 源文件路径
//   cuda_flags  : nvcc 编译标志(含 -gencode 多架构)
//   include_dirs: 头文件搜索路径
// ============================================================================
struct JitSpec {
    std::string name;
    std::vector<std::string> sources;
    std::vector<std::string> cuda_flags;
    std::vector<std::string> cxx_flags;
    std::vector<std::string> include_dirs;
    std::vector<std::string> ld_flags;
    bool need_device_linking = false;

    // 缓存目录与产物路径(由 KernelJitCache 注入根目录后计算)
    std::string cache_dir;                      // <root>/<name>/
    std::string library_path() const;           // <cache_dir>/<name>.so
    std::string ninja_path() const;             // <cache_dir>/build.ninja
    std::string lock_path() const;              // <cache_dir>/.lock
};

// 已加载的 JIT 模块句柄(dlopen 出来的 .so)
class JitModule {
public:
    explicit JitModule(void* handle, std::string path) : handle_(handle), path_(std::move(path)) {}
    ~JitModule();
    JitModule(const JitModule&) = delete;
    JitModule& operator=(const JitModule&) = delete;

    // 取出导出的函数符号(C 链接)
    template <typename FnPtr>
    FnPtr get_function(const std::string& symbol) {
        return reinterpret_cast<FnPtr>(get_symbol(symbol));
    }

    void* get_symbol(const std::string& symbol);
    bool valid() const { return handle_ != nullptr; }

private:
    void* handle_ = nullptr;
    std::string path_;
};

// ============================================================================
// IKernelJitBuilder: 内核 JIT 构建器接口
//   不同后端(CUDA / 未来 ROCm / CPU SIMD)各自实现:
//     1. emit_source()  : 用模板 + KernelTraits 渲染出源码;
//     2. make_spec()    : 组装 JitSpec(含目标架构 flag)。
// ============================================================================
class IKernelJitBuilder {
public:
    virtual ~IKernelJitBuilder() = default;

    // 渲染源码到 cache_dir, 返回生成的源文件路径列表
    virtual std::vector<std::string> emit_source(const std::string& cache_dir,
                                                 const KernelTraits& traits) = 0;

    // 组装完整 JitSpec(已含 -gencode 多架构标志)
    virtual JitSpec make_spec(const std::string& op_name, const KernelTraits& traits,
                              const std::vector<CudaArch>& archs) = 0;
};

// ============================================================================
// KernelJitCache: 全局编译/加载/缓存管理(进程级单例)
// ----------------------------------------------------------------------------
// 工作流(借鉴 flashinfer build_and_load):
//   build_and_load(spec):
//     1. 命中内存缓存 -> 直接返回 JitModule;
//     2. AOT 产物存在(预编译.so) -> 直接 dlopen;
//     3. 否则: 生成 ninja -> 调 nvcc 编译 -> dlopen -> 入缓存。
//   多进程/多线程通过 lock_path 文件锁避免重复编译。
// ============================================================================
class KernelJitCache {
public:
    static KernelJitCache& instance();

    void set_cache_root(const std::string& root) { cache_root_ = root; }
    const std::string& cache_root() const { return cache_root_; }

    // 完成 spec.cache_dir 等路径计算
    void finalize_paths(JitSpec& spec) const;

    // 编译(若需要)并加载, 返回共享模块句柄
    std::shared_ptr<JitModule> build_and_load(JitSpec spec);

private:
    KernelJitCache();
    bool compile(const JitSpec& spec);   // 生成 ninja + 运行 nvcc
    std::shared_ptr<JitModule> load(const std::string& library_path);

    std::string cache_root_;
    std::mutex mutex_;
    std::map<std::string, std::shared_ptr<JitModule>> loaded_;  // name -> module
};

// 生成 ninja 构建脚本内容(借鉴 flashinfer generate_ninja_build_for_op)
std::string GenerateNinjaBuild(const JitSpec& spec);

// 写文件(内容不同才写, 避免无谓重编译)
bool WriteIfDifferent(const std::string& path, const std::string& content);

}  // namespace kernel
