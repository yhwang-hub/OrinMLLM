#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include "base_kernel_jit_builder.h"
#include "common.h"

namespace kernel {

// ============================================================================
// KernelFactory: 算子 -> 内核实现 的统一入口
// ----------------------------------------------------------------------------
// 端侧推理设计:
//  1. 静态注册的 AOT 内核(已编译进二进制, 如 ops/cuda 下的手写 kernel) 优先;
//  2. 未命中时通过 JIT(IKernelJitBuilder) 针对当前设备架构在线编译并缓存;
//  3. 以 (op_name, device, KernelTraits.signature) 为键缓存函数指针, 二次零开销。
// ============================================================================
class KernelFactory {
public:
    static KernelFactory& instance();

    // 注册一个 JIT builder(某算子的源码模板与编译参数提供者)
    void register_builder(const std::string& op_name,
                          std::shared_ptr<IKernelJitBuilder> builder);

    // 注册一个 AOT 内核函数指针(直接可用, 跳过 JIT)
    void register_aot(const std::string& key, void* fn_ptr);

    // 获取内核函数指针:
    //   - 先查 AOT 注册表;
    //   - 再查 JIT 缓存;
    //   - 否则触发 JIT 编译当前设备架构, 缓存后返回。
    void* get_kernel(const std::string& op_name, common::DeviceType device,
                     const KernelTraits& traits,
                     const std::string& exported_symbol);

    // 强类型便捷封装
    template <typename FnPtr>
    FnPtr get(const std::string& op_name, common::DeviceType device,
              const KernelTraits& traits, const std::string& exported_symbol) {
        return reinterpret_cast<FnPtr>(get_kernel(op_name, device, traits, exported_symbol));
    }

private:
    KernelFactory() = default;

    static std::string make_key(const std::string& op_name, common::DeviceType device,
                                const KernelTraits& traits);

    std::mutex mutex_;
    std::map<std::string, std::shared_ptr<IKernelJitBuilder>> builders_;
    std::map<std::string, void*> aot_table_;
    std::map<std::string, void*> jit_fn_cache_;
    std::map<std::string, std::shared_ptr<JitModule>> jit_module_cache_;
};

}  // namespace kernel
