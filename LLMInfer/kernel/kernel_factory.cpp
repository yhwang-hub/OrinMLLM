#include "kernel_factory.h"
#include <glog/logging.h>

namespace kernel {

KernelFactory& KernelFactory::instance() {
    static KernelFactory inst;
    return inst;
}

std::string KernelFactory::make_key(const std::string& op_name, common::DeviceType device,
                                    const KernelTraits& traits) {
    return op_name + "#dev" + std::to_string(static_cast<int>(device)) + "#" + traits.signature();
}

void KernelFactory::register_builder(const std::string& op_name,
                                     std::shared_ptr<IKernelJitBuilder> builder) {
    std::lock_guard<std::mutex> lock(mutex_);
    builders_[op_name] = std::move(builder);
}

void KernelFactory::register_aot(const std::string& key, void* fn_ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    aot_table_[key] = fn_ptr;
}

void* KernelFactory::get_kernel(const std::string& op_name, common::DeviceType device,
                                const KernelTraits& traits,
                                const std::string& exported_symbol) {
    const std::string key = make_key(op_name, device, traits);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        // 1) AOT 注册表(手写/预编译内核优先)
        auto a = aot_table_.find(key);
        if (a != aot_table_.end()) return a->second;
        a = aot_table_.find(op_name);  // 退化: 设备/traits 无关的通用实现
        if (a != aot_table_.end()) return a->second;

        // 2) JIT 函数缓存
        auto f = jit_fn_cache_.find(key);
        if (f != jit_fn_cache_.end()) return f->second;
    }

    // 3) 触发 JIT 编译(仅 CUDA)
    if (device != common::DeviceType::kDeviceCUDA) {
        LOG(ERROR) << "No kernel for op '" << op_name << "' on non-CUDA device.";
        return nullptr;
    }

    std::shared_ptr<IKernelJitBuilder> builder;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto b = builders_.find(op_name);
        if (b == builders_.end()) {
            LOG(ERROR) << "No JIT builder registered for op '" << op_name << "'.";
            return nullptr;
        }
        builder = b->second;
    }

    // 针对当前可见设备架构编译(端侧通常单一架构)
    auto archs = DetectLocalArchs();
    if (archs.empty()) {
        LOG(ERROR) << "No CUDA device detected for JIT of '" << op_name << "'.";
        return nullptr;
    }

    builder->emit_source(KernelJitCache::instance().cache_root() + "/" + op_name + "_" +
                             traits.signature(),
                         traits);
    JitSpec spec = builder->make_spec(op_name, traits, archs);
    auto module = KernelJitCache::instance().build_and_load(std::move(spec));
    if (!module) return nullptr;

    void* fn = module->get_symbol(exported_symbol);
    if (!fn) return nullptr;

    std::lock_guard<std::mutex> lock(mutex_);
    jit_module_cache_[key] = module;
    jit_fn_cache_[key] = fn;
    return fn;
}

}  // namespace kernel
