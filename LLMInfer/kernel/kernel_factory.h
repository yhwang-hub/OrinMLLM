#pragma once

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "kernel/base_kernel_jit_builder.h"

namespace orinmllm::kernel {

using KernelFunction = void (*)();

class KernelFactory {
 public:
	KernelFactory() = default;
	~KernelFactory() = default;

	bool register_aot(const KernelTraits& traits, KernelFunction function);
	bool resolve(const KernelTraits& traits, KernelFunction* const out_function);
	bool prepare_jit(const KernelTraits& traits, JitSpec* const out_spec);
	bool set_jit_builder(std::unique_ptr<BaseKernelJitBuilder> builder);
	std::size_t cache_size() const;
	void clear();

	static KernelFactory& instance();

 private:
	mutable std::mutex mutex_;
	std::unordered_map<std::string, KernelFunction> functions_;
	std::unique_ptr<BaseKernelJitBuilder> jit_builder_;
};

std::unique_ptr<BaseKernelJitBuilder> CreateCudaKernelJitBuilder(
		const std::filesystem::path& cache_root = {});

}  // namespace orinmllm::kernel
