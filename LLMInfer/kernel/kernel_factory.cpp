#include "kernel/kernel_factory.h"

namespace orinmllm::kernel {

bool KernelFactory::register_aot(const KernelTraits& traits, KernelFunction function) {
	if (traits.op_name.empty() || function == nullptr) {
		return false;
	}
	std::lock_guard<std::mutex> lock(mutex_);
	functions_[traits.signature()] = function;
	return true;
}

bool KernelFactory::resolve(const KernelTraits& traits, KernelFunction* const out_function) {
	if (out_function == nullptr || traits.op_name.empty()) {
		return false;
	}
	std::lock_guard<std::mutex> lock(mutex_);
	const auto iter = functions_.find(traits.signature());
	if (iter == functions_.end()) {
		*out_function = nullptr;
		return false;
	}
	*out_function = iter->second;
	return *out_function != nullptr;
}

bool KernelFactory::prepare_jit(const KernelTraits& traits, JitSpec* const out_spec) {
	if (out_spec == nullptr || traits.op_name.empty()) {
		return false;
	}
	std::lock_guard<std::mutex> lock(mutex_);
	if (jit_builder_ == nullptr) {
		jit_builder_ = CreateCudaKernelJitBuilder();
	}
	if (!jit_builder_->make_spec(traits, out_spec)) {
		return false;
	}
	std::string ninja_content;
	if (!GenerateNinjaBuild(*out_spec, &ninja_content)) {
		return false;
	}
	return WriteIfDifferent(out_spec->source_path(), out_spec->source_code) &&
				 WriteIfDifferent(out_spec->build_path(), ninja_content);
}

bool KernelFactory::set_jit_builder(std::unique_ptr<BaseKernelJitBuilder> builder) {
	if (builder == nullptr) {
		return false;
	}
	std::lock_guard<std::mutex> lock(mutex_);
	jit_builder_ = std::move(builder);
	return true;
}

std::size_t KernelFactory::cache_size() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return functions_.size();
}

void KernelFactory::clear() {
	std::lock_guard<std::mutex> lock(mutex_);
	functions_.clear();
}

KernelFactory& KernelFactory::instance() {
	static KernelFactory factory;
	return factory;
}

}  // namespace orinmllm::kernel
