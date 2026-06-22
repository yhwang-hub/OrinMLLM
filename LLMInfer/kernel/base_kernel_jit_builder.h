#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "kernel/kernel_traits.h"

namespace orinmllm::kernel {

struct JitSpec {
	std::string name;
	std::string source_code;
	std::vector<std::string> cuda_flags;
	std::vector<std::string> include_dirs;
	std::vector<std::string> ld_flags;
	std::filesystem::path cache_dir;

	std::filesystem::path source_path() const;
	std::filesystem::path library_path() const;
	std::filesystem::path build_path() const;
	std::filesystem::path lock_path() const;
};

class BaseKernelJitBuilder {
 public:
	BaseKernelJitBuilder() = default;
	virtual ~BaseKernelJitBuilder() = default;

	BaseKernelJitBuilder(const BaseKernelJitBuilder&) = delete;
	BaseKernelJitBuilder& operator=(const BaseKernelJitBuilder&) = delete;

	virtual bool emit_source(const KernelTraits& traits, std::string* const out_source) const = 0;
	virtual bool make_spec(const KernelTraits& traits, JitSpec* const out_spec) const = 0;
};

bool WriteIfDifferent(const std::filesystem::path& path, const std::string& content);
bool GenerateNinjaBuild(const JitSpec& spec, std::string* const out_content);

}  // namespace orinmllm::kernel
