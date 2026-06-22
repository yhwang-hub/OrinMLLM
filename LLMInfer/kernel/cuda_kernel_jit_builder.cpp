#include "kernel/base_kernel_jit_builder.h"

#include <fstream>
#include <sstream>

namespace orinmllm::kernel {
namespace {

std::string SanitizeName(const std::string& input) {
	std::string result;
	result.reserve(input.size());
	for (const char ch : input) {
		const bool is_word = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
												 (ch >= '0' && ch <= '9') || ch == '_';
		result.push_back(is_word ? ch : '_');
	}
	return result.empty() ? "kernel" : result;
}

}  // namespace

std::filesystem::path JitSpec::source_path() const { return cache_dir / (name + ".cu"); }

std::filesystem::path JitSpec::library_path() const { return cache_dir / ("lib" + name + ".so"); }

std::filesystem::path JitSpec::build_path() const { return cache_dir / "build.ninja"; }

std::filesystem::path JitSpec::lock_path() const { return cache_dir / (name + ".lock"); }

bool WriteIfDifferent(const std::filesystem::path& path, const std::string& content) {
	if (!path.parent_path().empty()) {
		std::filesystem::create_directories(path.parent_path());
	}
	std::ifstream input(path);
	if (input.good()) {
		std::stringstream buffer;
		buffer << input.rdbuf();
		if (buffer.str() == content) {
			return true;
		}
	}
	std::ofstream output(path, std::ios::trunc);
	if (!output.good()) {
		return false;
	}
	output << content;
	return output.good();
}

bool GenerateNinjaBuild(const JitSpec& spec, std::string* const out_content) {
	if (out_content == nullptr || spec.name.empty()) {
		return false;
	}
	std::ostringstream oss;
	oss << "nvcc = nvcc\n";
	oss << "cflags = -std=c++17 -shared -Xcompiler=-fPIC";
	for (const std::string& flag : spec.cuda_flags) {
		oss << " " << flag;
	}
	for (const std::string& include_dir : spec.include_dirs) {
		oss << " -I" << include_dir;
	}
	oss << "\n";
	oss << "ldflags =";
	for (const std::string& flag : spec.ld_flags) {
		oss << " " << flag;
	}
	oss << "\n";
	oss << "rule build_so\n";
	oss << "  command = $nvcc $cflags $in -o $out $ldflags\n";
	oss << "build " << spec.library_path().string() << ": build_so "
			<< spec.source_path().string() << "\n";
	*out_content = oss.str();
	return true;
}

class CudaKernelJitBuilder final : public BaseKernelJitBuilder {
 public:
	explicit CudaKernelJitBuilder(std::filesystem::path cache_root = {})
			: cache_root_(cache_root.empty() ? std::filesystem::temp_directory_path() / "orinmllm_jit"
																			 : std::move(cache_root)) {}
	~CudaKernelJitBuilder() override = default;

	bool emit_source(const KernelTraits& traits, std::string* const out_source) const override {
		if (out_source == nullptr || traits.op_name.empty()) {
			return false;
		}
		const std::string func_name = SanitizeName(traits.op_name) + "_entry";
		std::ostringstream oss;
		oss << "extern \"C\" __global__ void " << func_name << "() {}\n";
		oss << "extern \"C\" int " << func_name << "_traits() { return " << traits.sm << "; }\n";
		*out_source = oss.str();
		return true;
	}

	bool make_spec(const KernelTraits& traits, JitSpec* const out_spec) const override {
		if (out_spec == nullptr || traits.op_name.empty()) {
			return false;
		}
		JitSpec spec;
		spec.name = SanitizeName(traits.signature());
		spec.cache_dir = cache_root_ / spec.name;
		if (!emit_source(traits, &spec.source_code)) {
			return false;
		}
		for (const int32_t sm : DefaultCudaSms()) {
			spec.cuda_flags.push_back("-gencode=arch=compute_" + std::to_string(sm) +
																",code=sm_" + std::to_string(sm));
		}
		spec.cuda_flags.push_back("-DORINMLLM_JIT=1");
		if (traits.sm >= 90) {
			spec.cuda_flags.push_back("-DORINMLLM_ENABLE_HOPPER=1");
		}
		*out_spec = std::move(spec);
		return true;
	}

 private:
	std::filesystem::path cache_root_;
};

std::unique_ptr<BaseKernelJitBuilder> CreateCudaKernelJitBuilder(
		const std::filesystem::path& cache_root) {
	return std::make_unique<CudaKernelJitBuilder>(cache_root);
}

}  // namespace orinmllm::kernel
