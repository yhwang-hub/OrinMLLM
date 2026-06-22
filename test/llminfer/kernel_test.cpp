#include "kernel/kernel_factory.h"
#include "test/llminfer/test_utils.h"

#include <filesystem>
#include <string>

namespace {

int g_called = 0;

void FakeKernel() { ++g_called; }

}  // namespace

int main() {
  using orinmllm::kernel::DataType;
  using orinmllm::kernel::DeviceType;
  using orinmllm::kernel::JitSpec;
  using orinmllm::kernel::KernelFactory;
  using orinmllm::kernel::KernelFunction;
  using orinmllm::kernel::KernelTraits;

  KernelTraits traits;
  traits.op_name = "rms_norm";
  traits.device = DeviceType::kCuda;
  traits.dtype = DataType::kFloat16;
  traits.sm = 87;
  traits.head_dim = 128;
  traits.tile_m = 16;
  traits.tile_n = 32;
  traits.is_fused = true;
  EXPECT_TRUE_OR_EXIT(traits.signature().find("rms_norm") != std::string::npos);

  KernelFactory factory;
  EXPECT_TRUE_OR_EXIT(factory.register_aot(traits, FakeKernel));
  KernelFunction function = nullptr;
  EXPECT_TRUE_OR_EXIT(factory.resolve(traits, &function));
  EXPECT_TRUE_OR_EXIT(function != nullptr);
  function();
  EXPECT_EQ_OR_EXIT(g_called, 1);

  JitSpec spec;
  EXPECT_TRUE_OR_EXIT(factory.prepare_jit(traits, &spec));
  EXPECT_TRUE_OR_EXIT(std::filesystem::exists(spec.source_path()));
  EXPECT_TRUE_OR_EXIT(std::filesystem::exists(spec.build_path()));
  EXPECT_TRUE_OR_EXIT(spec.source_code.find("rms_norm_entry") != std::string::npos);
  EXPECT_TRUE_OR_EXIT(!spec.cuda_flags.empty());
  return EXIT_SUCCESS;
}