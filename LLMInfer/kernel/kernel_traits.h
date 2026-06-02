#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kernel {

// ============================================================================
// 目标 GPU 架构(compute capability) -- 端侧多设备
//   Jetson Orin : SM 8.7  (compute_87)
//   L4 / Ada    : SM 8.9  (compute_89)
//   RTX 5070    : SM 12.0 (compute_120, Blackwell, 需 CUDA >= 12.9)
// ============================================================================
struct CudaArch {
    int major = 0;
    int minor = 0;
    // 9.x/10.x/12.x 需要 'a'/'f' 后缀(CUTLASS 扩展指令), <9 无后缀。
    std::string suffix;  // "", "a", "f"

    std::string arch_string() const {  // e.g. "87", "90a", "120f"
        return std::to_string(major) + (suffix.empty() ? std::to_string(minor)
                                                        : std::to_string(minor) + suffix);
    }
};

// 已知端侧设备的架构预设
namespace arch {
inline CudaArch JetsonOrin() { return CudaArch{8, 7, ""}; }
inline CudaArch L4()         { return CudaArch{8, 9, ""}; }
inline CudaArch RTX5070()    { return CudaArch{12, 0, "f"}; }
inline CudaArch Hopper()     { return CudaArch{9, 0, "a"}; }
}  // namespace arch

// 标准化 (major, minor) -> 带正确后缀的 CudaArch (借鉴 flashinfer)
CudaArch NormalizeArch(int major, int minor);

// 探测当前进程可见的所有 GPU 架构(去重)
std::vector<CudaArch> DetectLocalArchs();

// 生成 nvcc -gencode 标志列表
//   - 对每个 arch 产生 -gencode=arch=compute_XX,code=sm_XX
//   - 端侧通常只编译目标设备架构, 减小体积与编译时间
std::vector<std::string> GenGencodeFlags(const std::vector<CudaArch>& archs);

// ============================================================================
// KernelTraits: 描述一个待 JIT 的内核的"类型化"参数
//   - 数据类型(fp16/bf16/fp8/int8) -> 模板展开;
//   - tile 形状等编译期常量 -> 通过宏注入。
// ============================================================================
struct KernelTraits {
    std::string in_dtype = "half";     // 输入数据类型(C++ 名)
    std::string out_dtype = "half";    // 输出数据类型
    std::string acc_dtype = "float";   // 累加器类型
    int cta_m = 0, cta_n = 0, cta_k = 0;  // tile 形状(0 表示不使用)
    std::vector<std::string> defines;     // 额外 -D 宏

    // 生成稳定的、可作为缓存键/文件名的签名
    std::string signature() const;
};

// 将 LLMInfer DataType 名映射到 CUDA C++ 类型名
const char* DTypeToCudaType(const std::string& dtype_name);

}  // namespace kernel
